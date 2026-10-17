import torch
import argparse
import os
import sys
import torch.optim as optim
import wandb
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.dataset_utils import get_train_val_dataloaders
from utils.model_utils import get_model, get_standard_model
from utils.utils import replace_batchnorm_with_instance_norm, str2bool, set_seed
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule
from train import train_single_epoch
from utils.saver import Saver

def main(args=None, sam_args=None, saver=None):
    requested = str(args['device']).strip().lower()
    if requested.startswith('cuda'):
        if torch.cuda.is_available():
            if ':' in requested: args['device'] = torch.device(requested)
            else:
                device_id = int(args.get('device_id', 0))
                n_gpus = torch.cuda.device_count()
                if device_id >= n_gpus:
                    print(f"[WARNING] device_id={device_id} out of range "
                          f"(only {n_gpus} GPU(s) available). Using cuda:0.")
                    device_id = 0
                args['device'] = torch.device(f"cuda:{device_id}")
            torch.cuda.set_device(args['device'])
            print(f"[INFO] Using device: {args['device']} "
                  f"({torch.cuda.get_device_name(args['device'])})")
        else:
            print(f"[WARNING] '{requested}' requested but CUDA not available. "
                  f"Falling back to CPU.")
            args['device'] = torch.device('cpu')
            print(f"[INFO] Using device: {args['device']}")
    else:
        if requested not in ('cpu',): print(f"[WARNING] Unknown device '{requested}'. Using CPU.")
        args['device'] = torch.device('cpu')
        print(f"[INFO] Using device: {args['device']}")

    if args['use_sam']:
        print(f"[INFO] Loading SAM model: {sam_args['model_type']}")
        try: sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM '{sam_args['model_type']}' from "
                               f"'{sam_args['sam_checkpoint']}': {e}")    

        sam.to(device=args['device'])
        sam.eval()
        for parameter in sam.parameters(): parameter.requires_grad_(False)
        print(f"[INFO] SAM model loaded and moved to {args['device']}; evaluation mode enabled.")
        img_dim = sam.image_encoder.img_size
        transform = ResizeLongestSide(img_dim)
    else:
        sam = None
        img_dim = int(args['Idim'])
        transform = None
        print(f"[INFO] SAM disabled; img_dim = {img_dim}")

    train_dataloader, val_dataloader = get_train_val_dataloaders(args, img_dim)
    img_ch = train_dataloader.dataset[0][args['image_key']].shape[0]
    print(f"[INFO] Image channels: {img_ch}")

    if not args['use_standard_net']:
        model = get_model(args)
        print(f"[INFO] Phase 1 model: {type(model).__name__} (prompt learner)")
    else:
        model = get_standard_model(args, img_ch)
        print(f"[INFO] Phase 2 model: {type(model).__name__} (aggregator)")
    model = model.to(args['device'])
    print(f"[INFO] Model moved to {args['device']}")    

    if args['disable_batchnorm_running_stats']: replace_batchnorm_with_instance_norm(model)

    if not args['use_sam']:
        if not args['segmentor_finetune_backbone']:
            for param in model.parameters(): param.requires_grad = False
            for param in model.segmentor.parameters(): param.requires_grad = True
            params_to_optimize = list(model.segmentor.parameters())
            print(f"[INFO] Phase 2a: optimizing only the aggregator "
                    f"({len(params_to_optimize)} tensors)")
        else:
            for param in model.parameters(): param.requires_grad = True
            params_to_optimize = list(model.parameters())
            print(f"[INFO] Phase 2b: fine-tuning all parameters "
                    f"({len(params_to_optimize)} tensors)")
    else:
        for param in model.parameters(): param.requires_grad = True
        params_to_optimize = list(model.parameters())
        print(f"[INFO] Phase 1: optimizing prompt learner "
                f"({len(params_to_optimize)} tensors)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in params_to_optimize)
    trainable_pct = 100.0 * trainable_params / max(total_params, 1)
    print(f"[INFO] Total parameters    : {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"[INFO] Trainable parameters: {trainable_params:,} "
            f"({trainable_params / 1e6:.2f}M, {trainable_pct:.2f}%)") 

    lr, wd = float(args['learning_rate']), float(args['WD'])
    if args['optim'] == 'Adam':optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=wd)
    elif args['optim'] == 'SGD':optimizer = optim.SGD(params_to_optimize, lr=lr, weight_decay=wd, momentum=float(args['momentum']))
    elif args['optim'] == 'AdamW':optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=wd)
    else: raise ValueError(f"Unknown optimizer: {args['optim']}")

    n_params = sum(p.numel() for p in params_to_optimize)
    print(f"[INFO] Optimizer: {type(optimizer).__name__}, "
            f"lr={lr}, wd={wd}, params={n_params:,}")

    if args['use_scheduler']:
        steps_per_epoch = len(train_dataloader)
        sched_type = args['scheduler']
        gamma = float(args['scheduler_gamma'])
        if sched_type == 'StepLR':
            step_size = steps_per_epoch * int(args['scheduler_step'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif sched_type == 'MultiStepLR':
            n_epochs = int(args['epoches'])
            milestones = [int(n_epochs * 0.3) * steps_per_epoch, int(n_epochs * 0.6) * steps_per_epoch, int(n_epochs * 0.9) * steps_per_epoch]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        elif sched_type == 'ExponentialLR': scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif sched_type == 'ReduceLROnPlateau': scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=int(args['scheduler_patience']))    
        elif sched_type == 'CosineAnnealingLR':
            T_max = int(args['schedCos_Tmax'])
            if T_max <= 10: T_max = int(args['epoches'])
            T_max = T_max * steps_per_epoch
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=float(args['schedCos_eta_min']), last_epoch=int(args['schedCos_last_epoch']))            
        elif sched_type == 'WarmupCosineSchedule':
            warmup_epochs = int(args['warmup_epochs'])
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=steps_per_epoch * warmup_epochs, t_total=steps_per_epoch * int(args['epoches']), cycles=float(args['warmup_cycles']), last_epoch=int(args['schedWarmupCos_last_epoch']))
        elif sched_type == 'WarmupLinearSchedule':
            warmup_epochs = int(args['warmup_epochs'])
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=steps_per_epoch * warmup_epochs, t_total=steps_per_epoch * int(args['epoches']))
        else: raise ValueError(f"Unknown scheduler: {sched_type}")    
        print(f"[INFO] Scheduler: {sched_type} "
              f"(steps_per_epoch={steps_per_epoch})")
    else:
        scheduler = None
        print(f"[INFO] Scheduler: None (constant LR)")

    if args['wandb_watch']: wandb.watch(model, log='all', log_freq=args['wandb_watch_freq'], log_graph=True)

    best = 0
    path_best = os.path.join(saver.path, 'best.csv')
    f_best = open(path_best, 'w', encoding='utf-8')
    f_best.write(f'epoch,{args['best_metric']}\n')
    f_best.flush()
    print(f"[INFO] Tracking best model by '{args['best_metric']}' in: {path_best}")

    cuda_available = torch.cuda.is_available()
    use_amp = bool(args['use_cuda_amp']) and cuda_available
    args['use_amp_resolved'] = use_amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"[INFO] AMP enabled: {use_amp} (requested={args['use_cuda_amp']}, "
          f"cuda_available={cuda_available})")

    for epoch in range(int(args['epoches'])):
        avg_loss, avg_dice, avg_iou= train_single_epoch(train_dataloader, model.train(), sam.eval() if args['use_sam'] else None, optimizer, epoch, args, saver, scheduler, scaler)
        saver.log_loss('train_loss', avg_loss, epoch)
        saver.log_loss('train_dice', avg_dice, epoch)
        saver.log_loss('train_iou', avg_iou, epoch)


if __name__ == '__main__':
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="""
============================================================
            POLYP SEGMENTATION FRAMEWORK
============================================================
Method : SAM-Guided Prompt Learning
Task   : Automated Polyp Segmentation
Input  : Colonoscopy RGB Images
Output : Binary Polyp Segmentation Masks
============================================================""", formatter_class=argparse.RawDescriptionHelpFormatter)
    print(parser.description)

    parser.add_argument('--exp_name', required=True, help='Experiment name (required)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], help='WandB mode')
    parser.add_argument('--wandb_project', default='PolypLite-SAM', help='WandB project name')
    parser.add_argument('--wandb_entity', default=None, help='WandB entity (username or team)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='Wandb tags')
    parser.add_argument('--use_sam', type=str2bool, default=True, help='Use SAM (phase 1) vs aggregator only (phase 2)')
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_b', choices = ['sam_vit_b', 'medsam_vit_b', 'sam_vit_h'], help='SAM checkpoint')
    parser.add_argument('--sam_version', type=str, default='vit_b', choices = ['vit_b', 'vit_h'], help='SAM version')
    parser.add_argument('--device_id', type=int, default=0, help='GPU index')
    parser.add_argument('--device', default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('-task', '--task', default='polyp', help='Task type')
    parser.add_argument('-nW', '--nW', type=int, default=0, help='Number of workers for train loader')
    parser.add_argument('-nW_eval', '--nW_eval', type=int, default=0, help='Number of workers for val loader')
    parser.add_argument('-bs', '--Batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--split_path', type=str, default=os.path.join(PROJECT_DIR, 'data', 'split.json'), help='Path to the JSON split file') 
    parser.add_argument('--n_fold', type=int, default=0, help='Fold index')
    parser.add_argument('--cache_rate', type=float, default=0.0, help='Dataset cache rate (0.0 to 1.0)')
    parser.add_argument('--image_key', type=str, default='image',help='Key for image in the data dict')
    parser.add_argument('--mask_key', type=str, default='mask', help='Key for mask in the data dict')
    parser.add_argument('--use_standard_net', type=str2bool, default=False, help='Use standard segmentor (phase 2) vs prompt learner (phase 1)')
    parser.add_argument('--model_emb', type=str, default='HardNet', help='Prompt learner backbone')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint to resume/init from')
    parser.add_argument('-depth_wise', '--depth_wise', type = str2bool, default=False, help='Use depthwise HarDNet')
    parser.add_argument('-order', '--order', default=68, type=int, help='HarDNet variant: 39 | 68 | 85')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='Use ImageNet pretrained weights')
    parser.add_argument('--net', type=str, default='HardNetSegmentor', help='Segmentor net (phase 2)')
    parser.add_argument('--net_segmentor', type=str, default='PointwiseConv', choices=['PointwiseConv', 'ConvUpsample', 'TransposeConvUpsample', 'UnetLike'], help='Aggregator head')
    parser.add_argument('--segmentor_finetune_backbone', type=str2bool, default=False, help='If we want to train the miniSegmentor finetuning also the backbone')
    parser.add_argument('--upsample_factor_list', type=int, nargs='+', default=[2, 2], help='Upsample factors for ConvUpsample aggregator')
    parser.add_argument('--disable_batchnorm_running_stats', type=str2bool, default=False, help='Disable batchnorm running stats')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, help='Learning rate')
    parser.add_argument('-WD', '--WD', default=1e-4, help='Weight decay')
    parser.add_argument('--optim', type=str, default='AdamW', choices=['Adam', 'SGD', 'AdamW'], help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum (only used with SGD)')
    parser.add_argument('--use_scheduler', type=str2bool, default=False, help='Enable LR scheduler')
    parser.add_argument('--scheduler', type=str, default='StepLR', choices=['StepLR', 'MultiStepLR', 'ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'WarmupCosineSchedule', 'WarmupLinearSchedule'], help='Scheduler type')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Gamma (LR decay factor)')
    parser.add_argument('--scheduler_step', type=int, default=100, help='Scheduler step for StepLR, and first step of MultiStepLR')
    parser.add_argument('-epoches', '--epoches', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--schedCos_Tmax', type=int, default=10, help='T_max for CosineAnnealingLR')
    parser.add_argument('--schedCos_eta_min', type=float, default=0.0001, help='eta_min for CosineAnnealingLR')
    parser.add_argument('--schedCos_last_epoch', type=int, default=-1, help='last_epoch for CosineAnnealingLR')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--warmup_cycles', type=float, default=0.5, help='Warmup cycles')
    parser.add_argument('--schedWarmupCos_last_epoch', type=int, default=-1, help='WarmupCosineSchedule last_epoch (use only if you are resuming a training)')
    parser.add_argument('--wandb_watch', type=str2bool, default=False, help='Enable WandB watch')
    parser.add_argument('--wandb_watch_freq', type=int, default=10, help='Wandb watch frequency')
    parser.add_argument('--best_metric', type=str, default='dice_f', choices=['dice_f', 'dice_t', 'IoU'], help='Best metric')
    parser.add_argument('--use_cuda_amp', type=str2bool, default=True, help='Use mixed precision training')
    parser.add_argument('--include_background', type=str2bool, default=False, help='Include background class in loss')
    parser.add_argument('--theashold_discretize', type=float, default=0.5, help='Threshold for binarization')
    parser.add_argument('--criterion', type=str, default='dice_ce', choices=['dice', 'dice_ce'], help='Loss function')
    parser.add_argument('-Idim', '--Idim', type=int, default=512, help='Input image size (resize resolution)')
    parser.add_argument('--save_train_images', type=str2bool, default=False, help='Save training images to WandB')
    args = vars(parser.parse_args())

    set_seed(args['seed'])

    args['experiment_name'] = args['exp_name']
    output_root = 'results'

    saver = Saver(output_folder=output_root,
                experiment_name=args['experiment_name'],
                wandb_mode=args['wandb_mode'],
                wandb_project=args['wandb_project'],
                wandb_entity=args['wandb_entity'],
                args=args)
    cmd = "python " + " ".join(sys.argv)
    saver.log_cmd(cmd)

    args['exp_folder'] = saver.path
    args['path'] = os.path.join(saver.ckpt_path, 'net_last.pth')
    args['path_best'] = os.path.join(saver.ckpt_path, 'net_best.pth')
    args['vis_folder'] = saver.vis_path
    saver.log_hparams(args)

    if args['use_sam']:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        sam_ckpt = os.path.join(project_dir, 'cp', f"{args['sam_ckpt']}.pth")
        if not os.path.exists(sam_ckpt):
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt}\n"
                                    f"Please download {args['sam_ckpt']}.pth into the cp/ folder.")
        sam_args = {'sam_checkpoint': sam_ckpt,
                'model_type': args['sam_version'],
                'generator_args': {
                'points_per_side': 8,
                'pred_iou_thresh': 0.95,
                'stability_score_thresh': 0.7,
                'crop_n_layers': 0,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 0,
                'point_grids': None,
                'box_nms_thresh': 0.7,},
                'gpu_id': args['device_id'],
                    }
        print(f"[INFO] SAM checkpoint path: {sam_ckpt}")
        print(f"[INFO] SAM Model type : {args['sam_version']}")
    else:
        sam_args = None
        print("[INFO] SAM is disabled.")

    main(args=args, sam_args=sam_args, saver=saver)