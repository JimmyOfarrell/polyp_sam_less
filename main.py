import torch
import os
import sys
import wandb
import argparse
import torch.optim as optim
from utils.saver import Saver
from inference import inference_ds_monai
from train import train_single_epoch, train_single_epoch_monai
from utils.model_utils import get_model, get_standard_model
from utils.dataset_utils import get_train_val_dataloaders
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry
from utils.utils import str2bool, set_seed, disable_batchnorm_running_stats
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule

def main(args=None, sam_args=None, saver=None):
    requested = str(args['device']).strip()
    if requested.startswith('cuda'):
        if torch.cuda.is_available():
            if ':' in requested: args['device'] = torch.device(requested)
            else: args['device'] = torch.device("cuda:"+str(args['device_id']))    
        else:
            print(f"[WARNING] '{requested}' requested but CUDA not available. Falling back to CPU.")
            args['device'] = torch.device('cpu')
    else: args['device'] = torch.device('cpu')
    print(f"[INFO] Using device: {args['device']}")            

    
    if args['use_sam']:
        sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        sam.to(device=args['device'])
        img_dim = sam.image_encoder.img_size
        if args['disable_batchnorm_running_stats']: disable_batchnorm_running_stats(sam)
    else:
        sam = None
        img_dim = int(args['Idim'])
    
    train_dataloader, val_dataloader = get_train_val_dataloaders(args, img_dim)
    img_ch = train_dataloader.dataset[0][args['image_key']].shape[0]
    
    if not args['use_standard_net']: model = get_model(args, sam, img_ch)
    else: model = get_standard_model(args, img_ch)
        
    if args['disable_batchnorm_running_stats']: disable_batchnorm_running_stats(model)
    
    if not args['use_sam']:
        if not args['segmentor_finetune_backbone']:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.segmentor.parameters():
                param.requires_grad = True
            params_to_optimize = list(model.segmentor.parameters())
        else:
            for param in model.parameters():
                param.requires_grad = True
            params_to_optimize = list(model.parameters())
    else:
        # If using SAM, optimize all model parameters
        params_to_optimize = list(model.parameters())

    if args['optim']=='Adam':
        optimizer = optim.Adam(params_to_optimize, lr=float(args['learning_rate']), weight_decay=float(args['WD']))
    elif args['optim']=='SGD':
        optimizer = optim.SGD(params_to_optimize, lr=float(args['learning_rate']), weight_decay=float(args['WD']), momentum=float(args['momentum']))
    elif args['optim']=='AdamW':
        optimizer = optim.AdamW(params_to_optimize, lr=float(args['learning_rate']), weight_decay=float(args['WD']))
    else: raise ValueError(f"Unknown optimizer: {args['optim']}")

    if args['use_scheduler']:
        steps_per_epoch = len(train_dataloader)
        if args['scheduler']=='StepLR':
            step_size = steps_per_epoch * args['scheduler_step']
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=args['scheduler_gamma'])
        elif args['scheduler']== 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=args['scheduler_gamma'])
        elif args['scheduler']== 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args['scheduler_gamma'])
        elif args['scheduler']== 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args['scheduler_gamma'], patience=args['scheduler_patience'], verbose=True)
        elif args['scheduler']== 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['schedCos_Tmax'], eta_min=args['schedCos_eta_min'], last_epoch=args['schedCos_last_epoch'])            
        elif args['scheduler']== 'WarmupCosineSchedule':
            warmup_epochs = args['warmup_epochs']
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=steps_per_epoch * warmup_epochs, t_total=steps_per_epoch * args['epoches'], cycles=args['warmup_cycles'], last_epoch=args['schedWarmupCos_last_epoch'])
            print('Using WarmupCosineSchedule with {} warmup epochs, {} cycles, {} steps per epoch'.format(warmup_epochs, args['warmup_cycles'], steps_per_epoch))
        elif args['scheduler']== 'WarmupLinearSchedule':
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=args['epoches'])
    else:
        scheduler = None

    
    if args['wandb_watch']: wandb.watch(model, log='all', log_freq=args['wandb_watch_freq'], log_graph=True)
    
    best = 0
    path_best = os.path.join(saver.path,'best.csv')
    f_best = open(path_best, 'w')

    scaler = torch.cuda.amp.GradScaler(enabled=args['use_cuda_amp'])
    
    for epoch in range(int(args['epoches'])):
        if args['task'] in ['pancreas', 'spleen', 'prostate', 'brats', 'mslesseg']:
            avg_loss, avg_dice, avg_iou= train_single_epoch_monai(train_dataloader, model.train(), sam.eval() if args['use_sam'] else None, optimizer, transform, epoch, args, saver, scheduler, scaler)
        else:
            avg_loss= train_single_epoch(train_dataloader, model.train(), sam.eval(), optimizer, epoch, args)
        saver.log_loss('train_loss', avg_loss, epoch)
        saver.log_loss('train_dice', avg_dice, epoch)
        saver.log_loss('train_iou', avg_iou, epoch)
        with torch.no_grad():
            if args['task'] in ['pancreas', 'spleen', 'prostate', 'brats', 'mslesseg']:
                dice_true, dice_false, IoU_val, val_loss = inference_ds_monai(val_dataloader, model.eval(), sam, transform, epoch, args, saver)
                saver.log_loss('val_loss', val_loss, epoch)
                saver.log_loss('IoU', IoU_val, epoch)
                saver.log_loss('Dice/include_true', dice_true, epoch)
                saver.log_loss('Dice/include_false', dice_false, epoch)
                
            else:
                dice, IoU_val = inference_ds(val_dataloader, model.eval(), sam, transform, epoch, args)
                saver.log_loss('IoU', IoU_val, epoch)
                saver.log_loss('Dice', dice, epoch)
            if args['best_metric']=='dice_f':
                metric = dice_false
            elif args['best_metric']=='dice_t':
                metric = dice_true
            elif args['best_metric']=='IoU':
                metric = IoU_val
            else:
                raise ValueError('Best metric not recognized')
            if metric > best:
                torch.save(model.state_dict(), args['path_best'])
                best = metric
                print('best results: ' + str(best))
                f_best.write(str(epoch) + ',' + str(best) + '\n')
                f_best.flush()
        
        if epoch % int(args['save_every']) == 0 or epoch == int(args['epoches'])-1:
            saver.save_model(model, 'net_last', epoch)
        
    f_best.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Polyp Segmentation with SAM-guided Prompt Learning')
    # ============================================================
    # 1) CORE / TASK
    # ============================================================
    parser.add_argument('--exp_name', required=True, help='Experiment name (required)')
    parser.add_argument('-task', '--task', default='polyp', help='Task type: polyp | mslesseg | pancreas | ...', required=False)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--split_path', type=str, default='data/split.json', help='Path to the JSON split file')
    parser.add_argument('--data_dir', type=str, default='/data', help='Path to the dataset directory', required=False)
    # ============================================================
    # 2) IMAGE / TASK SPECIFICATION
    # ============================================================
    parser.add_argument('--image_key', type=str, default='image',help='Key for image in the data dict')
    parser.add_argument('--mask_key', type=str, default='mask', help='Key for mask in the data dict')
    parser.add_argument('--out_channels', type=int, default=1, help='Output channels')
    parser.add_argument('-order', '--order', default=68, type=int, help='HarDNet variant: 39 | 68 | 85', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', type = str2bool, default=False, help='Use depthwise HarDNet', required=False)
    parser.add_argument('-Idim', '--Idim', default=1024, help='Input image size (resize resolution)', required=False)
    parser.add_argument('--num_slices', type=int, default=1, help='Number of slices')
    # ============================================================
    # 3) TRAINING HYPERPARAMETERS
    # ============================================================
    parser.add_argument('-lr', '--learning_rate', default=1e-4, help='Learning rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=2, help='Batch size', required=False)
    parser.add_argument('-epoches', '--epoches', type=int, default=150, help='Number of training epochs', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='Weight decay', required=False)
    parser.add_argument('--optim', type=str, default='AdamW', choices=['Adam', 'SGD', 'AdamW'], help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum (only used with SGD)')
    parser.add_argument('--acc_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--criterion', type=str, default='dice_ce', choices=['dice', 'dice_ce'], help='Loss function')
    parser.add_argument('--include_background', type=str2bool, default=False, help='Include background class in loss')
    parser.add_argument('--theashold_discretize', type=float, default=0.5, help='Threshold for binarization')
    parser.add_argument('--save_every', type=int, default=10, help='Save model every N epochs')
    # ============================================================
    # 4) SCHEDULER
    # ============================================================
    parser.add_argument('--use_scheduler', type=str2bool, default=False, help='Enable LR scheduler')
    parser.add_argument('--scheduler', type=str, default='StepLR', choices=['StepLR', 'MultiStepLR', 'ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'WarmupCosineSchedule', 'WarmupLinearSchedule'], help='Scheduler type')
    parser.add_argument('--scheduler_step', type=int, default=100, help='Scheduler step for StepLR, and first step of MultiStepLR')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Gamma (LR decay factor)')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience for ReduceLROnPlateau')
    parser.add_argument('--schedCos_Tmax', type=int, default=10, help='T_max for CosineAnnealingLR')
    parser.add_argument('--schedCos_eta_min', type=float, default=0.0001, help='eta_min for CosineAnnealingLR')
    parser.add_argument('--schedCos_last_epoch', type=int, default=-1, help='last_epoch for CosineAnnealingLR')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--warmup_cycles', type=float, default=0.5, help='Warmup cycles')
    parser.add_argument('--schedWarmupCos_last_epoch', type=int, default=-1, help='WarmupCosineSchedule last_epoch (use only if you are resuming a training)')
    # ============================================================
    # 5) DATALOADER
    # ============================================================
    parser.add_argument('-nW', '--nW', default=0, help='Number of workers for train loader', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='Number of workers for val loader', required=False)    
    parser.add_argument('--cache_rate', type=float, default=0.0, help='Dataset cache rate (0.0 to 1.0)')
    parser.add_argument('--n_fold', type=int, default=0, help='Fold index')
    parser.add_argument('--train_batch_size', type=int, default=8, help='[legacy] duplicate of Batch_size')
    # ============================================================
    # 6) MODEL / ARCHITECTURE
    # ============================================================
    parser.add_argument('--model_emb', type=str, default='HardNet', help='Prompt learner backbone')
    parser.add_argument('--net', type=str, default='HardNetSegmentor', help='Segmentor net (phase 2)')
    parser.add_argument('--net_segmentor', type=str, default='PointwiseConv', choices=['PointwiseConv', 'ConvUpsample', 'TransposeConvUpsample', 'UnetLike'], help='Aggregator head')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='Use ImageNet pretrained weights')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint to resume/init from')
    parser.add_argument('--upsample_factor_list', type=int, nargs='+', default=[2, 2], help='Upsample factors for ConvUpsample aggregator')
    parser.add_argument('--decoder_channels', type=int, nargs='+', default=[], help='Decoder channels for UnetLike segmentor/aggregator')
    parser.add_argument('--remove_maxpool', type=str2bool, default=False, help='Remove first maxpool when ResNet is used as embedding model')
    parser.add_argument('--segmentor_finetune_backbone', type=str2bool, default=False, help='If we want to train the miniSegmentor finetuning also the backbone')
    parser.add_argument('--disable_batchnorm_running_stats', type=str2bool, default=False, help='Disable batchnorm running stats')
    # ============================================================
    # 7) SAM
    # ============================================================
    parser.add_argument('--use_sam', type=str2bool, default=True, help='Use SAM (phase 1) vs aggregator only (phase 2)')
    parser.add_argument('--use_standard_net', type=str2bool, default=False, help='Use standard segmentor (phase 2) vs prompt learner (phase 1)')
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_b', choices = ['sam_vit_b', 'medsam_vit_b', 'sam_vit_h'], help='SAM checkpoint')
    parser.add_argument('--sam_version', type=str, default='vit_b', choices = ['vit_b', 'vit_h'], help='SAM version')
    parser.add_argument('--sam_zeroshot', type=str2bool, default=False, help='SAM zero shot')
    # ============================================================
    # 8) DEVICE / PERFORMANCE
    # ============================================================
    parser.add_argument('--device', default='cuda', help="Device: 'cuda' or 'cpu'", required=False)
    parser.add_argument('--device_id', default=0, help='GPU index', required=False)
    parser.add_argument('--use_cuda_amp', type=str2bool, default=True, help='Use mixed precision training')
    # ============================================================
    # 9) LOGGING — WandB
    # ============================================================
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], help='wandb mode', required=False)
    parser.add_argument('--wandb_project', default='polyp_sam_less', help='wandb project name', required=False)
    parser.add_argument('--wandb_entity', default=None, help='WandB entity (username or team)', required=False)
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='Wandb tags')
    parser.add_argument('--wandb_watch', type=str2bool, default=False, help='Enable WandB watch')
    parser.add_argument('--wandb_watch_freq', type=int, default=10, help='Wandb watch frequency')
    # ============================================================
    # 10) VISUALIZATION / SAVING
    # ============================================================
    parser.add_argument('--save_train_images', type=str2bool, default=False, help='Save training images to WandB')
    parser.add_argument('--save_val_images', type=str2bool, default=False, help='Save validation images to WandB')
    # ============================================================
    # 11) EVALUATION
    # ============================================================
    parser.add_argument('--best_metric', type=str, default='dice_f', choices=['dice_f', 'dice_t', 'IoU'], help='Best metric')

    # ============================================================
    # Parse args, seed, and set up experiment
    # ============================================================
    args = vars(parser.parse_args())
    set_seed(args['seed'])

    # Experiment name and folder
    args['experiment_name'] = args['exp_name']
    output_root = 'results'
    args['exp_folder'] = os.path.join(output_root, args['experiment_name'])

    # Create Saver (creates timestamped folder, initializes WandB)
    saver = Saver(
        output_folder=output_root,
        experiment_name=args['experiment_name'],
        wandb_mode=args['wandb_mode'],
        wandb_project=args['wandb_project'],
        wandb_entity=args['wandb_entity'],
        args=args,
    )

    # Log hyperparameters and command
    saver.log_hparams(args)
    cmd = "python " + " ".join(sys.argv)
    saver.log_cmd(cmd)

    # Register paths for training / inference
    args['path']       = os.path.join(saver.path, 'net_last.pth')
    args['path_best']  = os.path.join(saver.path, 'net_best.pth')
    args['vis_folder'] = os.path.join(saver.path, 'vis')

    # ============================================================
    # SAM arguments
    # ============================================================
    if args['use_sam']:
        sam_ckpt = os.path.join('cp', f"{args['sam_ckpt']}.pth")

        if not os.path.exists(sam_ckpt):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {sam_ckpt}\n"
                f"Please download {args['sam_ckpt']}.pth into the cp/ folder."
            )

        sam_args = {
            'sam_checkpoint': sam_ckpt,
            'model_type': args['sam_version'],
            'generator_args': {
                'points_per_side': 8,
                'pred_iou_thresh': 0.95,
                'stability_score_thresh': 0.7,
                'crop_n_layers': 0,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 0,
                'point_grids': None,
                'box_nms_thresh': 0.7,
            },
            'gpu_id': args['device_id'],
        }
        print(f"[SAM] Checkpoint : {sam_ckpt}")
        print(f"[SAM] Model type : {args['sam_version']}")
    else:
        sam_args = None

    main(args=args, sam_args=sam_args, saver=saver)

