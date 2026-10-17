import argparse
import torch
import os
import sys
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.dataset_utils import get_train_val_dataloaders
from utils.model_utils import get_model, get_standard_model
from utils.saver import Saver
from utils.utils import str2bool, set_seed, disable_batchnorm_running_stats

def main(args: dict | None = None, sam_args: dict | None = None, saver: Saver | None = None) -> None:
    requested = str(args['device']).strip().lower()
    if requested.startswith('cuda'):
        if torch.cuda.is_available():
            # Use explicit GPU index if given (e.g., 'cuda:1')
            if ':' in requested:
                args['device'] = torch.device(requested)
            else:
                device_id = int(args.get('device_id', 0))
                n_gpus = torch.cuda.device_count()
                if device_id >= n_gpus:
                    print(f"[WARNING] device_id={device_id} out of range "
                          f"(only {n_gpus} GPU(s) available). Using cuda:0.")
                    device_id = 0
                args['device'] = torch.device(f"cuda:{device_id}")
            # Set default CUDA device (useful in multi-GPU environments)
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
        try:
            sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])

        except Exception as e:
            raise RuntimeError(
                f"Failed to load SAM '{sam_args['model_type']}' from "
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
    print(f"[INFO] Data loaders initialized | Train batches: {len(train_dataloader)} | Validation batches: {len(val_dataloader)}")
    img_ch = train_dataloader.dataset[0][args['image_key']].shape[0]
    print(f"[INFO] Image channels: {img_ch}")

    if not args['use_standard_net']:
        model = get_model(args, sam, img_ch)
        print(f"[INFO] Phase 1 model: {type(model).__name__} (prompt learner)")
    else:
        model = get_standard_model(args, img_ch)
        print(f"[INFO] Phase 2 model: {type(model).__name__} (aggregator)")

    model = model.to(args['device'])
    print(f"[INFO] Model moved to {args['device']}")       

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
    parser.add_argument('--use_sam', type=str2bool, default=True, help='Use SAM (phase 1) vs aggregator only (phase 2)')
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_b', choices = ['sam_vit_b', 'medsam_vit_b', 'sam_vit_h'], help='SAM checkpoint')
    parser.add_argument('--sam_version', type=str, default='vit_b', choices = ['vit_b', 'vit_h'], help='SAM version')
    parser.add_argument('--device_id', type=int, default=0, help='GPU index')
    parser.add_argument('--device', default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('-Idim', '--Idim', type=int, default=512, help='Input image size')
    parser.add_argument('-task', '--task', default='polyp', help='Task type: polyp | mslesseg | pancreas | ...')
    parser.add_argument('-nW', '--nW', default=0, help='Number of workers for train loader')
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='Number of workers for val loader')  
    parser.add_argument('-bs', '--Batch_size', default=2, help='Batch size')
    parser.add_argument('--split_path', type=str, default=os.path.join(PROJECT_DIR, 'data', 'split.json'), help='Path to the JSON split file')
    parser.add_argument('--n_fold', type=int, default=0, help='Fold index')
    parser.add_argument('--cache_rate', type=float, default=0.0, help='Dataset cache rate (0.0 to 1.0)')
    parser.add_argument('--image_key', type=str, default='image',help='Key for image in the data dict')
    parser.add_argument('--mask_key', type=str, default='mask', help='Key for mask in the data dict')
    parser.add_argument('--use_standard_net', type=str2bool, default=False, help='Use standard segmentor (phase 2) vs prompt learner (phase 1)')
    parser.add_argument('--model_emb', type=str, default='HardNet', help='Prompt learner backbone')
    parser.add_argument('-depth_wise', '--depth_wise', type = str2bool, default=False, help='Use depthwise HarDNet')
    parser.add_argument('-order', '--order', default=68, type=int, help='HarDNet variant: 39 | 68 | 85')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='Use ImageNet pretrained weights')
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
    saver.log_hparams(args)
    cmd = "python " + " ".join(sys.argv)
    saver.log_cmd(cmd)

    args['exp_folder'] = saver.path
    args['path'] = os.path.join(saver.ckpt_path, 'net_last.pth')
    args['path_best'] = os.path.join(saver.ckpt_path, 'net_best.pth')
    args['vis_folder'] = saver.vis_path

    if args['use_sam']:
        project_dir = os.path.dirname(os.path.abspath(__file__))
        sam_ckpt = os.path.join(project_dir, 'cp', f"{args['sam_ckpt']}.pth")
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
        print(f"[INFO] SAM checkpoint path: {sam_ckpt}")
        print(f"[INFO] SAM Model type : {args['sam_version']}")

    else:
         sam_args = None
         print("[INFO] SAM is disabled.")

    main(args=args, sam_args=sam_args, saver=saver)     