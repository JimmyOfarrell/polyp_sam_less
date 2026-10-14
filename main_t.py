import argparse
import os
import sys
from utils.saver import Saver
from utils.utils import str2bool, set_seed, disable_batchnorm_running_stats
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    ============================================================
                POLYP SEGMENTATION FRAMEWORK
    ============================================================
    Method : SAM-Guided Prompt Learning
    Task   : Automated Polyp Segmentation
    Input  : Colonoscopy RGB Images
    Output : Binary Polyp Segmentation Masks
    ============================================================
 """)
    print(parser.description)
    parser.add_argument('--exp_name', required=True, help='Experiment name (required)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], help='WandB mode', required=False)
    parser.add_argument('--wandb_project', default='polyp_sam_less', help='WandB project name', required=False)
    parser.add_argument('--wandb_entity', default=None, help='WandB entity (username or team)', required=False)
    parser.add_argument('--use_sam', type=str2bool, default=True, help='Use SAM (phase 1) vs aggregator only (phase 2)')
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_b', choices = ['sam_vit_b', 'medsam_vit_b', 'sam_vit_h'], help='SAM checkpoint')
    parser.add_argument('--sam_version', type=str, default='vit_b', choices = ['vit_b', 'vit_h'], help='SAM version')
    parser.add_argument('--device_id', default=0, help='GPU index', required=False)
    args = vars(parser.parse_args())

    set_seed(args['seed'])

    args['experiment_name'] = args['exp_name']
    output_root = 'results'
    args['exp_folder'] = os.path.join(output_root, args['experiment_name'])
    print(f"[INFO] Experiment output directory: {args['exp_folder']}")

    saver = Saver(output_folder=output_root,
                  experiment_name=args['experiment_name'],
                  wandb_mode=args['wandb_mode'],
                  wandb_project=args['wandb_project'],
                  wandb_entity=args['wandb_entity'],
                  args=args)
    saver.log_hparams(args)
    cmd = "python " + " ".join(sys.argv)
    saver.log_cmd(cmd)

    args['path'] = os.path.join(saver.path, 'net_last.pth')
    args['path_best'] = os.path.join(saver.path, 'net_best.pth')
    args['vis_folder'] = os.path.join(saver.path, 'vis')

    if args['use_sam']:
        sam_ckpt = os.path.join('cp', f"{args['sam_ckpt']}.pth")
        print(f"[INFO] SAM checkpoint path: {sam_ckpt}")
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
        print(args)
        print(sam_args)
    else: sam_args = None