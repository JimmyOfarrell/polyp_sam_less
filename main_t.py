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
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], help='wandb mode', required=False)
    parser.add_argument('--wandb_project', default='polyp_sam_less', help='wandb project name', required=False)
    parser.add_argument('--wandb_entity', default=None, help='WandB entity (username or team)', required=False)
    args = vars(parser.parse_args())

    set_seed(args['seed'])
    args['experiment_name'] = args['exp_name']
    output_root = 'results'
    args['exp_folder'] = os.path.join(output_root, args['experiment_name'])
    saver = Saver(output_folder=output_root,
            experiment_name=args['experiment_name'],
            wandb_mode=args['wandb_mode'],
            wandb_project=args['wandb_project'],
            wandb_entity=args['wandb_entity'],
            args=args)
    saver.log_hparams(args)
    cmd = "python " + " ".join(sys.argv)
    saver.log_cmd(cmd)
    print(args)