import torch.optim as optim
import torch
import os
from utils.saver import Saver
import sys
from inference import inference_ds_monai
from train import train_single_epoch, train_single_epoch_monai
from utils.model_utils import get_model, get_standard_model
from utils.dataset_utils import get_dataset
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry
from utils.utils import str2bool, set_seed, disable_batchnorm_running_stats
import wandb
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule

def main(args=None, sam_args=None, saver=None):
    if args['device']=='cuda':
        args['device'] = torch.device("cuda:"+str(args['device_id']))
    else:
        args['device'] = torch.device("cpu")
    
    if args['use_sam']:
        sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        sam.to(device=args['device'])
        img_dim = sam.image_encoder.img_size
        transform = ResizeLongestSide(img_dim)
        if args['disable_batchnorm_running_stats']:
            disable_batchnorm_running_stats(sam)
    else:
        sam = None
        img_dim = int(args['Idim'])
        transform = None
    
    ds, ds_val = get_dataset(args, transform, img_dim)
    img_ch = ds.dataset[0][0][args['image_key']].shape[0]
    
    if not args['use_standard_net']:
        model = get_model(args, sam, img_ch)
    else:
        model = get_standard_model(args, img_ch)
        
    if args['disable_batchnorm_running_stats']:
            disable_batchnorm_running_stats(model)
    
    if not args['use_sam']:
        if not args['segmentor_finetune_backbone']:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.segmentor.parameters():
                param.requires_grad = True
            params_to_optimize = model.segmentor.parameters()
        else:
            for param in model.parameters():
                param.requires_grad = True
            params_to_optimize = model.parameters()
    else:
        # If using SAM, optimize all model parameters
        params_to_optimize = model.parameters()

    if args['optim']=='Adam':
        optimizer = optim.Adam(params_to_optimize,
                            lr=float(args['learning_rate']),
                            weight_decay=float(args['WD']))
    elif args['optim']=='SGD':
        optimizer = optim.SGD(params_to_optimize,
                            lr=float(args['learning_rate']),
                            weight_decay=float(args['WD']),
                            momentum=float(args['momentum']))
    elif args['optim']=='AdamW':
        optimizer = optim.AdamW(params_to_optimize,
                            lr=float(args['learning_rate']),
                            weight_decay=float(args['WD']))
    if args['use_scheduler']:
        steps_per_epoch = len(ds)
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

    
    if args['wandb_watch']:
        wandb.watch(model, log='all', log_freq=args['wandb_watch_freq'], log_graph=True)
    
    best = 0
    path_best = os.path.join(saver.path,'best.csv')
    f_best = open(path_best, 'w')

    scaler = torch.cuda.amp.GradScaler(enabled=args['use_cuda_amp'])
    
    for epoch in range(int(args['epoches'])):
        if args['task'] in ['pancreas', 'spleen', 'prostate', 'brats', 'mslesseg']:
            avg_loss, avg_dice, avg_iou= train_single_epoch_monai(ds, model.train(), sam.eval() if args['use_sam'] else None, optimizer, transform, epoch, args, saver, scheduler, scaler)
        else:
            avg_loss= train_single_epoch(ds, model.train(), sam.eval(), optimizer, transform, epoch, args)
        saver.log_loss('train_loss', avg_loss, epoch)
        saver.log_loss('train_dice', avg_dice, epoch)
        saver.log_loss('train_iou', avg_iou, epoch)
        with torch.no_grad():
            if args['task'] in ['pancreas', 'spleen', 'prostate', 'brats', 'mslesseg']:
                dice_true, dice_false, IoU_val, val_loss = inference_ds_monai(ds_val, model.eval(), sam, transform, epoch, args, saver)
                saver.log_loss('val_loss', val_loss, epoch)
                saver.log_loss('IoU', IoU_val, epoch)
                saver.log_loss('Dice/include_true', dice_true, epoch)
                saver.log_loss('Dice/include_false', dice_false, epoch)
                
            else:
                dice, IoU_val = inference_ds(ds_val, model.eval(), sam, transform, epoch, args)
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
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=0.01, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=2, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', type=int, default=5000, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='mslesseg', help='Task type (e.g., pancreas, spleen, prostate, brats, mslesseg, tooth)', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', type = str2bool, default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, type=int, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=512, help='image size', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    parser.add_argument('--wandb_mode', default='online', help='wandb mode', required=False)
    parser.add_argument('--device', default='cuda', help='device', required=False)
    parser.add_argument('--device_id', default=0, help='device id', required=False)
    parser.add_argument('--wandb_project', default='AutoSam_Pancreas', help='wandb project', required=False)
    parser.add_argument('--wandb_entity', default='fproietto', help='wandb entity', required=False)
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--data_dir', type=str, default='', help='Path to the dataset directory', required=False)
    parser.add_argument('--split_path', type=str, default='3DAutoSam_Pancreas/Dataset.json', help='Path to the split file')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=10, help='Save model every n epochs')
    parser.add_argument('--image_key', type=str, default='image',help='Modality to use for training')
    parser.add_argument('--mask_key', type=str, default='mask', help='Mask modality')
    parser.add_argument('--spatial_crop_size', type=int, default=224, help='Spatial crop size')
    parser.add_argument('--pos_rand_crop', type=float, default=0.75, help='Positive ratio for random crop')
    parser.add_argument('--neg_rand_crop', type=float, default=0.25, help='Negative ratio for random crop')
    parser.add_argument('--axcodes', type=str, default='RAS', help='Axial codes')
    parser.add_argument('--num_slices', type=int, default=1, help='Number of slices')
    parser.add_argument('--n_fold', type=int, default=0, help='Specify fold')
    parser.add_argument('--criterion', type=str, default='dice', choices=['dice', 'dice_ce'], help='Criterion')
    parser.add_argument('--include_background', type=str2bool, default=False, help='Include background in loss')
    parser.add_argument('--acc_steps', type=int, default=2, help='Number of accumulation steps')
    parser.add_argument('--cache_rate', type=float, default=.7, help='Cache rate')
    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum (if SGD is used)')
    parser.add_argument('--use_scheduler', type=str2bool, default=False, help='If use scheduler or not')
    parser.add_argument('--scheduler', type=str, default='StepLR', choices=['StepLR', 'MultiStepLR', 'ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'WarmupCosineSchedule', 'WarmupLinearSchedule'], help='Scheduler')
    parser.add_argument('--scheduler_step', type=int, default=100, help='Scheduler step for StepLR, and first step of MultiStepLR')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1, help='Scheduler gamma')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='ReduceLROnPlateau Scheduler patience')
    parser.add_argument('--schedCos_Tmax', type=int, default=10, help='CosineAnnealingLR Tmax')
    parser.add_argument('--schedCos_eta_min', type=float, default=0.0001, help='CosineAnnealingLR eta_min')
    parser.add_argument('--schedCos_last_epoch', type=int, default=-1, help='CosineAnnealingLR last_epoch')
    parser.add_argument('--best_metric', type=str, default='IoU', help='Best metric') #until now, only IoU is implemented as best metric to save the best model
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--warmup_cycles', type=float, default=0.5, help='Warmup cycles')
    parser.add_argument('--schedWarmupCos_last_epoch', type=int, default=-1, help='WarmupCosineSchedule last_epoch (use only if you are resuming a training)')
    parser.add_argument('--use_cuda_amp', type=str2bool, default=True, help='Use mixed precision training')

    parser.add_argument('--save_train_images', type=str2bool, default=True, help='Save images')
    parser.add_argument('--save_val_images', type=str2bool, default=True, help='Save images')
    parser.add_argument('--theashold_discretize', type=float, default=0.01, help='Threshold for discretization')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Checkpoint path to continue training')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='Pretrained model')
    parser.add_argument('--model_emb', type=str, default='HardNet', help='Model embedding')

    parser.add_argument('--remove_maxpool', type=str2bool, default=False, help='Remove first maxpool when ResNet is used as embedding model')
    
    parser.add_argument('--use_sam', type=str2bool, default=True, help='Use SAM')
    parser.add_argument('--use_standard_net', type=str2bool, default=False, help='Use standard net')
    parser.add_argument('--net', type=str, default='HardNetSegmentor', help='Net')
    parser.add_argument('--net_segmentor', type=str, default='PointwiseConv', help='Net segmentor')
    parser.add_argument('--sam_zeroshot', type=str2bool, default=False, help='SAM zero shot')
    parser.add_argument('--upsample_factor_list', type=int, nargs='+', default=[2, 2], help='Upsample factor list for ConvUpsample segmentor')
    
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_b', choices = ['sam_vit_b', 'medsam_vit_b', 'sam_vit_h'], help='SAM checkpoint')
    parser.add_argument('--sam_version', type=str, default='vit_b', choices = ['vit_b', 'vit_h'], help='SAM version')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=[], help='Wandb tags')
    parser.add_argument('--wandb_watch', type=str2bool, default=False, help='Wandb watch')
    parser.add_argument('--wandb_watch_freq', type=int, default=10, help='Wandb watch frequency')

    parser.add_argument('--out_channels', type=int, default=2, help='Output channels')
    parser.add_argument('--label_values', type=int, nargs='+', default=[1, 2, 4], help='Label values to discretize mask (primarily for the BRATS dataset; may not apply to other datasets)')
    parser.add_argument('--center_crop', type=str2bool, default=False, help='Center crop the input images')
    parser.add_argument('--train_with_crop', type=str2bool, default=True, help='Train with crop')

    parser.add_argument('--segmentor_finetune_backbone', type=str2bool, default=False, help='If we want to train the miniSegmentor finetuning also the backbone')
    parser.add_argument('--decoder_channels', type=int, nargs='+', default=[], help='Decoder channels for UnetLike segmentor/aggregator')
    
    
    parser.add_argument('--disable_batchnorm_running_stats', type=str2bool, default=False, help='Disable batchnorm running stats')
    
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    
    args = vars(parser.parse_args())
    
    set_seed(args['seed'])
    
    args['experiment_name']= args['exp_name']
    args['exp_folder'] = os.path.join('results', args['experiment_name'])
    saver = Saver(output_folder='results', experiment_name=args['experiment_name'], wandb_mode=args['wandb_mode'], wandb_project=args['wandb_project'], wandb_entity=args['wandb_entity'], args=args)
    args['path']= saver.path
    saver.log_hparams(args)
    cmd = str(sys.argv)
    saver.log_cmd(cmd)
    args['path'] = os.path.join(saver.path,
                                'net_last.pth')
    args['path_best'] = os.path.join(saver.path,
                                     'net_best.pth')
    args['vis_folder'] = os.path.join(saver.path, 'vis')
    
    if args['use_sam']:
        sam_ckpt = "cp/" + args['sam_ckpt'] + ".pth"
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
    else:
        sam_args = None
    main(args=args, sam_args=sam_args, saver=saver)

