import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import norm_batch, gen_step, get_input_dict
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import AsDiscrete, Activations
from utils.model_utils import sam_call
from torch.cuda.amp import autocast


def train_single_epoch(ds, model, sam, optimizer, epoch, args):
    loss_list = []
    pbar = tqdm(ds)
    criterion = nn.BCELoss()
    Idim = int(args['Idim'])
    optimizer.zero_grad()
    for ix, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)
        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks = norm_batch(sam_call(batched_input, sam, dense_embeddings))
        loss = gen_step(optimizer, gts, masks, criterion, accumulation_steps=4, step=ix)
        loss_list.append(loss)
        pbar.set_description(
            '(train | {}) epoch {epoch} ::'
            ' loss {loss:.4f}'.format(
                'Medical',
                epoch=epoch,
                loss=np.mean(loss_list)
            ))
    return np.mean(loss_list)

def train_single_epoch_monai(ds, model, sam, optimizer, transform, epoch, args, saver, scheduler=None, scaler=None):
    #post-transformation of labels
    #discretize_labels = AsDiscrete(threshold = args['theashold_discretize'])
    one_hot = AsDiscrete(to_onehot=2, dim=1)
    post_trans = AsDiscrete(threshold=args['theashold_discretize'])
    sigmoid = Activations(sigmoid=True)
    
    diceMetric = DiceMetric(include_background=args['include_background'], reduction='mean')
    iouMetric = MeanIoU(include_background=args['include_background'], reduction='mean')
    
    iou_list = []
    dice_list = []
    loss_list = []
    pbar = tqdm(ds)
    apply_sigm = True if (getattr(model, "is_segmentor", False) or args['net']=='Unet') and args['use_standard_net'] else False
    if args['criterion'] == 'dice':
        criterion = DiceLoss(include_background=args['include_background'], sigmoid=apply_sigm)
    elif args['criterion'] == 'dice_ce':
        criterion = DiceCELoss(include_background=args['include_background'], sigmoid=apply_sigm)
    else:
        raise ValueError('Criterion not recognized')
    Idim = int(args['Idim'])
    optimizer.zero_grad()
    for ix,  sample in enumerate(pbar):      
        
        if isinstance(sample, list): sample = sample[0]
        imgs = sample[args['image_key']].squeeze(-1)
        gts = sample[args['mask_key']].squeeze(-1)#.squeeze(1)
        original_sz = torch.tensor(np.array([sample[args['image_key']][i].meta['spatial_shape'][:2] for i in range(len(sample[args['image_key']]))]))
        img_sz =torch.tensor(imgs.shape[2:]).repeat(len(sample[args['image_key']]), 1)
        orig_imgs = imgs.to(args['device'])
        gts = gts.to(args['device'])
        
        #orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        if args['num_slices'] == 1:
            orig_imgs_small = F.interpolate(orig_imgs, size=(Idim, Idim), mode='bilinear', align_corners=False)
        elif args['num_slices'] > 1:
            orig_imgs_small = F.interpolate(orig_imgs, size=(orig_imgs.shape[2], Idim, Idim), mode='trilinear', align_corners=False)
        
        with autocast(enabled=args['use_cuda_amp']):
            if sam is not None:
                dense_embeddings = model(orig_imgs_small)
                batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
                masks = norm_batch(sam_call(batched_input, sam, dense_embeddings, args))
            else:
                masks = model(orig_imgs)
            size = masks.shape[2:]
            gts_resized = F.interpolate(gts, size=size, mode='bilinear', align_corners=False)
            imgs_resized = F.interpolate(orig_imgs, size=size, mode='bilinear', align_corners=False)

            #one_hot_masks = one_hot(masks)
            onehot_gts=one_hot(post_trans(gts_resized))
            if masks.shape[1] == 1:
                onehot_masks = one_hot(post_trans(masks))
            elif masks.shape[1] == 2:
                if apply_sigm:
                    onehot_masks = post_trans(sigmoid(masks))
                else:
                    onehot_masks = post_trans(masks)
            
            if args['save_train_images'] and ix<10 and epoch % 5 == 0:
                if masks.shape[1]==2:
                    saver.log_slices(imgs_resized.unsqueeze(-1).detach(), onehot_masks[:, 1, ...].unsqueeze(1).unsqueeze(-1).detach(), gts_resized.unsqueeze(-1).detach(), epoch, ix, 'train')
                else:
                    saver.log_slices(imgs_resized.unsqueeze(-1).detach(), onehot_masks[:, 1, ...].unsqueeze(1).unsqueeze(-1).detach(), gts_resized.unsqueeze(-1).detach(), epoch, ix, 'train')
            
            diceMetric(onehot_masks, onehot_gts)
            iouMetric(onehot_masks, onehot_gts)
            dice = diceMetric.aggregate().item()
            iou = iouMetric.aggregate().item()
            
            if masks.shape[1] == 1:
                loss=criterion(masks, gts_resized)
            elif masks.shape[1] == 2:
                loss=criterion(masks, onehot_gts)
            loss_list.append(loss.item())
            iou_list.append(iou)
            dice_list.append(dice)
        
        scaler.scale(loss).backward()

        if (ix + 1) % args['acc_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if args['use_scheduler']:
            if args['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(loss)
            else:
                scheduler.step()
            
            saver.log_loss('lr', optimizer.param_groups[0]['lr'], epoch)
        pbar.set_description(
            '(train | {}) epoch {epoch} ::'
            ' loss {loss:.4f}'.format(
                'Medical',
                epoch=epoch,
                loss=np.mean(loss_list)
            ))
        diceMetric.reset()
        iouMetric.reset()
    return np.mean(loss_list), np.mean(dice_list), np.mean(iou_list)