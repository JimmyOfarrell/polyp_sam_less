import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast
from utils.model_utils import sam_call
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import AsDiscrete, Activations
from utils.utils import norm_batch, gen_step, get_input_dict

def train_single_epoch(train_data_loader, model, sam, optimizer, epoch, args, saver, scheduler=None, scaler=None):
    one_hot = AsDiscrete(to_onehot=2, dim=1)
    post_trans = AsDiscrete(threshold=args['theashold_discretize'])
    sigmoid = Activations(sigmoid=True)

    diceMetric = DiceMetric(include_background=args['include_background'], reduction='mean')
    iouMetric = MeanIoU(include_background=args['include_background'], reduction='mean')
    
    iou_list = []
    dice_list = []
    loss_list = []
    pbar = tqdm(train_data_loader)
    apply_sigm = bool(getattr(model, "is_segmentor", False) and args['use_standard_net'])
    
    if args['criterion'] == 'dice': criterion = DiceLoss(include_background=args['include_background'], sigmoid=apply_sigm)
    elif args['criterion'] == 'dice_ce': criterion = DiceCELoss(include_background=args['include_background'], sigmoid=apply_sigm)
    else: raise ValueError(f"Criterion '{args['criterion']}' not recognized. "
                           f"Choose from: 'dice', 'dice_ce'.")

    Idim = int(args['Idim'])
    optimizer.zero_grad()

    running_loss = 0.0
    last_logged_lr = None
    for ix, sample in enumerate(pbar):
        imgs = sample[args['image_key']]
        gts  = sample[args['mask_key']]
        B = imgs.shape[0]

        original_sz = torch.tensor(np.array([sample[args['image_key']][i].meta['spatial_shape'][:2] for i in range(len(sample[args['image_key']]))]))
        img_sz =torch.tensor(imgs.shape[2:]).repeat(len(sample[args['image_key']]), 1)

        orig_imgs = imgs.to(args['device'])
        gts = gts.to(args['device'])

        orig_imgs_small = F.interpolate(orig_imgs, size=(Idim, Idim), mode='bilinear', align_corners=False)

        with autocast('cuda', enabled=args['use_cuda_amp'] and torch.cuda.is_available()):
            if sam is not None:
                dense_embeddings = model(orig_imgs_small)
                batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
                masks = norm_batch(sam_call(batched_input, sam, dense_embeddings, args))
            else: masks = model(orig_imgs)

            size = masks.shape[2:]
            gts_resized = F.interpolate(gts, size=size, mode='nearest')

            onehot_gts=one_hot(post_trans(gts_resized))

            if masks.shape[1] == 1:
                pred_polyp = post_trans(masks)
                onehot_masks = one_hot(pred_polyp)
            elif masks.shape[1] == 2:
                probs = sigmoid(masks) if apply_sigm else masks
                pred_polyp = post_trans(probs[:, 1:2])
                onehot_masks = torch.cat([1.0 - pred_polyp, pred_polyp], dim=1)
            
            if args['save_train_images'] and ix < 10 and epoch % 5 == 0:
                imgs_resized = F.interpolate(orig_imgs, size=size, mode='bilinear', align_corners=False)
                pred_2d = onehot_masks[:, 1].detach()
                gt_2d   = gts_resized.squeeze(1).detach()
                saver.log_samples_2d(imgs_resized.detach(), pred_2d, gt_2d, step=epoch, name=f"batch{ix}", phase='train')

            diceMetric(onehot_masks, onehot_gts)
            iouMetric(onehot_masks, onehot_gts)

            dice = diceMetric.aggregate().item()
            iou = iouMetric.aggregate().item()

            if np.isnan(dice): dice = 1.0
            if np.isnan(iou): iou = 1.0
            
            if masks.shape[1] == 1: loss=criterion(masks, gts_resized)
            elif masks.shape[1] == 2: loss=criterion(masks, onehot_gts)
            else: raise ValueError(f"Unexpected mask channels: {masks.shape[1]}")

            if not torch.isfinite(loss):
                print(f"[WARNING] Non-finite loss at epoch {epoch}, iter {ix}: "
                      f"{loss.item():.4f}. Skipping batch.")
                optimizer.zero_grad()
                diceMetric.reset()
                iouMetric.reset()
                continue

            loss_list.append(loss.item())
            iou_list.append(iou)
            dice_list.append(dice)
            running_loss += loss.item()
        
        scaler.scale(loss).backward()

        is_last = (ix + 1) == len(train_data_loader)
        if ((ix + 1) % args['acc_steps'] == 0) or is_last:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if args['use_scheduler']:
            if args['scheduler'] == 'ReduceLROnPlateau': scheduler.step(loss.item())
            else: scheduler.step()

            if ix == 0:
                current_lr = optimizer.param_groups[0]['lr']
                saver.log_loss('lr', current_lr, epoch)
                last_logged_lr = current_lr

        pbar.set_description(
        f'(train | Medical) epoch {epoch} :: '
        f'loss {running_loss / (ix + 1):.4f}')

        diceMetric.reset()
        iouMetric.reset()

    if args['use_scheduler']: saver.log_loss('lr_end', optimizer.param_groups[0]['lr'], epoch)

    return np.mean(loss_list), np.mean(dice_list), np.mean(iou_list)