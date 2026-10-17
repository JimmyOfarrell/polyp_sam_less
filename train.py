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

    # ============================================================
    # Determine if the loss should apply sigmoid internally.
    #   - Phase 1 (ModelEmb + SAM): masks are probabilities → sigmoid=False
    #   - Phase 2 (HardNetSegmentor): masks are raw logits → sigmoid=True
    # ============================================================
    apply_sigm = bool((getattr(model, "is_segmentor", False) or args['net'] == 'Unet') and args['use_standard_net']
)
    if args['criterion'] == 'dice': criterion = DiceLoss(include_background=args['include_background'], sigmoid=apply_sigm)
    elif args['criterion'] == 'dice_ce': criterion = DiceCELoss(include_background=args['include_background'], sigmoid=apply_sigm)
    else:
        raise ValueError(
        f"Criterion '{args['criterion']}' not recognized. "
        f"Choose from: 'dice', 'dice_ce'.")

    # ============================================================
    # Idim: target input size (used by F.interpolate above).
    # optimizer.zero_grad(): clears accumulated gradients at the
    # start of the epoch (safe flush before the training loop).
    # ============================================================
    Idim = int(args['Idim'])
    optimizer.zero_grad()

    # ============================================================
    # Training loop over the DataLoader
    # ============================================================
    for ix, sample in enumerate(pbar):
        # --------------------------------------------------------
        # 1) Handle the MS-style crop-based output.
        #    RandCropByPosNegLabeld returns a list of dicts (one per
        #    crop). We use the first crop. For polyp, `sample` is a
        #    plain dict, so this branch is skipped.
        # --------------------------------------------------------
        if isinstance(sample, list): sample = sample[0]
        # --------------------------------------------------------
        # 2) Extract image and mask from the batch.
        #
        #    Polyp (2D RGB):
        #        imgs : (B, 3, 1024, 1024)
        #        gts  : (B, 1, 1024, 1024)
        #        squeeze(-1) -> no-op (last dim is width, not 1)
        #
        #    MS (3D medical):
        #        imgs : (B, C, H, W, 1)  -> squeeze -> (B, C, H, W)
        #        gts  : (B, 1, H, W, 1)  -> squeeze -> (B, 1, H, W)
        # --------------------------------------------------------
        imgs = sample[args['image_key']].squeeze(-1)
        gts  = sample[args['mask_key']].squeeze(-1)

        # ============================================================
        # Batch size and placeholder size tensors.
        #   - For polyp, all images are resized to (Idim, Idim) by the
        #     transforms, so `original_sz` and `img_sz` are constant.
        #   - `sam_call` in the polyp pipeline does NOT consume these
        #     fields; they are kept for API compatibility with the MS
        #     `get_input_dict`.
        # ============================================================
        B = imgs.shape[0]
        
        original_sz = torch.tensor([[Idim, Idim]] * B, dtype=torch.int64)
        img_sz = torch.tensor(imgs.shape[2:]).repeat(B, 1)

        # ============================================================
        # Move image and GT to the target device (GPU/CPU).
        # ============================================================
        orig_imgs = imgs.to(args['device'])
        gts = gts.to(args['device'])
        
        if args['num_slices'] == 1: orig_imgs_small = F.interpolate(orig_imgs, size=(Idim, Idim), mode='bilinear', align_corners=False)
        elif args['num_slices'] > 1: orig_imgs_small = F.interpolate(orig_imgs, size=(orig_imgs.shape[2], Idim, Idim), mode='trilinear', align_corners=False)
        
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