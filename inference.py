import numpy as np
from tqdm import tqdm
import torch
from utils.utils import norm_batch, get_input_dict
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric, MeanIoU
from monai.transforms import AsDiscrete, Activations
from utils.model_utils import sam_call
import torch.nn.functional as F
from torch.cuda.amp import autocast

def inference_ds_monai(ds, model, sam, transform, epoch, args, saver):
    print('Inference started...')
    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list_true = []
    dice_list_false = []
    loss_list = []
    Idim = int(args['Idim'])
    diceMetric_true = DiceMetric(include_background=True, reduction='mean')
    diceMetric_false = DiceMetric(include_background=False, reduction='mean')
    iouMetric = MeanIoU(include_background=args['include_background'], reduction='mean')
    post_trans = AsDiscrete(threshold=args['theashold_discretize'])
    sigmoid = Activations(sigmoid=True)

    #post-transformation of labels
    one_hot = AsDiscrete(to_onehot=2, dim=1)
    
    apply_sigm = True if (getattr(model, "is_segmentor", False) or args['net']=='Unet') and args['use_standard_net'] else False
    if args['criterion'] == 'dice':
        criterion = DiceLoss(include_background=args['include_background'], sigmoid=apply_sigm)
    elif args['criterion'] == 'dice_ce':
        criterion = DiceCELoss(include_background=args['include_background'], sigmoid=apply_sigm)
    else:
        raise ValueError('Criterion not recognized')
    
    for ix,  sample in enumerate(pbar):
        assert len(sample[args['image_key']]) == 1
        imgs = sample[args['image_key']]
        gts = sample[args['mask_key']]
        original_sz_batch = torch.tensor(np.array([sample[args['image_key']][i].meta['spatial_shape'][:2] for i in range(len(sample[args['image_key']]))]))
        img_sz =torch.tensor(imgs.shape[2:]).repeat(len(sample[args['image_key']]), 1)
        orig_imgs = imgs.to(args['device'])
        gts = gts.to(args['device'])

        with autocast(enabled=args['use_cuda_amp']):
            for idx in range(orig_imgs.shape[2]):
                slice_block = slice = orig_imgs[..., idx, :, :].to(args['device'])
                slice_small = F.interpolate(slice_block, size=(Idim, Idim), mode='bilinear', align_corners=False)
                with torch.no_grad():
                    if sam is not None:
                        dense_embeddings = model(slice_small)
                        batched_input = get_input_dict(slice, original_sz_batch, img_sz)
                        pred = norm_batch(sam_call(batched_input, sam, dense_embeddings, args))
                    else:
                        pred = model(slice_block)
                    
                if idx == 0:
                    depth = orig_imgs.shape[2]
                    masks = torch.zeros((pred.shape[0], pred.shape[1], depth, pred.shape[2], pred.shape[3]), device=args['device'])
                    gts_resized = torch.zeros((masks.shape[0], 1, depth, masks.shape[3], masks.shape[4]), device=args['device'])
                    imgs_resized = torch.zeros((masks.shape[0], 1, depth, masks.shape[3], masks.shape[4]), device=args['device'])

                
                masks[:,:, idx] = pred
                gts_resized[:,:, idx] = F.interpolate(gts[:,:, idx], size=pred.shape[2:], mode='bilinear', align_corners=True)
                imgs_resized[:,:, idx] = F.interpolate(slice, size=pred.shape[2:], mode='bilinear', align_corners=True)

            onehot_gts=one_hot(post_trans(gts_resized))
            if masks.shape[1] == 1:
                onehot_masks = one_hot(post_trans(masks))
            elif masks.shape[1] == 2:
                if apply_sigm:
                    onehot_masks = post_trans(sigmoid(masks))
                else:
                    onehot_masks = post_trans(masks)
            
            if args['save_val_images'] and ix<5 and epoch % 5 == 0:
                saver.log_slices(imgs_resized.movedim(2, -1), onehot_masks[:, 1, ...].unsqueeze(1).detach().movedim(2, -1), gts_resized.detach().movedim(2, -1), epoch, ix, 'val')
                
            diceMetric_true(onehot_masks, onehot_gts)
            dice_true = diceMetric_true.aggregate().item()
            diceMetric_false(onehot_masks, onehot_gts)
            dice_false = diceMetric_false.aggregate().item()
            iouMetric(onehot_masks, onehot_gts)
            iou = iouMetric.aggregate().item()
            iou_list.append(iou)
            dice_list_true.append(dice_true)
            dice_list_false.append(dice_false)
            
            
            if masks.shape[1] == 1:
                loss=criterion(masks, gts_resized)
            elif masks.shape[1] == 2:
                loss=criterion(masks, onehot_gts)
            loss_list.append(loss.item())
            
        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Dice True {dice_true:.4f} :: Dice False {dice_false:.4f} :: IoU {iou:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                dice_true=np.mean(dice_list_true),
                dice_false=np.mean(dice_list_false),
                iou=np.mean(iou_list)))
        
        iouMetric.reset()
        diceMetric_true.reset()
        diceMetric_false.reset()
        
    return np.mean(dice_list_true),np.mean(dice_list_false), np.mean(iou_list), np.mean(loss_list)