import argparse
import torch
import numpy as np
import os
import torch.nn.functional as F
import random
from monai.utils import set_determinism
from torch import nn

def set_seed(seed):
    random.seed(seed)  # Per il modulo random di Python
    np.random.seed(seed)  # Per NumPy
    torch.manual_seed(seed)  # Per il generatore di numeri casuali di PyTorch
    torch.cuda.manual_seed(seed)  # Per i generatori CUDA su ogni GPU
    #torch.cuda.manual_seed_all(seed)  # Per tutti i generatori CUDA (se si usa multi-GPU)
    
    # Imposta PyTorch in modalità deterministica (potrebbe rallentare l'esecuzione)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False  # Disabilita l'ottimizzazione basata sul benchmarking

    # MONAI (necessario perché utilizza il proprio generatore di numeri casuali)
    set_determinism(seed=seed)

def norm_batch(x):
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def Dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def gen_step(optimizer, gts, masks, criterion, accumulation_steps, step):
    size = masks.shape[2:]
    gts_sized = F.interpolate(gts.unsqueeze(dim=1), size, mode='nearest')
    loss = criterion(masks, gts_sized) + Dice_loss(masks, gts_sized)
    loss.backward()
    if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def get_input_dict(imgs, original_sz, img_sz, point_coords=None, point_labels=None, boxes=None, mask_inputs=None):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        singel_input = {
            'image': img,
            'original_size': original_size,
            'image_size': input_size
        }
        if point_coords is not None:
            singel_input['point_coords'] = point_coords
        if point_labels is not None:
            singel_input['point_labels'] = point_labels
        if boxes is not None:
            singel_input['boxes'] = boxes
        if mask_inputs is not None:
            singel_input['mask_inputs'] = mask_inputs
        batched_input.append(singel_input)
    return batched_input

def get_input_dict_OLD(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        singel_input = {
            'image': img,
            'original_size': original_size,
            'image_size': input_size, 
            'point_coords': None,
            'point_labels': None,
        }
        batched_input.append(singel_input)
    return batched_input

def postprocess_masks(masks_dict):
    masks = torch.zeros((len(masks_dict), *masks_dict[0]['low_res_logits'].squeeze().shape)).unsqueeze(dim=1).cuda()
    ious = torch.zeros(len(masks_dict)).cuda()
    for i in range(len(masks_dict)):
        cur_mask = masks_dict[i]['low_res_logits'].squeeze()
        cur_mask = (cur_mask - cur_mask.min()) / (cur_mask.max() - cur_mask.min())
        masks[i, 0] = cur_mask.squeeze()
        ious[i] = masks_dict[i]['iou_predictions'].squeeze()
    return masks, ious


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def disable_batchnorm_running_stats(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None


