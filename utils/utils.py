import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import argparse
from monai.utils import set_determinism
from torch.nn.modules.batchnorm import _BatchNorm
from pathlib import Path
from typing import List, Optional, Tuple, Union

def set_seed(seed, strict=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_determinism(seed=seed)
    if not strict:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    print(
        f"[INFO] Seed set to {seed} | "
        f"Strict reproducibility: {strict} | "
        f"CUDA available: {torch.cuda.is_available()}")   

def norm_batch(x):
    return torch.sigmoid(x)


# ===========================================================================
# Losses
# ===========================================================================

def tversky_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Tversky loss for binary segmentation with automatic dtype conversion.

        Tversky = (TP + smooth) / (TP + alpha*FN + beta*FP + smooth)

    Special cases:
        - alpha = beta = 0.5  ->  Dice loss
        - alpha = beta = 1.0  ->  Jaccard / IoU loss
        - alpha > beta        ->  Penalize FN more (good for small targets)
        - beta > alpha        ->  Penalize FP more

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth mask, shape (B, C, H, W).
    y_pred : torch.Tensor
        Predicted probabilities (after sigmoid), shape (B, C, H, W).
    alpha, beta : float
        Weights for FN and FP.
    smooth : float
        Smoothing constant.

    Returns
    -------
    torch.Tensor
        Scalar loss (mean over the batch).
    """
    # Ensure y_true matches y_pred dtype to prevent AMP (FP16/BF16) mismatches.
    y_true = y_true.to(dtype=y_pred.dtype)

    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1.0 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1.0 - y_true) * y_pred, dim=(1, 2, 3))

    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1.0 - tversky.mean()


# ===========================================================================
# Metrics
# ===========================================================================

def get_dice_ji_torch(
    predict: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Dice coefficient and Jaccard Index PER-SAMPLE with zero-mask handling.

    Handles polyp-free frames correctly: if both prediction and target are
    entirely empty (all zeros), the metric returns 1.0 (perfect prediction
    for a truly negative sample).

    Parameters
    ----------
    predict : torch.Tensor
        Predicted binary mask, shape (B, C, H, W), values in {0, 1} or bool.
    target : torch.Tensor
        Ground-truth binary mask, shape (B, C, H, W), values in {0, 1} or bool.
    eps : float, default 1e-7
        Small epsilon to avoid division by zero.

    Returns
    -------
    (dice, ji) : tuple of torch.Tensor
        dice : Tensor of shape (B,) -- one Dice value per sample.
        ji   : Tensor of shape (B,) -- one Jaccard Index value per sample.
    """
    predict = predict.bool()
    target = target.bool()

    dims = (-3, -2, -1)  # C, H, W
    tp = (predict & target).sum(dim=dims).float()
    fp = (predict & ~target).sum(dim=dims).float()
    fn = (~predict & target).sum(dim=dims).float()

    denom_j = tp + fp + fn
    denom_d = 2.0 * tp + fp + fn

    # If both target and prediction are empty, score = 1.0 (correct negative).
    ji = torch.where(denom_j == 0, torch.ones_like(tp), tp / (denom_j + eps))
    dice = torch.where(
        denom_d == 0, torch.ones_like(tp), (2.0 * tp) / (denom_d + eps)
    )

    return dice, ji


# ===========================================================================
# Experiment folder creation
# ===========================================================================

def create_experiment_folder(base_path: str) -> str:
    """
    Create a new numbered experiment subdirectory inside `base_path`.

    Scans `base_path` for existing folders named 'expN' (N is an integer),
    finds the next available N, and creates `base_path/expN`.

    Parameters
    ----------
    base_path : str
        Parent directory where the numbered folder will be created.

    Returns
    -------
    str
        Path to the newly created folder.
    """
    base = Path(base_path)
    base.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(r"^exp(\d+)$")
    existing_indices = []
    for item in base.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                existing_indices.append(int(match.group(1)))

    next_index = max(existing_indices, default=-1) + 1
    new_folder = base / f"exp{next_index}"
    new_folder.mkdir(exist_ok=False)
    return str(new_folder)


# ===========================================================================
# Training step helper
# ===========================================================================

def gen_step(
    optimizer: torch.optim.Optimizer,
    gts: torch.Tensor,
    masks: torch.Tensor,
    criterion: callable,
    accumulation_steps: int = 4,
    step: int = 0,
    is_last_step: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
) -> float:
    """
    One training step with gradient accumulation and mixed precision.

    Loss = criterion(masks, gts_resized) + tversky_loss(gts_resized, masks)

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    gts : torch.Tensor
        Ground-truth mask, shape (B, H, W) or (B, 1, H, W).
    masks : torch.Tensor
        Predicted probabilities in [0, 1], shape (B, C, H', W').
    criterion : callable
        Loss function applied to (masks, gts_resized).
    accumulation_steps : int, default 4
    step : int, default 0
    is_last_step : bool, default False
        If True, forces an optimizer step and gradient zeroing even if
        the accumulation boundary has not been reached. Use on the last
        batch of each epoch to avoid leaking leftover gradients.
    scaler : torch.cuda.amp.GradScaler or None
    tversky_alpha, tversky_beta : float

    Returns
    -------
    float
        Loss value (as a standard Python float).
    """
    # --- 1) Ensure gts is 4D and contiguous ---
    if gts.ndim == 3:
        gts = gts.unsqueeze(1)
    gts = gts.contiguous()

    # --- 2) Resize GT to match predicted mask resolution ---
    size = masks.shape[2:]
    gts_sized = F.interpolate(gts, size=size, mode="nearest")

    # --- 3) Combined loss calculation ---
    loss = (
        criterion(masks, gts_sized)
        + tversky_loss(
            gts_sized, masks, alpha=tversky_alpha, beta=tversky_beta
        )
    )

    # --- 4) Normalize by accumulation steps ---
    loss_normalized = loss / accumulation_steps

    # --- 5) Backward pass ---
    if scaler is not None:
        scaler.scale(loss_normalized).backward()
    else:
        loss_normalized.backward()

    # --- 6) Optimizer update ---
    should_update = ((step + 1) % accumulation_steps == 0) or is_last_step
    if should_update:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return loss.item()


def get_input_dict(imgs, original_sz, img_sz, point_coords=None, point_labels=None, boxes=None, mask_inputs=None, swap_original_size=True):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size    = tuple(int(x) for x in img_sz[i].tolist())
        original_size = tuple(int(x) for x in original_sz[i].tolist())
        batched_input.append({'image': img,
                              'original_size': original_size,
                              'image_size': input_size})

    return batched_input

# ===========================================================================
# SAM output post-processing
# ===========================================================================

def postprocess_masks(
    masks_dict: List[dict],
    device: Optional[torch.device] = None,
    use_sigmoid: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Consolidate SAM outputs from a list of per-sample dicts into batched tensors.

    Handles multi-mask outputs correctly: if SAM returns K masks per sample
    (multimask_output=True), the mask with the highest IoU prediction is
    selected, and the corresponding IoU is stored alongside it.

    Parameters
    ----------
    masks_dict : list of dict
        Each dict must contain 'low_res_masks' (or 'low_res_logits')
        and optionally 'iou_predictions'.
    device : torch.device, optional
        Target device. If None, uses the device of the first mask.
    use_sigmoid : bool, default True
        If True, applies sigmoid (recommended).
        If False, applies min-max (legacy).

    Returns
    -------
    (masks, ious) : tuple of torch.Tensor
        masks : (N, 1, H, W), values in [0, 1].
        ious  : (N,), IoU of the SELECTED mask per sample.
    """
    if len(masks_dict) == 0:
        raise ValueError("masks_dict is empty.")

    first = masks_dict[0]
    mask_key = "low_res_masks" if "low_res_masks" in first else "low_res_logits"

    if mask_key not in first:
        raise KeyError("Entry must contain 'low_res_masks' or 'low_res_logits'.")

    first_mask = first[mask_key]
    if device is None:
        device = first_mask.device

    if first_mask.dim() < 2:
        raise ValueError(f"Unsupported mask dim: {first_mask.dim()}")
    H, W = first_mask.shape[-2:]

    N = len(masks_dict)
    dtype = first_mask.dtype

    masks = torch.zeros((N, 1, H, W), device=device, dtype=dtype)
    ious = torch.zeros(N, device=device, dtype=torch.float32)

    for i, entry in enumerate(masks_dict):
        cur_mask = entry[mask_key]
        iou_pred = entry.get("iou_predictions", None)

        if cur_mask.device != device:
            cur_mask = cur_mask.to(device)
        if iou_pred is not None and iou_pred.device != device:
            iou_pred = iou_pred.to(device)

        if cur_mask.dim() == 4:
            cur_mask = cur_mask.squeeze(0)

        if cur_mask.dim() == 3:
            K = cur_mask.shape[0]
            if K > 1:
                if iou_pred is None:
                    best_idx = 0
                    iou_value = 1.0
                else:
                    iou_flat = iou_pred.flatten()
                    if iou_flat.numel() != K:
                        raise ValueError(
                            f"IoU predictions ({iou_flat.numel()}) do not "
                            f"match number of masks ({K})."
                        )
                    best_idx = int(iou_flat.argmax().item())
                    iou_value = iou_flat[best_idx].item()
                cur_mask_2d = cur_mask[best_idx]
                ious[i] = iou_value
            else:
                cur_mask_2d = cur_mask[0]
                if iou_pred is not None:
                    ious[i] = iou_pred.flatten()[0].to(ious.dtype)
        elif cur_mask.dim() == 2:
            cur_mask_2d = cur_mask
            if iou_pred is not None:
                ious[i] = iou_pred.flatten()[0].to(ious.dtype)
        else:
            raise ValueError(f"Unsupported mask shape: {tuple(cur_mask.shape)}")

        if use_sigmoid:
            cur_mask_2d = torch.sigmoid(cur_mask_2d)
        else:
            cmin = cur_mask_2d.min()
            cmax = cur_mask_2d.max()
            cur_mask_2d = (cur_mask_2d - cmin) / (cmax - cmin + 1e-6)

        masks[i, 0] = cur_mask_2d

    return masks, ious


# ===========================================================================
# Argument parsing helper
# ===========================================================================

_TRUE_VALUES = frozenset(("yes", "true", "t", "y", "1"))
_FALSE_VALUES = frozenset(("no", "false", "f", "n", "0"))


def str2bool(v: Union[bool, int, str]) -> bool:
    """
    Convert a string/int/bool value to a boolean for argparse.

    Accepts:
        - True/False (bool)                 -> returned as-is
        - 1 / 0 (int)                       -> True / False
        - 'yes' / 'no'                      -> True / False
        - 'true' / 'false'                  -> True / False
        - 't' / 'f'                         -> True / False
        - 'y' / 'n'                         -> True / False
        - '1' / '0'                         -> True / False

    Case-insensitive for strings.

    Parameters
    ----------
    v : bool, int, str
        Value to convert.

    Returns
    -------
    bool

    Raises
    ------
    argparse.ArgumentTypeError
        If the value cannot be converted.
    """
    if isinstance(v, bool):
        return v

    if isinstance(v, int):
        if v == 1:
            return True
        if v == 0:
            return False
        raise argparse.ArgumentTypeError(
            f"Cannot convert integer '{v}' to bool (only 0 and 1 accepted)."
        )

    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE_VALUES:
            return True
        if s in _FALSE_VALUES:
            return False
        raise argparse.ArgumentTypeError(
            f"Cannot convert '{v}' to bool. "
            f"Accepted: {sorted(_TRUE_VALUES | _FALSE_VALUES)}."
        )

    raise argparse.ArgumentTypeError(
        f"Cannot convert '{v}' of type {type(v).__name__} to bool."
    )


def replace_batchnorm_with_instance_norm(model):

    def _replace(m):
        if isinstance(m, _BatchNorm):
            return nn.InstanceNorm2d(
                num_features=m.num_features,
                eps=m.eps,
                momentum=m.momentum if m.momentum is not None else 0.1,
                affine=True,
                track_running_stats=False,
            )
        return m

    def _walk(module, prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, _BatchNorm):
                new_layer = _replace(child)
                setattr(module, name, new_layer)
                print(f"[disable_bn] replaced {full_name}: "
                      f"{type(child).__name__} → {type(new_layer).__name__}")
            elif len(list(child.children())) > 0:
                _walk(child, full_name)

    n_bn_before = sum(1 for m in model.modules() if isinstance(m, _BatchNorm))
    _walk(model)
    n_bn_after = sum(1 for m in model.modules() if isinstance(m, _BatchNorm))
    n_in_after = sum(1 for m in model.modules() if isinstance(m, nn.InstanceNorm2d))

    print(f"[disable_bn] {n_bn_before} BatchNorm layer(s) → "
          f"{n_in_after} InstanceNorm2d layer(s) "
          f"(remaining BatchNorm: {n_bn_after})")
    return model