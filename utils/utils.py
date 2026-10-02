"""
Utility functions for the polyp segmentation project.

This module provides:
    - Reproducibility helpers (set_seed)
    - Logit-to-probability conversion (logits_to_probs, alias norm_batch)
    - Loss functions (tversky_loss)
    - Metric helpers (get_dice_ji_torch)
    - Experiment folder creation (create_experiment_folder)
    - Training step helper (gen_step)
    - SAM input construction (get_input_dict, postprocess_masks)
    - Argument parsing (str2bool)
    - Model preparation (disable_batchnorm_running_stats)
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.utils import set_determinism
from torch.nn.modules.batchnorm import _BatchNorm


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(seed: int, strict: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    MONAI's set_determinism handles Python, NumPy, PyTorch seeds and sets
    cudnn.deterministic=True and cudnn.benchmark=False by default.

    Parameters
    ----------
    seed : int
        Random seed to use.
    strict : bool, default True
        If True, keeps cudnn.deterministic=True and cudnn.benchmark=False
        (slower but reproducible).
        If False, sets cudnn.deterministic=False and cudnn.benchmark=True
        (faster but not fully reproducible across runs on GPU).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_determinism(seed=seed)

    if not strict:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ===========================================================================
# Logit -> Probability conversion
# ===========================================================================

def logits_to_probs(x: torch.Tensor) -> torch.Tensor:
    """
    Convert SAM's raw logits to probabilities in [0, 1] via sigmoid.

    This function replaces the legacy min-max normalization (used in the
    original MS-SAM-LESS framework) with `torch.sigmoid`, which is the
    mathematically correct transformation from logits to probabilities.

    Why sigmoid instead of min-max?
    --------------------------------
    SAM outputs logits where:
        - logit > 0  ->  foreground (target object)
        - logit < 0  ->  background
        - logit = 0  ->  decision boundary

    sigmoid preserves this boundary at probability 0.5:
        sigmoid(0) = 0.5

    In contrast, min-max normalization shifts the boundary on a per-image
    basis:
        boundary = (0 - x_min) / (x_max - x_min)

    This means the boundary depends on each image's min/max, which can
    produce false positives on images WITHOUT a target object. For
    example, in a polyp-free frame where all logits are negative
    (e.g., [-15, -2]), min-max maps the highest logit (-2) to 1.0,
    creating a fake polyp.

    For polyp segmentation, where many frames may not contain a polyp,
    this is critical. sigmoid is the correct choice.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of raw logits from SAM's mask decoder.
        Typically of shape (B, 1, 256, 256).

    Returns
    -------
    torch.Tensor
        Probabilities in [0, 1], same shape and dtype as input.
    """
    return torch.sigmoid(x)


# Backward-compatible alias (used by existing code).
norm_batch = logits_to_probs


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


# ===========================================================================
# SAM input construction
# ===========================================================================

def get_input_dict(
    imgs,
    original_sz,
    img_sz,
    point_coords=None,
    point_labels=None,
    boxes=None,
    mask_inputs=None,
    swap_original_size=True,
):
    """
    Build per-sample input dicts for SAM (polyp 2D pipeline).

    This is a polyp-focused variant of the original MS function. It keeps
    the same signature (plus one optional keyword) so it remains a
    drop-in replacement, while fixing several issues for 2D RGB data.

    Parameters
    ----------
    imgs : torch.Tensor
        Batch of images, shape (B, C, H, W).
        For polyp: (B, 3, 1024, 1024), already ImageNet-normalized.
    original_sz : torch.Tensor, numpy.ndarray, or list/tuple of pairs
        Per-sample ORIGINAL sizes, shape (B, 2).
        For polyp, PIL's `meta['spatial_shape']` provides (W, H).
        This function will swap it to (H, W) by default
        (see `swap_original_size`).
    img_sz : torch.Tensor, numpy.ndarray, or list/tuple of pairs
        Per-sample CURRENT sizes, shape (B, 2). Usually (1024, 1024).
    point_coords, point_labels, boxes, mask_inputs : optional, batched
        Optional prompts. When provided, the first dim MUST equal B.
        Each one is passed PER-SAMPLE (indexed by i), matching SAM's
        native API. NOT used by the polyp pipeline (polyp uses dense
        embeddings only), but kept for API completeness.
    swap_original_size : bool, default True
        If True, `original_size` is swapped from (W, H) to (H, W).
        Default True is correct for polyp because PIL returns (W, H).
        Set to False if your input already follows (H, W) order.

    Returns
    -------
    list of dict
        One dict per sample, with keys:
            'image'         : (C, H, W) tensor
            'original_size' : (h, w) tuple of ints
            'image_size'    : (h, w) tuple of ints
            (+ any provided prompt, per-sample)
    """
    B = imgs.shape[0]
    batched_input = []

    for i, img in enumerate(imgs):
        # --- Sizes: robustly handle tensor / numpy / list / tuple ---
        original_size = _size_to_tuple(original_sz[i])
        input_size    = _size_to_tuple(img_sz[i])

        # --- Polyp fix: PIL meta gives (W, H); swap to (H, W) ---
        if swap_original_size:
            original_size = (original_size[1], original_size[0])

        single_input = {
            "image": img,
            "original_size": original_size,
            "image_size": input_size,
        }

        # --- Optional prompts: per-sample, with batch-size validation ---
        if point_coords is not None:
            _check_batch_dim(point_coords, B, "point_coords")
            single_input["point_coords"] = _slice_prompt(point_coords, i, ndim=2)

        if point_labels is not None:
            _check_batch_dim(point_labels, B, "point_labels")
            single_input["point_labels"] = _slice_prompt(point_labels, i, ndim=1)

        if boxes is not None:
            _check_batch_dim(boxes, B, "boxes")
            single_input["boxes"] = _slice_prompt(boxes, i, ndim=2)

        if mask_inputs is not None:
            _check_batch_dim(mask_inputs, B, "mask_inputs")
            single_input["mask_inputs"] = mask_inputs[i]

        batched_input.append(single_input)

    return batched_input


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _size_to_tuple(x):
    """
    Convert a size (tensor / numpy array / list / tuple) into a plain
    tuple of ints, regardless of its original shape.

    Handles shapes: (2,), (1, 2), (2, 1), [h, w], (h, w).
    """
    if hasattr(x, "flatten"):                 # torch.Tensor
        return tuple(int(v) for v in x.flatten().tolist())
    if hasattr(x, "tolist"):                  # numpy array
        return tuple(int(v) for v in x.tolist())
    return tuple(int(v) for v in x)           # python list / tuple


def _check_batch_dim(x, B, name):
    """
    Assert that x has a first dimension equal to B.
    Used only for optional prompts, to catch batch mismatches early.
    """
    if not hasattr(x, "shape"):
        raise TypeError(
            f"Prompt '{name}' must have a `.shape` attribute, "
            f"got {type(x).__name__}"
        )
    if x.shape[0] != B:
        raise ValueError(
            f"Prompt '{name}' batch size mismatch: "
            f"expected {B}, got {x.shape[0]}"
        )


def _slice_prompt(x, i, ndim):
    """
    Return the i-th sample of a batched prompt. If the sample already has
    the right number of dims, prepend a singleton so SAM sees shape (1, N, ...).
    """
    s = x[i]
    if s.dim() == ndim:
        return s.unsqueeze(0)
    return s


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


# ===========================================================================
# Model preparation
# ===========================================================================

def disable_batchnorm_running_stats(model: nn.Module) -> nn.Module:
    """
    Disable running statistics for all BatchNorm layers in the model.

    Uses `torch.func.replace_all_batch_norm_modules_` (PyTorch >= 2.0)
    to update every BatchNorm layer in-place consistently:
        - track_running_stats is set to False
        - running_mean and running_var are set to None
        - the internal state remains consistent (no RuntimeError in eval).

    The counting uses `_BatchNorm` to cover all derived classes
    (BatchNorm1d/2d/3d, SyncBatchNorm, and any future subclasses).

    WARNING
    -------
    After calling this function, BatchNorm layers in eval mode will use
    the statistics of the current input batch (per-forward-pass) instead
    of running averages. This means:
        - In eval mode with batch_size=1, the variance is zero, which
          can produce NaN outputs. Use batch_size >= 2 for evaluation.
        - For batch_size=1 inference, consider replacing BatchNorm with
          GroupNorm or LayerNorm instead.

    For polyp segmentation, where validation and testing use batch_size=1,
    do NOT enable this feature (keep
    `args['disable_batchnorm_running_stats'] = False`).

    Parameters
    ----------
    model : nn.Module
        The model whose BatchNorm layers will be modified in-place.

    Returns
    -------
    nn.Module
        The same model, modified in-place.
    """
    from torch.func import replace_all_batch_norm_modules_

    replace_all_batch_norm_modules_(model)

    count = sum(1 for m in model.modules() if isinstance(m, _BatchNorm))
    print(
        f"[INFO] disable_batchnorm_running_stats: "
        f"updated {count} BatchNorm layer(s) to track_running_stats=False."
    )

    return model