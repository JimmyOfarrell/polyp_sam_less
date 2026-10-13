"""
Polyp segmentation transforms.

This module contains transforms for polyp segmentation:
- ResizeWithRatio: Resize image and mask preserving aspect ratio.
- get_polyp_transform: Build training and validation/test transforms.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Any, Dict, Mapping, Optional, Tuple
from monai.transforms import (
    MapTransform,
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    ScaleIntensityRangeD,
    NormalizeIntensityD,
    ResizeWithPadOrCropd,
    RandFlipD,
    RandRotate90D,
    EnsureTyped)

class ResizeWithRatio(MapTransform):
    """
    Aspect-ratio-preserving resizer for 2D polyp segmentation samples.

    Resizes the longest side of the input image (and its corresponding
    mask) to ``image_size`` while preserving the original aspect ratio.
    The shorter side is scaled proportionally. Any further padding or
    cropping to a fixed square resolution is expected to be handled
    downstream (e.g. by ``ResizeWithPadOrCropd``).

    Image and mask use independent interpolation modes so that binary
    masks are never interpolated with a smooth kernel:

    * image -> bilinear / bicubic / area / nearest (optional antialiasing)
    * mask  -> nearest (default) or bilinear

    Expected tensor layout for both image and mask is ``(C, H, W)``,
    matching the output of MONAI's ``EnsureChannelFirstD``.

    Parameters
    ----------
    image_key : str
        Dictionary key of the image tensor.
    mask_key : str
        Dictionary key of the mask tensor.
    image_size : int
        Target size for the *longest* side of the resized tensors.
    image_mode : str, optional
        Interpolation mode for the image. One of
        ``"bilinear"``, ``"bicubic"``, ``"area"`` or ``"nearest"``.
        Defaults to ``"bilinear"``. Recommended: ``"bicubic"`` for
        polyp segmentation (better edge preservation).
    mask_mode : str, optional
        Interpolation mode for the mask. Should remain ``"nearest"``
        to preserve binary values. Defaults to ``"nearest"``.
    antialias : bool, optional
        If ``True``, applies antialiasing when downsampling the image.
        Only supported by ``"bilinear"`` and ``"bicubic"`` modes and
        ignored for masks. Defaults to ``False``. Recommended: ``True``
        for high-resolution polyp images.

    Notes
    -----
    * ``align_corners`` is set to ``False`` for bilinear/bicubic
      (modern PyTorch convention) and omitted for nearest/area.
    * Target side lengths are rounded with :func:`round` to minimise
      aspect-ratio drift while staying integral.
    * A lower bound of 1 is enforced on each target side to avoid
      degenerate zero-size tensors on extreme aspect ratios.
    """
    
    def __init__(self, image_key: str, mask_key: str, image_size: int, image_mode: str = "bilinear", mask_mode: str = "nearest", antialias: bool = False) -> None:
        super().__init__(keys=[image_key, mask_key])

        if image_mode not in ("bilinear", "bicubic", "area", "nearest"): raise ValueError(f"Unsupported image_mode '{image_mode}'. ""Choose from 'bilinear', 'bicubic', 'area', 'nearest'.")
        if mask_mode not in ("nearest", "bilinear"): raise ValueError(f"Unsupported mask_mode '{mask_mode}'. ""Use 'nearest' to preserve binary values.")

        self.image_key: str = image_key
        self.mask_key: str = mask_key
        self.image_size: int = int(image_size)
        self.image_mode: str = image_mode
        self.mask_mode: str = mask_mode
        self.antialias: bool = bool(antialias)

        # align_corners is only valid for bilinear / bicubic
        self._image_align_corners: Optional[bool] = (None if image_mode in ("nearest", "area") else False)

    def __call__(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Apply the resize transform to a single sample.

        Parameters
        ----------
        data : Mapping[str, Any]
            Dictionary-like sample containing at least ``image_key``
            and ``mask_key`` tensors with layout ``(C, H, W)``.

        Returns
        -------
        Dict[str, Any]
            A shallow copy of ``data`` with resized image and mask.

        Raises
        ------
        KeyError
            If ``image_key`` or ``mask_key`` is missing from ``data``.
        ValueError
            If either tensor does not have exactly three dimensions.
        """
        if self.image_key not in data:
            raise KeyError(
                f"Image key '{self.image_key}' not found in the input sample. "
                f"Available keys: {list(data.keys())}")
        if self.mask_key not in data:
            raise KeyError(
                f"Mask key '{self.mask_key}' not found in the input sample. "
                f"Available keys: {list(data.keys())}")
        
        d: Dict[str, Any] = dict(data)
        image: torch.Tensor = d[self.image_key]
        mask: torch.Tensor = d[self.mask_key]

        if image.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor (C, H, W) for image, "
                f"got shape {tuple(image.shape)}")
        if mask.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor (C, H, W) for mask, "
                f"got shape {tuple(mask.shape)}")

        h, w = int(image.shape[-2]), int(image.shape[-1])
        new_h, new_w = self._compute_target_size(h, w)

        # F.interpolate requires float tensors for smooth modes
        if not image.is_floating_point(): image = image.float()   
        if not mask.is_floating_point(): mask = mask.float()
    
        img_4d = image.unsqueeze(0)
        mask_4d = mask.unsqueeze(0)

        resized_img = self._resize(img_4d, size=(new_h, new_w), mode=self.image_mode, is_mask=False)
        resized_mask = self._resize(mask_4d, size=(new_h, new_w), mode=self.mask_mode, is_mask=True)

        d[self.image_key] = resized_img.squeeze(0)
        d[self.mask_key] = resized_mask.squeeze(0)

        return d
    
    def _compute_target_size(self, h: int, w: int) -> Tuple[int, int]:
        """
        Compute the target ``(new_h, new_w)`` preserving aspect ratio.

        Parameters
        ----------
        h, w : int
            Original height and width.

        Returns
        -------
        Tuple[int, int]
            New height and width such that
            ``max(new_h, new_w) == image_size`` and both are >= 1.
        """
        if h == w: return self.image_size, self.image_size
        if h > w: return self.image_size, max(1, round(w * (self.image_size / h)))
        return max(1, round(h * (self.image_size / w))), self.image_size

    def _resize(self, tensor: torch.Tensor, size: Tuple[int, int], mode: str, is_mask: bool) -> torch.Tensor:
        """
        Wrap ``F.interpolate`` with the correct kwargs for the given mode.

        Parameters
        ----------
        tensor : torch.Tensor
            4D tensor of shape ``(1, C, H, W)``.
        size : Tuple[int, int]
            Target ``(H, W)``.
        mode : str
            Interpolation mode passed to ``F.interpolate``.
        is_mask : bool
            Whether the tensor is a mask. Antialiasing is disabled for
            masks regardless of ``self.antialias``.

        Returns
        -------
        torch.Tensor
            Resized 4D tensor.
        """
        kwargs: Dict[str, Any] = {"size": size, "mode": mode}
        if mode in ("bilinear", "bicubic"):
            kwargs["align_corners"] = self._image_align_corners
            if self.antialias and not is_mask:
                kwargs["antialias"] = True
        # For 'area' and 'nearest', neither align_corners nor antialias
        # are accepted by PyTorch, so we intentionally leave them out.
        return F.interpolate(tensor, **kwargs)        

def get_polyp_transforms(args: Mapping[str, Any], sam_image_dim: Optional[int] = None) -> Tuple[Compose, Compose]:
    """
    Build training and validation/test transforms for polyp segmentation.

    Args:
        args: Dictionary containing configuration parameters.
            - 'image_key': Key for image in data dict (e.g., 'image')
            - 'mask_key': Key for mask in data dict (e.g., 'mask')
        sam_image_dim: Target image size for resize and pad/crop (e.g., 352).
            Must be a positive integer.

    Returns:
        Tuple of (train_transforms, val_test_transforms).

    Raises:
        ValueError: If sam_image_dim is None or not a positive integer.
        KeyError: If 'image_key' or 'mask_key' is missing from args.
    """
    # --- Validate sam_image_dim -----------------------------------------
    if sam_image_dim is None: raise ValueError("`sam_image_dim` must be provided (e.g., 1024). Got None.")
    if not isinstance(sam_image_dim, int) or sam_image_dim <= 0: raise ValueError(f"`sam_image_dim` must be a positive int, got {sam_image_dim!r}.")
    if 'image_key' not in args: raise KeyError("`args` is missing required key 'image_key'.")
    if "mask_key" not in args: raise KeyError("`args` is missing required key 'mask_key'.")

    KEYS = [args['image_key'], args['mask_key']]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # -------------------------------------------------------------------------
    # Training transforms (with data augmentation)
    # -------------------------------------------------------------------------
    train_transforms = Compose([
        LoadImageD(KEYS),
        EnsureChannelFirstD(keys=KEYS),
        ScaleIntensityRangeD(keys=[args['image_key']], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityD(keys=[args['image_key']], subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        ResizeWithRatio(image_key=KEYS[0], mask_key=KEYS[1], image_size=sam_image_dim, image_mode='bicubic', mask_mode='nearest', antialias=True),
        ResizeWithPadOrCropd(keys=KEYS, spatial_size=(sam_image_dim, sam_image_dim),mode='constant'),
        RandFlipD(keys=KEYS, prob=0.5, spatial_axis=0),
        RandFlipD(keys=KEYS, prob=0.5, spatial_axis=1),
        RandRotate90D(keys=KEYS, prob=0.5, spatial_axes=(0, 1)),
        EnsureTyped(keys=KEYS)
    ])

    # -------------------------------------------------------------------------
    # Validation/Test transforms (without data augmentation)
    # -------------------------------------------------------------------------
    val_test_transforms = Compose([
        LoadImageD(KEYS),
        EnsureChannelFirstD(keys=KEYS),
        ScaleIntensityRangeD(keys=[args['image_key']], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityD(keys=[args['image_key']], subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        ResizeWithRatio(image_key=KEYS[0], mask_key=KEYS[1], image_size=sam_image_dim, image_mode='bicubic', mask_mode='nearest', antialias=True),
        ResizeWithPadOrCropd(keys=KEYS, spatial_size=(sam_image_dim, sam_image_dim), mode='constant'),
        EnsureTyped(keys=KEYS)
    ])
    
    return train_transforms, val_test_transforms

# ===========================================================================
# (Optional) Transform variants to try later
# ===========================================================================
#
# Baseline:
#   - ScaleIntensityRangeD + NormalizeIntensityD
#   - ResizeWithRatio + ResizeWithPadOrCropd
#   - RandFlipD (H + V)
#   - RandRotate90D
#
# Future improvements to try (one at a time):
#
#   1) RandRotateD(keys=KEYS, range_x=30 * math.pi / 180,
#                  prob=0.5, keep_size=True)
#      → Replace RandRotate90D with continuous rotation.
#
#   2) RandZoomD(keys=KEYS, min_zoom=0.8, max_zoom=1.2,
#                mode=('bilinear', 'nearest'),
#                prob=0.5, padding_mode='constant', align_corners=False)
#      → Place AFTER ResizeWithPadOrCropd (on fixed-size inputs).
#
#   3) RandAdjustContrastD(keys=[image_key], prob=0.3, gamma=(0.7, 1.5))
#      → Place BEFORE ScaleIntensityRangeD (on raw [0, 255]).
#
#   4) RandGaussianNoiseD(keys=[image_key], prob=0.2, mean=0.0, std=0.05)
#      → Place BEFORE ScaleIntensityRangeD (on raw [0, 255]).
#
#   5) RandShiftIntensityD(keys=[image_key], offsets=0.1, prob=0.5)
#      → Place BETWEEN ScaleIntensityRangeD and NormalizeIntensityD
#        (on [0, 1] values).
#
#   6) RandGaussianSmoothD(keys=[image_key], prob=0.1,
#                          sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0))
#      → Place BEFORE ScaleIntensityRangeD.