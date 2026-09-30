"""
Polyp segmentation transforms.

This module contains transforms for polyp segmentation:
- ResizeWithRatio: Resize image and mask preserving aspect ratio.
- get_polyp_transform: Build training and validation/test transforms.
"""

import torch.nn.functional as F
from monai.transforms import MapTransform
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    ScaleIntensityRangeD,
    NormalizeIntensityD,
    ResizeWithPadOrCropd,
    RandFlipD,
    RandRotate90D,
    EnsureTyped,
)


class ResizeWithRatio(MapTransform):
    """
    Resize 2D image and mask preserving aspect ratio.

    - Image is resized with bilinear interpolation (smooth).
    - Mask is resized with nearest interpolation (preserves binary values).
    - Padding/cropping to a fixed square size is handled downstream
      by ResizeWithPadOrCropd.

    Input shapes:
        image: (C, H, W)     e.g., (3, 529, 622)
        mask:  (1, H, W)     e.g., (1, 529, 622)

    Output shapes:
        image: (C, new_h, new_w)
        mask:  (1, new_h, new_w)
        where max(new_h, new_w) == image_size.
    """
    
    def __init__(self, image_key, mask_key, image_size, image_mode='bilinear', mask_mode='nearest'):
        super().__init__(keys=[image_key, mask_key])
        self.image_key = image_key
        self.mask_key = mask_key
        self.image_size = image_size
        self.image_mode = image_mode
        self.mask_mode = mask_mode
        self._image_align_corners = (None if image_mode in ('nearest', 'area') else False)

    def __call__(self, data):
        if self.image_key not in data:
            raise KeyError(f"Image key '{self.image_key}' not found in data")
        if self.mask_key not in data:
            raise KeyError(f"Mask key '{self.mask_key}' not found in data")
        
        d = dict(data)
        image = d[self.image_key]
        mask = d[self.mask_key]

        if image.ndim != 3:
            raise ValueError(f"Expected 3D tensor (C, H, W) for image, but got shape {tuple(image.shape)}")
        if mask.ndim != 3:
            raise ValueError(f"Expected 3D tensor (C, H, W) for mask, but got shape {tuple(mask.shape)}")

        h, w = image.shape[-2], image.shape[-1]

        if h == w:
            new_h = new_w = self.image_size
        elif h > w:
            new_h = self.image_size
            new_w = round(w * (self.image_size / h))
        else:
            new_w = self.image_size
            new_h = round(h * (self.image_size / w))
            
        img_4d = image.unsqueeze(0)
        mask_4d = mask.unsqueeze(0)
        resized_img = F.interpolate(img_4d, size=(new_h, new_w), mode=self.image_mode, align_corners=self._image_align_corners)
        resized_mask = F.interpolate(mask_4d, size=(new_h, new_w), mode=self.mask_mode)
        d[self.image_key] = resized_img.squeeze(0)
        d[self.mask_key] = resized_mask.squeeze(0)

        return d


def get_polyp_transforms(args, sam_image_dim=None, get_bbox=False):
    """
    Build training and validation/test transforms for polyp segmentation.
    
    Args:
        args: Dictionary containing configuration parameters.
            - 'image_key': Key for image in data dict (e.g., 'image')
            - 'mask_key': Key for mask in data dict (e.g., 'mask')
        sam_image_dim: Target image size for SAM model (e.g., 1024).
        get_bbox: Whether to generate bounding boxes (not used in this project).
    
    Returns:
        Tuple of (train_transforms, val_test_transforms)
    """
    KEYS = [args['image_key'], args['mask_key']]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # -------------------------------------------------------------------------
    # Training transforms (with data augmentation)
    # -------------------------------------------------------------------------
    train_transforms = Compose([
        LoadImageD(KEYS),
        EnsureChannelFirstD(keys=KEYS),
        ScaleIntensityRangeD(
            keys=[args['image_key']],
            a_min=0.0,
            a_max=255.0,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        NormalizeIntensityD(
            keys=[args['image_key']],
            subtrahend=IMAGENET_MEAN,
            divisor=IMAGENET_STD,
            channel_wise=True
        ),
        ResizeWithRatio(
            image_key=KEYS[0],
            mask_key=KEYS[1],
            image_size=sam_image_dim,
            image_mode='bilinear',
            mask_mode='nearest'
        ),
        ResizeWithPadOrCropd(
            keys=KEYS,
            spatial_size=(sam_image_dim, sam_image_dim),
            mode='constant'
        ),
        RandFlipD(keys=KEYS, prob=0.5, spatial_axis=0),
        RandFlipD(keys=KEYS, prob=0.5, spatial_axis=1),
        RandRotate90D(keys=KEYS, prob=0.5, spatial_axes=(0, 1)),
        EnsureTyped(keys=KEYS, track_meta=False)
    ])

    # -------------------------------------------------------------------------
    # Validation/Test transforms (without data augmentation)
    # -------------------------------------------------------------------------
    val_test_transforms = Compose([
        LoadImageD(KEYS),
        EnsureChannelFirstD(keys=KEYS),
        ScaleIntensityRangeD(
            keys=[args['image_key']],
            a_min=0.0,
            a_max=255.0,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        NormalizeIntensityD(
            keys=[args['image_key']],
            subtrahend=IMAGENET_MEAN,
            divisor=IMAGENET_STD,
            channel_wise=True
        ),
        ResizeWithRatio(
            image_key=KEYS[0],
            mask_key=KEYS[1],
            image_size=sam_image_dim,
            image_mode='bilinear',
            mask_mode='nearest'
        ),
        ResizeWithPadOrCropd(
            keys=KEYS,
            spatial_size=(sam_image_dim, sam_image_dim),
            mode='constant'
        ),
        EnsureTyped(keys=KEYS, track_meta=True)
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