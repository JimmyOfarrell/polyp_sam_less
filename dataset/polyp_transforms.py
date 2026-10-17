import torch.nn.functional as F
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
    def __init__(self, image_key, mask_key, image_size, image_mode="bilinear", mask_mode="nearest", antialias=False):
        super().__init__(keys=[image_key, mask_key])

        if image_mode not in ("bilinear", "bicubic", "area", "nearest"): raise ValueError(f"Unsupported image_mode '{image_mode}'. ""Choose from 'bilinear', 'bicubic', 'area', 'nearest'.")
        if mask_mode not in ("nearest", "bilinear"): raise ValueError(f"Unsupported mask_mode '{mask_mode}'. ""Use 'nearest' to preserve binary values.")

        self.image_key = image_key
        self.mask_key = mask_key
        self.image_size = int(image_size)
        self.image_mode = image_mode
        self.mask_mode = mask_mode
        self.antialias = bool(antialias)

        # align_corners is only valid for bilinear / bicubic
        self._image_align_corners = (None if image_mode in ("nearest", "area") else False)

    def __call__(self, data):
        if self.image_key not in data:
            raise KeyError(
                f"Image key '{self.image_key}' not found in the input sample. "
                f"Available keys: {list(data.keys())}")
        if self.mask_key not in data:
            raise KeyError(
                f"Mask key '{self.mask_key}' not found in the input sample. "
                f"Available keys: {list(data.keys())}")
        
        d = dict(data)
        image = d[self.image_key]
        mask = d[self.mask_key]

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
    
    def _compute_target_size(self, h, w):
        if h == w: return self.image_size, self.image_size
        if h > w: return self.image_size, max(1, round(w * (self.image_size / h)))
        return max(1, round(h * (self.image_size / w))), self.image_size

    def _resize(self, tensor, size, mode, is_mask):
        kwargs = {"size": size, "mode": mode}
        if mode in ("bilinear", "bicubic"):
            kwargs["align_corners"] = self._image_align_corners
            if self.antialias and not is_mask:
                kwargs["antialias"] = True
        # For 'area' and 'nearest', neither align_corners nor antialias
        # are accepted by PyTorch, so we intentionally leave them out.
        return F.interpolate(tensor, **kwargs)        

def get_polyp_transforms(args, sam_image_dim= None):
    if sam_image_dim is None: raise ValueError("`sam_image_dim` must be provided (e.g., 1024). Got None.")
    if not isinstance(sam_image_dim, int) or sam_image_dim <= 0: raise ValueError(f"`sam_image_dim` must be a positive int, got {sam_image_dim!r}.")
    if 'image_key' not in args: raise KeyError("`args` is missing required key 'image_key'.")
    if "mask_key" not in args: raise KeyError("`args` is missing required key 'mask_key'.")

    KEYS = [args['image_key'], args['mask_key']]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transforms = Compose([
        LoadImageD(KEYS, reader="PILReader", reverse_indexing=False),
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

    val_test_transforms = Compose([
        LoadImageD(KEYS, reader="PILReader", reverse_indexing=False),
        EnsureChannelFirstD(keys=KEYS),
        ScaleIntensityRangeD(keys=[args['image_key']], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityD(keys=[args['image_key']], subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        ResizeWithRatio(image_key=KEYS[0], mask_key=KEYS[1], image_size=sam_image_dim, image_mode='bicubic', mask_mode='nearest', antialias=True),
        ResizeWithPadOrCropd(keys=KEYS, spatial_size=(sam_image_dim, sam_image_dim), mode='constant'),
        EnsureTyped(keys=KEYS)
    ])
    
    return train_transforms, val_test_transforms