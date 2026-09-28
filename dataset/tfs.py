from monai.transforms import MapTransform
import copy
from monai.transforms import Compose, LoadImageD, EnsureTyped, ResizeD, ResizeWithPadOrCropd, OrientationD, RandFlipD, RandRotate90D, RandCropByPosNegLabelD, EnsureChannelFirstD, NormalizeIntensityD, ScaleIntensityD, SpacingD, TransposeD


class ResizeWithRatio(MapTransform):
    """
    Resize image and mask with aspect ratio preservation, using
    different interpolation modes for image (bilinear) and mask (nearest).

    Suitable for 2D data (e.g., polyp segmentation). For 3D data,
    use 'trilinear' for image and 'nearest' for mask.
    """
        
    def __init__(self, image_key, mask_key, image_size, image_mode='bilinear', mask_mode='nearest'):
        super().__init__(keys=[image_key, mask_key])
        self.image_key = image_key
        self.mask_key = mask_key
        self.image_size = image_size
        self.image_mode = image_mode
        self.mask_mode = mask_mode

    def __call__(self, data):
        d = dict(data)
        image = d[self.image_key]
        h, w = image.shape[1:3]

        if h == w:
            new_h = new_w = self.image_size
        elif h > w:
            new_h = self.image_size
            new_w = int(w * (self.image_size / h))
        else:
            new_w = self.image_size
            new_h = int(h * (self.image_size / w))
            
        image_resize = ResizeD(keys=[self.image_key], spatial_size=(new_h, new_w), mode=self.image_mode)
        mask_resize = ResizeD(keys=[self.mask_key], spatial_size=(new_h, new_w), mode=self.mask_mode)

        d = image_resize(d)
        d = mask_resize(d)
        
        return d

def get_mslesseg_transform(args, sam_image_dim = None, get_bbox=False):
    KEYS = [args['image_key'], args['mask_key']]
    transform_train = [
        LoadImageD(KEYS),
        EnsureChannelFirstD(keys=KEYS),
        NormalizeIntensityD(keys = (args['image_key'])),
        ScaleIntensityD(keys = (args['image_key'])),
        SpacingD(KEYS, pixdim=(1., 1., 1.), mode = ("bilinear", "nearest")),
        OrientationD(KEYS, axcodes = args['axcodes']),
        RandCropByPosNegLabelD(KEYS, label_key=args['mask_key'], spatial_size=(-1,-1,args['num_slices']), pos=args['pos_rand_crop'], neg=args['neg_rand_crop']),
        EnsureTyped(keys=KEYS, device=args['device']),
        ResizeWithRatio(keys = KEYS, image_size = sam_image_dim, mode = 'trilinear'),
        ResizeWithPadOrCropd(keys = KEYS, spatial_size = (sam_image_dim,sam_image_dim,-1), mode = 'constant'),
        RandFlipD(KEYS, prob = 0.5, spatial_axis=0),
        RandRotate90D(KEYS, prob=0.5, spatial_axes=(0,1))
    ]

    train_transforms = Compose(transform_train)
        
    transform_val_test = [
        LoadImageD(KEYS),
        EnsureChannelFirstD(keys=KEYS),
        NormalizeIntensityD(keys = (args['image_key'])),
        ScaleIntensityD(keys = (args['image_key'])),
        OrientationD(KEYS, axcodes = args['axcodes']),
        EnsureTyped(keys=KEYS, device=args['device']),
        ResizeWithRatio(keys = KEYS, image_size = sam_image_dim, mode = 'trilinear'),
        ResizeWithPadOrCropd(keys = KEYS, spatial_size = (sam_image_dim,sam_image_dim,-1), mode = 'constant'),
        TransposeD(keys=KEYS, indices=(0,3,1,2))
    ]
    
    test_val_transforms = Compose(transform_val_test)
    
    return train_transforms, test_val_transforms