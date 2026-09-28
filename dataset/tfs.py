from monai.transforms import MapTransform
import copy
from monai.transforms import Compose, LoadImageD, EnsureTyped, ResizeD, ResizeWithPadOrCropd, OrientationD, RandFlipD, RandRotate90D, RandCropByPosNegLabelD, EnsureChannelFirstD, NormalizeIntensityD, ScaleIntensityD, SpacingD, TransposeD


class ResizeWithRatio(MapTransform):
    def __init__(self, keys, image_size, mode='bilinear'):
        super().__init__(keys)
        self.image_size = image_size
        self.mode=mode

    def __call__(self, data):
        d = copy.deepcopy(data)
        image = d[self.keys[0]]
        h, w = image.shape[1:3]
        if h != w:
            if h > w:
                new_h = self.image_size
                scale_factor = new_h / h
                new_w = int(w * scale_factor)
            else:
                new_w = self.image_size
                scale_factor = new_w / w
                new_h = int(h * scale_factor)
            new_size = (new_h, new_w) if image.ndim == 3 else (new_h, new_w, image.shape[3])
        else:
            new_size = (self.image_size, self.image_size) if image.ndim == 3 else (self.image_size, self.image_size, image.shape[3])
            
        resize_transform = ResizeD(keys=self.keys, spatial_size=new_size, mode=self.mode)
        
        d = resize_transform(d)

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