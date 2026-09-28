import sys
import json
from monai.data import CacheDataset
from dataset.tfs import get_mslesseg_transform


class MSLesSeg(CacheDataset):
    def __init__(self, root_dir, split_path, mask, section, num_fold, transforms, seed=100, cache_num=sys.maxsize, cache_rate=0., num_workers=0):
        self.mask = mask
        self.section = section
        self.num_fold = num_fold    
        data = self._generate_data_list(split_path)

        super().__init__(data, transforms, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)

    def _generate_data_list(self, split_path):
        with open(split_path) as fp:
            data = json.load(fp)

        if self.section == 'validation':
            datalist = data[f'fold{self.num_fold}']['val']
        elif self.section == 'train':
            datalist = data[f'fold{self.num_fold}']['train']
        elif self.section == 'test':
            datalist = data['test']
        else: 
            raise ValueError(
                f"Unsupported section: {self.section}, "
                "available options are ['train', 'validation', 'test']."
            )
        return datalist


def get_mslesseg_dataset(args, sam_image_dim = None):
    split_path = args['split_path']
    mask = args['mask_key']
    num_fold = args['n_fold']
    if args['sam_zeroshot']:
        get_bbox = True
    else:
        get_bbox = False
    transform_train, transform_val_test = get_mslesseg_transform(args, sam_image_dim, get_bbox=get_bbox)
    ds_train = MSLesSeg(args['data_dir'], split_path, mask, section='train', num_fold=num_fold, transforms=transform_train, cache_rate=args['cache_rate'])
    ds_val = MSLesSeg(args['data_dir'], split_path, mask, section='validation', num_fold=num_fold, transforms=transform_val_test, cache_rate=args['cache_rate'])
    return ds_train, ds_val


def get_test_mslesseg_dataset(args, sam_image_dim = None):
    split_path = args['split_path']
    mask = args['mask_key']
    num_fold = args['n_fold']
    if args['sam_zeroshot']:
        get_bbox = True
    else:
        get_bbox = False
    transform_train, transform_val_test = get_mslesseg_transform(args, sam_image_dim, get_bbox=get_bbox)
    ds_test = MSLesSeg(args['data_dir'], split_path, mask, section='test', num_fold=num_fold, transforms=transform_val_test, cache_rate=args['cache_rate'])
    return ds_test