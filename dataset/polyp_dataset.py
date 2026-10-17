from __future__ import annotations
import sys
import json
from monai.data import CacheDataset
from pathlib import Path
from dataset.polyp_transforms import get_polyp_transforms

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

class PolypDataset(CacheDataset):
    def __init__(self, split_path, section, num_fold, transforms, dataset_origin=None, cache_num=sys.maxsize, cache_rate=0.0, num_workers= 0, progress= False):
        self.section = section
        self.num_fold = num_fold
        self.dataset_origin = dataset_origin
        self.cache_rate = cache_rate

        data = self._generate_data_list(split_path)
        self.num_samples = len(data)

        super().__init__(data, transforms, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers, progress=progress)

    def _generate_data_list(self, split_path):
        with open(split_path) as fp: data = json.load(fp)
        datalist = []
        if self.section == 'train': datalist = data[f'fold{self.num_fold}']['train']
        elif self.section == 'validation': datalist = data[f'fold{self.num_fold}']['val']
        elif self.section == 'test':
            datalist = data['test']
            if self.dataset_origin is not None:
                filtered = [sample for sample in datalist if sample.get("dataset_origin") == self.dataset_origin]
                if not filtered:
                    available = sorted({s.get("dataset_origin") for s in data["test"]})
                    raise ValueError(
                        f"No samples found for dataset_origin="
                        f"'{self.dataset_origin}'. "
                        f"Available origins: {available}")
                datalist = filtered
        else:
            raise ValueError(
                f"Unsupported section: '{self.section}'. "
                "Available options are ['train', 'validation', 'test'].")
        
        for sample in datalist:
            sample['image'] = str(PROJECT_ROOT / sample['image'])
            sample['mask']  = str(PROJECT_ROOT / sample['mask'])

        return datalist

    def __repr__(self):
        parts = [
            f"section='{self.section}'",
            f"num_fold={self.num_fold}",
        ]
        if self.dataset_origin is not None:
            parts.append(f"dataset_origin='{self.dataset_origin}'")
        parts.append(f"num_samples={getattr(self, 'num_samples', '?')}")
        parts.append(f"cache_rate={self.cache_rate}")
        return f"PolypDataset({', '.join(parts)})"

def get_polyp_dataset(args, sam_image_dim=None):
    split_path = args['split_path']
    num_fold = args['n_fold']
    cache_rate = args.get('cache_rate', 0.0)

    train_transforms, val_transforms = get_polyp_transforms(args, sam_image_dim)

    ds_train = PolypDataset(split_path=split_path, section='train', num_fold=num_fold, transforms=train_transforms, cache_rate=cache_rate)
    ds_val = PolypDataset(split_path=split_path, section='validation', num_fold=num_fold, transforms=val_transforms, cache_rate=cache_rate)

    return ds_train, ds_val

def get_test_polyp_dataset(args, sam_image_dim=None, dataset_origin=None):
    split_path = args['split_path']
    num_fold = args['n_fold']
    cache_rate = 0.0

    _, val_test_transforms = get_polyp_transforms(args, sam_image_dim)
    ds_test = PolypDataset(split_path=split_path, section='test', num_fold=num_fold, transforms=val_test_transforms, dataset_origin=dataset_origin, cache_rate=cache_rate)

    return ds_test