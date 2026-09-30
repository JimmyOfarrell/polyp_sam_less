"""
Polyp Dataset
=============

This module defines the PolypDataset class for polyp segmentation.

PolypDataset is built on top of MONAI's CacheDataset and reads its list
of samples from a JSON split file. Supported sections:

    - 'train'      : uses data['fold{N}']['train']
    - 'validation' : uses data['fold{N}']['val']
    - 'test'       : uses data['test'] (at root level)

The 'test' section supports optional filtering by 'dataset_origin'
(e.g., 'Kvasir-SEG', 'CVC-ClinicDB', 'CVC-300', ...) for per-dataset
evaluation.

Expected JSON structure:

    {
      "fold0": {
        "train": [{"image": "...", "mask": "...", "dataset_origin": "..."}, ...],
        "val":   [{"image": "...", "mask": "...", "dataset_origin": "..."}, ...]
      },
      "test": [
        {"image": "...", "mask": "...", "dataset_origin": "...", "distribution": "..."},
        ...
      ]
    }
"""

import sys
import json
from typing import Optional
from monai.data import CacheDataset
from dataset.polyp_transforms import get_polyp_transforms


class PolypDataset(CacheDataset):
    """
    CacheDataset for polyp segmentation.

    Parameters
    ----------
    split_path : str
        Path to the JSON split file.
    section : str
        One of {'train', 'validation', 'test'}.
    num_fold : int
        Fold index (used only when section is 'train' or 'validation').
    transforms : Compose
        MONAI Compose pipeline applied to each sample.
    dataset_origin : str or None, optional
        If provided AND section == 'test', only samples whose
        'dataset_origin' field matches this value are kept.
        Examples: 'Kvasir-SEG', 'CVC-ClinicDB', 'CVC-300',
                  'CVC-ColonDB', 'ETIS-LaribPolypDB'.
        Ignored for other sections.
    cache_num : int, optional
        Maximum number of items to cache in RAM (default: unbounded).
    cache_rate : float, optional
        Fraction of the dataset to cache (0.0 to 1.0). Default 0.0.
    num_workers : int, optional
        Number of workers for parallel loading during __init__. Default 0.
    """

    def __init__(
        self,
        split_path: str,
        section: str,
        num_fold: int,
        transforms,
        dataset_origin: Optional[str] = None,
        cache_num: int = sys.maxsize,
        cache_rate: float = 0.0,
        num_workers: int = 0,
    ):
        self.section = section
        self.num_fold = num_fold
        self.dataset_origin = dataset_origin
        self.cache_rate = cache_rate

        data = self._generate_data_list(split_path)
        self.num_samples = len(data)

        super().__init__(
            data,
            transforms,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def _generate_data_list(self, split_path: str):
        """
        Read the JSON split file and return the list of samples
        corresponding to `self.section`.

        Raises
        ------
        ValueError
            - If `section` is not one of {'train', 'validation', 'test'}.
            - If `section='test'` and `dataset_origin` is given but no
              matching samples exist in the split file.
        """
        with open(split_path) as fp:
            data = json.load(fp)

        if self.section == 'train':
            datalist = data[f'fold{self.num_fold}']['train']

        elif self.section == 'validation':
            datalist = data[f'fold{self.num_fold}']['val']

        elif self.section == 'test':
            datalist = data['test']
            if self.dataset_origin is not None:
                datalist = [
                    sample for sample in datalist
                    if sample.get('dataset_origin') == self.dataset_origin
                ]
                if not datalist:
                    available = sorted({
                        s.get('dataset_origin') for s in data['test']
                    })
                    raise ValueError(
                        f"No samples found for dataset_origin="
                        f"'{self.dataset_origin}'. "
                        f"Available origins: {available}"
                    )

        else:
            raise ValueError(
                f"Unsupported section: '{self.section}'. "
                "Available options are ['train', 'validation', 'test']."
            )

        return datalist

    def __repr__(self) -> str:
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
    """
    Build training and validation datasets for polyp segmentation.

    Parameters
    ----------
    args : dict
        Configuration dictionary. Must contain:
            - 'split_path' : path to the JSON split file
            - 'n_fold'     : fold index
            - 'cache_rate' : cache fraction (0.0 to 1.0)
    sam_image_dim : int or None
        Target image size for SAM (e.g., 1024). If None, no resize.

    Returns
    -------
    (ds_train, ds_val) : tuple of PolypDataset
    """
    split_path = args['split_path']
    num_fold = args['n_fold']
    cache_rate = args.get('cache_rate', 0.0)

    train_transforms, val_transforms = get_polyp_transforms(
        args, sam_image_dim
    )

    ds_train = PolypDataset(
        split_path=split_path,
        section='train',
        num_fold=num_fold,
        transforms=train_transforms,
        cache_rate=cache_rate,
    )
    ds_val = PolypDataset(
        split_path=split_path,
        section='validation',
        num_fold=num_fold,
        transforms=val_transforms,
        cache_rate=cache_rate,
    )
    return ds_train, ds_val

def get_test_polyp_dataset(args, sam_image_dim=None, dataset_origin=None):
    """
    Build the test dataset for polyp segmentation.

    Parameters
    ----------
    args : dict
        Configuration dictionary. Must contain:
            - 'split_path' : path to the JSON split file
            - 'n_fold'     : fold index (ignored for test but kept for API)
            - 'cache_rate' : cache fraction (0.0 to 1.0)
    sam_image_dim : int or None
        Target image size for SAM (e.g., 1024). If None, no resize.
    dataset_origin : str or None
        If given, only samples whose 'dataset_origin' matches
        are included. Options:
            'Kvasir-SEG', 'CVC-ClinicDB', 'CVC-300',
            'CVC-ColonDB', 'ETIS-LaribPolypDB'.
        If None, all test samples are returned.

    Returns
    -------
    PolypDataset
    """
    split_path = args['split_path']
    num_fold = args['n_fold']
    cache_rate = args.get('cache_rate', 0.0)

    _, val_test_transforms = get_polyp_transforms(args, sam_image_dim)

    ds_test = PolypDataset(
        split_path=split_path,
        section='test',
        num_fold=num_fold,
        transforms=val_test_transforms,
        dataset_origin=dataset_origin,
        cache_rate=cache_rate,
    )
    return ds_test