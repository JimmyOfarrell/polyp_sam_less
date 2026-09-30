"""
Polyp Dataset
=============

This module defines the PolypDataset class for polyp segmentation.

PolypDataset is built on top of MONAI's PersistentDataset, which caches
transformed samples on disk. This is more memory-efficient than
CacheDataset but slightly slower due to disk I/O.

Supported sections:
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

Disk usage:
    The full polyp dataset (train + val + test) consumes approximately
    37.7 GB of disk space when cached (float32). Consider using a
    temporary directory (e.g., /kaggle/tmp/polyp_cache) if disk space
    is limited.
"""

import json
from typing import Optional
from dataset.polyp_transforms import get_polyp_transforms
from monai.data.utils import pickle_hashing
from monai.data import PersistentDataset


class PolypDataset(PersistentDataset):
    """
    PersistentDataset for polyp segmentation.

    Data is cached on disk rather than in RAM, which allows training
    on large datasets with limited memory (e.g., Colab Free / Kaggle).

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
    cache_dir : str
        Directory where cached tensors are stored on disk.
        Recommended locations:
            - Kaggle : '/kaggle/tmp/polyp_cache'
            - Colab  : '/content/polyp_cache'
            - Local  : './data/cache'
    dataset_origin : str or None, optional
        If provided AND section == 'test', only samples whose
        'dataset_origin' field matches this value are kept.
        Examples: 'Kvasir-SEG', 'CVC-ClinicDB', 'CVC-300',
                  'CVC-ColonDB', 'ETIS-LaribPolypDB'.
        Ignored for other sections.
    hash_func : callable or None, optional
        Function to compute a hash for each sample. If None, MONAI's
        default hash (based on the sample's content) is used.
        Recommended: leave as None unless you have a specific reason.
    """

    def __init__(
        self,
        split_path: str,
        section: str,
        num_fold: int,
        transforms,
        cache_dir: str,
        dataset_origin: Optional[str] = None,
        hash_func=None,
    ):
        self.section = section
        self.num_fold = num_fold
        self.dataset_origin = dataset_origin

        data = self._generate_data_list(split_path)
        self.num_samples = len(data)

        if hash_func is None: hash_func = pickle_hashing

        super().__init__(
            data=data,
            transform=transforms,
            cache_dir=cache_dir,
            hash_func=hash_func,
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
        parts.append(f"cache_dir='{self.cache_dir}'")
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
            - 'cache_dir'  : directory for on-disk cache
    sam_image_dim : int or None
        Target image size for SAM (e.g., 1024). If None, no resize.

    Returns
    -------
    (ds_train, ds_val) : tuple of PolypDataset
    """

    split_path = args['split_path']
    num_fold = args['n_fold']
    cache_dir = args['cache_dir']

    train_transforms, val_transforms = get_polyp_transforms(
        args, sam_image_dim
    )

    ds_train = PolypDataset(
        split_path=split_path,
        section='train',
        num_fold=num_fold,
        transforms=train_transforms,
        cache_dir=f"{cache_dir}/train",
    )
    ds_val = PolypDataset(
        split_path=split_path,
        section='validation',
        num_fold=num_fold,
        transforms=val_transforms,
        cache_dir=f"{cache_dir}/val",
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
            - 'cache_dir'  : directory for on-disk cache
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
    cache_dir = args['cache_dir']

    _, val_test_transforms = get_polyp_transforms(args, sam_image_dim)

    # Build a separate cache subdir per dataset_origin (if filtering).
    if dataset_origin is not None:
        # Sanitize name (avoid slashes in path).
        safe_name = dataset_origin.replace('/', '_').replace(' ', '_')
        subdir = f"test_{safe_name}"
    else:
        subdir = "test_all"

    ds_test = PolypDataset(
        split_path=split_path,
        section='test',
        num_fold=num_fold,
        transforms=val_test_transforms,
        cache_dir=f"{cache_dir}/{subdir}",
        dataset_origin=dataset_origin,
    )
    return ds_test