"""
Dataset Utilities
=================

This module builds train, validation, and test DataLoaders for polyp
segmentation. It is a thin wrapper around `get_polyp_dataset` and
`get_test_polyp_dataset` that produces ready-to-use PyTorch DataLoaders.

Assumes the underlying dataset is built on MONAI's CacheDataset,
which caches transformed samples in RAM. Cache behavior is controlled
via `args['cache_rate']`.
"""

import torch
from torch.utils.data import DataLoader

from dataset.polyp_dataset import (
    get_polyp_dataset,
    get_test_polyp_dataset,
)


# ===========================================================================
# Supported test datasets (from the polyp split)
# ===========================================================================

TEST_DATASETS = [
    'Kvasir-SEG',
    'CVC-ClinicDB',
    'CVC-300',
    'CVC-ColonDB',
    'ETIS-LaribPolypDB',
]


# ===========================================================================
# Train / Validation loaders
# ===========================================================================

def get_train_val_dataloaders(args, img_dim):
    """
    Build training and validation DataLoaders for polyp segmentation.

    Parameters
    ----------
    args : dict
        Configuration dictionary. Must contain:
            - 'split_path' : path to the JSON split file
            - 'n_fold'     : fold index
            - 'cache_rate' : fraction of the dataset cached in RAM (0.0 - 1.0)
            - 'Batch_size' : training batch size
            - 'nW'         : number of workers for training loader
            - 'nW_eval'    : number of workers for validation loader
            - 'image_key'  : key of the image in the data dict
            - 'mask_key'   : key of the mask in the data dict
    img_dim : int or None
        Target image size for SAM (e.g., 1024).

    Returns
    -------
    (train_loader, val_loader) : tuple of DataLoader
    """
    if args.get('task', 'polyp') != 'polyp':
        raise ValueError(
            f"Unsupported task: '{args.get('task')}'. "
            "This module only supports task='polyp'."
        )

    ds_train, ds_val = get_polyp_dataset(args, sam_image_dim=img_dim)

    # For CacheDataset, num_workers > 0 is safe (cache lives in RAM).
    n_workers_train = int(args.get('nW', 0))
    n_workers_val = int(args.get('nW_eval', 0))
    use_pin = n_workers_train > 0   # pin_memory only helps when workers > 0

    train_loader = DataLoader(
        ds_train,
        batch_size=int(args['Batch_size']),
        shuffle=True,
        num_workers=n_workers_train,
        drop_last=True,
        pin_memory=use_pin,
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=n_workers_val,
        drop_last=False,
        pin_memory=(n_workers_val > 0),
    )

    return train_loader, val_loader


# ===========================================================================
# Test loader — single dataset
# ===========================================================================

def get_test_dataset(args, img_dim, dataset_origin=None):
    """
    Build a test DataLoader for polyp segmentation.

    Parameters
    ----------
    args : dict
        Same structure as in `get_dataset`.
    img_dim : int or None
        Target image size for SAM (e.g., 1024).
    dataset_origin : str or None
        If given, only samples from this test dataset are used.
        Options:
            'Kvasir-SEG', 'CVC-ClinicDB', 'CVC-300',
            'CVC-ColonDB', 'ETIS-LaribPolypDB'.
        If None, all test samples are returned as a single loader.

    Returns
    -------
    test_loader : DataLoader
    """
    ds_test = get_test_polyp_dataset(
        args, sam_image_dim=img_dim, dataset_origin=dataset_origin
    )

    n_workers_test = int(args.get('nW_eval', 0))

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        num_workers=n_workers_test,
        drop_last=False,
        pin_memory=(n_workers_test > 0),
    )

    return test_loader


# ===========================================================================
# Test loaders — all datasets (dictionary)
# ===========================================================================

def get_all_test_datasets(args, img_dim):
    """
    Build a DataLoader for each test dataset.

    This is the standard setup for polyp papers (PraNet, Polyp-PVT, ...)
    where results are reported per-dataset.

    Parameters
    ----------
    args : dict
        Same structure as in `get_dataset`.
    img_dim : int or None
        Target image size for SAM (e.g., 1024).

    Returns
    -------
    dict
        Mapping from dataset name to DataLoader.
    """
    test_loaders = {}
    for name in TEST_DATASETS:
        test_loaders[name] = get_test_dataset(args, img_dim,
                                              dataset_origin=name)
    return test_loaders