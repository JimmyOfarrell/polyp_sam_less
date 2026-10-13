"""
Dataset Utilities
=================

This module builds train, validation, and test DataLoaders for polyp
segmentation. It is a thin wrapper around ``get_polyp_dataset`` and
``get_test_polyp_dataset`` that produces ready-to-use PyTorch
``DataLoader`` objects.

The underlying datasets are built on MONAI's ``CacheDataset``, which
caches transformed samples in RAM. Cache behaviour is controlled via
``args['cache_rate']``.
"""

from __future__ import annotations
import torch
from typing import Any, Dict, List, Mapping, Optional, Tuple
from torch.utils.data import DataLoader
from dataset.polyp_dataset import get_polyp_dataset, get_test_polyp_dataset


# ===========================================================================
# Supported test datasets (from the polyp split)
# ===========================================================================
TEST_DATASETS: List[str] = ["Kvasir-SEG", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB"]

# ===========================================================================
# Train / Validation loaders
# ===========================================================================
def get_train_val_dataloaders(args: Mapping[str, Any], img_dim: int) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders for polyp segmentation.

    The training loader shuffles and drops the last incomplete batch;
    the validation loader does neither.

    Parameters
    ----------
    args : Mapping[str, Any]
        Configuration dictionary. Must contain:

        * ``'split_path'`` : path to the JSON split file.
        * ``'n_fold'``     : fold index.
        * ``'image_key'``  : key for the image in the sample dict.
        * ``'mask_key'``   : key for the mask in the sample dict.
        * ``'Batch_size'`` : training batch size.

        Optional:

        * ``'cache_rate'`` : cache fraction (0.0 to 1.0). Defaults to 0.0.
        * ``'nW'``         : number of workers for the training loader.
          Defaults to 0.
        * ``'nW_eval'``    : number of workers for the validation loader.
          Defaults to 0.
        * ``'task'``       : kept for safety; if present, must be
          ``'polyp'``.

    img_dim : int
        Target image size for resize and pad/crop (e.g., 352). Must be
        a positive integer.

    Returns
    -------
    train_loader : DataLoader
        Training loader (shuffle=True, drop_last=True).
    val_loader : DataLoader
        Validation loader (shuffle=False, drop_last=False,
        batch_size=1).

    Raises
    ------
    ValueError
        If ``args['task']`` is present and not ``'polyp'``.
    """
    if args.get('task', 'polyp') != 'polyp':
        raise ValueError(
            f"Unsupported task: '{args.get('task')}'. "
            "This module only supports task='polyp'.")

    n_workers_train: int = int(args.get("nW", 0))
    n_workers_val: int = int(args.get("nW_eval", 0))
    batch_size: int = int(args["Batch_size"])

    ds_train, ds_val = get_polyp_dataset(args, sam_image_dim=img_dim)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=n_workers_train, drop_last=True, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=n_workers_val, drop_last=False, pin_memory=True)
    
    return train_loader, val_loader

# ===========================================================================
# Test loader — single dataset
# ===========================================================================
def get_test_dataset(args: Mapping[str, Any], img_dim: int, dataset_origin: Optional[str] = None) -> DataLoader:
    """
    Build a test DataLoader for polyp segmentation.

    A single loader is returned, covering either the entire test split
    or a subset filtered by ``dataset_origin``.

    Parameters
    ----------
    args : Mapping[str, Any]
        Same structure as in :func:`get_train_val_dataloaders`. Only
        ``'split_path'``, ``'image_key'``, ``'mask_key'``, and
        ``'cache_rate'`` are actually used here.

    img_dim : int
        Target image size for resize and pad/crop (e.g., 352). Must be
        a positive integer.

    dataset_origin : str or None, optional
        If provided, only samples from this test dataset are used.
        Options:

        * ``'Kvasir-SEG'``
        * ``'CVC-ClinicDB'``
        * ``'CVC-300'``
        * ``'CVC-ColonDB'``
        * ``'ETIS-LaribPolypDB'``

        If ``None``, all test samples are returned as a single loader.

    Returns
    -------
    test_loader : DataLoader
        Test loader (shuffle=False, drop_last=False, batch_size=1).
    """
    n_workers_test: int = int(args.get("nW_eval", 0))

    ds_test = get_test_polyp_dataset(args, sam_image_dim=img_dim, dataset_origin=dataset_origin)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=n_workers_test, drop_last=False,pin_memory=True)

    return test_loader

# ===========================================================================
# Test loaders — all datasets (dictionary)
# ===========================================================================
def get_all_test_datasets(args: Mapping[str, Any], img_dim: int) -> Dict[str, DataLoader]:
    """
    Build a separate DataLoader for each test dataset.

    This matches the standard setup used in polyp segmentation papers
    (PraNet, Polyp-PVT, ...), where metrics are reported per-dataset
    for cross-dataset generalisation.

    Parameters
    ----------
    args : Mapping[str, Any]
        Same structure as in :func:`get_train_val_dataloaders`.

    img_dim : int
        Target image size for resize and pad/crop (e.g., 352). Must be
        a positive integer.

    Returns
    -------
    Dict[str, DataLoader]
        Mapping from dataset name (one of :data:`TEST_DATASETS`) to a
        dedicated DataLoader.
    """
    test_loaders: Dict[str, DataLoader] = {}
    for name in TEST_DATASETS: test_loaders[name] = get_test_dataset(args, img_dim, dataset_origin=name)

    return test_loaders