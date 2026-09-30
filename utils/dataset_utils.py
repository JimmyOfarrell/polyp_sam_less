"""
Dataset Utilities
=================

This module builds train and validation DataLoaders for polyp segmentation.
It is a thin wrapper around `get_polyp_dataset` that produces ready-to-use
PyTorch DataLoaders.
"""

import torch
from torch.utils.data import DataLoader

from dataset.polyp_dataset import get_polyp_dataset


def get_dataset(args, img_dim):
    """
    Build training and validation DataLoaders for polyp segmentation.

    Parameters
    ----------
    args : dict
        Configuration dictionary. Must contain:
            - 'split_path'  : path to the JSON split file
            - 'n_fold'      : fold index
            - 'cache_dir'   : directory for on-disk cache (PersistentDataset)
            - 'Batch_size'  : training batch size
            - 'nW'          : number of workers for training loader
            - 'nW_eval'     : number of workers for validation loader
            - 'image_key'   : key of the image in the data dict (e.g., 'image')
            - 'mask_key'    : key of the mask in the data dict (e.g., 'mask')
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

    # ------------------------------------------------------------------
    # Training DataLoader
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        ds_train,
        batch_size=int(args['Batch_size']),
        shuffle=True,
        num_workers=int(args.get('nW', 0)),
        drop_last=True,
        pin_memory=False,   # transfer handled in the training loop
    )

    # ------------------------------------------------------------------
    # Validation DataLoader
    # ------------------------------------------------------------------
    val_loader = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.get('nW_eval', 0)),
        drop_last=False,
        pin_memory=False,
    )

    return train_loader, val_loader