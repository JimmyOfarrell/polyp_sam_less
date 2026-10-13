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

from __future__ import annotations
import sys
import json
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union
from monai.data import CacheDataset
from monai.transforms import Compose
from pathlib import Path
from dataset.polyp_transforms import get_polyp_transforms

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
PathLike = Union[str, Path]
SectionType = Literal["train", "validation", "test"]
TransformType = Union[Compose, Sequence[Callable[[Dict[str, Any]], Dict[str, Any]]]]

# ---------------------------------------------------------------------------
# Project root: assumed to be the parent of the folder containing this file.
# If the file lives at <project>/dataset/polyp_dataset.py, PROJECT_ROOT = <project>.
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class PolypDataset(CacheDataset):
    """
    CacheDataset for polyp segmentation.

    Reads a JSON split file and exposes one section ('train',
    'validation', or 'test') as a MONAI dataset. Sample paths in the
    JSON are treated as relative to ``PROJECT_ROOT`` and converted to
    absolute paths on construction.

    Parameters
    ----------
    split_path : Union[str, Path]
        Path to the JSON split file.
    section : {'train', 'validation', 'test'}
        Which subset of the split to load.
    num_fold : int
        Fold index (used only when ``section`` is ``'train'`` or
        ``'validation'``). Ignored for ``'test'``.
    transforms : Union[Compose, Sequence[Callable]]
        MONAI transform pipeline (or list of transforms) applied to
        each sample.
    dataset_origin : str, optional
        If provided AND ``section == 'test'``, only samples whose
        ``'dataset_origin'`` field matches this value are kept.
        Examples: ``'Kvasir-SEG'``, ``'CVC-ClinicDB'``, ``'CVC-300'``,
        ``'CVC-ColonDB'``, ``'ETIS-LaribPolypDB'``.
        Ignored for other sections.
    cache_num : int, optional
        Maximum number of items to cache in RAM. Defaults to
        ``sys.maxsize`` (unbounded).
    cache_rate : float, optional
        Fraction of the dataset to cache (0.0 to 1.0). Defaults to 0.0
        (no caching). Recommended 0.0 on Colab / Kaggle to keep RAM
        usage low.
    num_workers : int, optional
        Number of workers used during ``__init__`` for parallel loading
        and caching. Defaults to 0.
    progress : bool, optional
        Whether to display a progress bar while caching. Defaults to
        ``False`` to keep Colab / Kaggle logs clean.

    Attributes
    ----------
    section : str
        The section loaded.
    num_fold : int
        The fold index used (if applicable).
    dataset_origin : str or None
        The origin filter applied (test only), or ``None``.
    num_samples : int
        Number of samples in the loaded section.
    """


    def __init__(self, split_path: PathLike, section: SectionType, num_fold: int, transforms: TransformType, dataset_origin: Optional[str] = None, cache_num: int = sys.maxsize, cache_rate: float = 0.0, num_workers: int = 0, progress: bool = False) -> None:
        self.section: SectionType = section
        self.num_fold: int = num_fold
        self.dataset_origin: Optional[str] = dataset_origin
        self.cache_rate = cache_rate

        data: List[Dict[str, Any]] = self._generate_data_list(split_path)
        self.num_samples: int = len(data)

        super().__init__(data, transforms, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)

    def _generate_data_list(self, split_path: PathLike) -> List[Dict[str, Any]]:
        """
        Read the JSON split file and return the list of samples
        corresponding to ``self.section``.

        Sample paths (``'image'`` and ``'mask'``) are converted from
        relative to absolute by joining with ``PROJECT_ROOT``. The
        conversion happens in-place on the freshly loaded JSON data.

        Parameters
        ----------
        split_path : Union[str, Path]
            Path to the JSON split file.

        Returns
        -------
        List[Dict[str, Any]]
            List of samples, each containing at least the keys
            ``'image'``, ``'mask'``, and (usually) ``'dataset_origin'``.
            Paths are absolute.

        Raises
        ------
        ValueError
            - If ``self.section`` is not one of
              ``{'train', 'validation', 'test'}``.
            - If ``self.section == 'test'`` and ``dataset_origin`` was
              provided but no samples match it.
        """
        with open(split_path) as fp: data = json.load(fp)
        datalist: List[Dict[str, Any]]
        if self.section == 'train': datalist = data[f'fold{self.num_fold}']['train']
        elif self.section == 'validation': datalist = data[f'fold{self.num_fold}']['val']
        elif self.section == 'test':
            datalist = data['test']
            if self.dataset_origin is not None:
                filtered: List[Dict[str, Any]] = [sample for sample in datalist if sample.get("dataset_origin") == self.dataset_origin]
                if not filtered:
                    available: List[str] = sorted({s.get("dataset_origin") for s in data["test"]})
                    raise ValueError(
                        f"No samples found for dataset_origin="
                        f"'{self.dataset_origin}'. "
                        f"Available origins: {available}")
                datalist = filtered
        else:
            raise ValueError(
                f"Unsupported section: '{self.section}'. "
                "Available options are ['train', 'validation', 'test'].")
        
        # Convert relative paths to absolute.
        for sample in datalist:
            sample['image'] = str(PROJECT_ROOT / sample['image'])
            sample['mask']  = str(PROJECT_ROOT / sample['mask'])

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

def get_polyp_dataset(args: Mapping[str, Any], sam_image_dim: Optional[int] = None) -> Tuple["PolypDataset", "PolypDataset"]:
    """
    Build training and validation datasets for polyp segmentation.

    Both datasets are built from the same JSON split file. The training
    dataset applies data augmentation; the validation dataset does not.

    Parameters
    ----------
    args : Mapping[str, Any]
        Configuration dictionary. Must contain:

        * ``'split_path'`` : path to the JSON split file.
        * ``'n_fold'``     : fold index (used to select the fold in
          the split file).
        * ``'image_key'``  : key for the image in the sample dict.
        * ``'mask_key'``   : key for the mask in the sample dict.

        Optional:

        * ``'cache_rate'`` : cache fraction (0.0 to 1.0). Defaults to
          0.0 (no caching).

    sam_image_dim : int or None, optional
        Target image size for resize and pad/crop (e.g., 352). Must be
        a positive integer. ``None`` raises ``ValueError`` inside
        ``get_polyp_transforms``.

    Returns
    -------
    ds_train : PolypDataset
        Training dataset with augmentation.
    ds_val : PolypDataset
        Validation dataset without augmentation.

    Raises
    ------
    KeyError
        If ``'split_path'`` or ``'n_fold'`` is missing from ``args``.
    ValueError
        If ``sam_image_dim`` is ``None`` or not a positive integer.
    """
    split_path = args['split_path']
    num_fold = args['n_fold']
    cache_rate = args.get('cache_rate', 0.0)

    train_transforms, val_transforms = get_polyp_transforms(args, sam_image_dim)

    ds_train = PolypDataset(split_path=split_path, section='train', num_fold=num_fold, transforms=train_transforms, cache_rate=cache_rate)
    ds_val = PolypDataset(split_path=split_path, section='validation', num_fold=num_fold, transforms=val_transforms, cache_rate=cache_rate)

    return ds_train, ds_val

def get_test_polyp_dataset(args: Mapping[str, Any], sam_image_dim: Optional[int] = None, dataset_origin: Optional[str] = None) -> "PolypDataset":
    """
    Build the test dataset for polyp segmentation.

    The test section is loaded from the root-level ``'test'`` key of
    the split file. Optionally, samples can be filtered by their
    ``'dataset_origin'`` field for per-dataset evaluation.

    Parameters
    ----------
    args : Mapping[str, Any]
        Configuration dictionary. Must contain:

        * ``'split_path'`` : path to the JSON split file.
        * ``'image_key'``  : key for the image in the sample dict.
        * ``'mask_key'``   : key for the mask in the sample dict.

        Optional:

        * ``'cache_rate'`` : cache fraction (0.0 to 1.0). Defaults to
          0.0 (no caching).

    sam_image_dim : int or None, optional
        Target image size for resize and pad/crop (e.g., 352). Must be
        a positive integer. ``None`` raises ``ValueError`` inside
        ``get_polyp_transforms``.

    dataset_origin : str or None, optional
        If provided, only samples whose ``'dataset_origin'`` matches
        this value are included. Options:

        * ``'Kvasir-SEG'``
        * ``'CVC-ClinicDB'``
        * ``'CVC-300'``
        * ``'CVC-ColonDB'``
        * ``'ETIS-LaribPolypDB'``

        If ``None``, all test samples are returned.

    Returns
    -------
    PolypDataset
        Test dataset (no augmentation).

    Raises
    ------
    KeyError
        If ``'split_path'`` is missing from ``args``.
    ValueError
        If ``sam_image_dim`` is ``None`` or not a positive integer, or
        if ``dataset_origin`` is given but no matching samples exist.
    """
    split_path = args['split_path']
    num_fold = args['n_fold']
    cache_rate = args.get('cache_rate', 0.0)

    _, val_test_transforms = get_polyp_transforms(args, sam_image_dim)
    ds_test = PolypDataset(split_path=split_path, section='test', num_fold=num_fold, transforms=val_test_transforms, dataset_origin=dataset_origin, cache_rate=cache_rate)
    return ds_test