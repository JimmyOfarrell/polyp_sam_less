"""
Generate the JSON split file for the Polyp Segmentation project.

This script reads the TRAIN and TEST datasets from disk, performs a
source-aware 90/10 train/val split on the TRAIN dataset, and writes
the resulting split to a JSON file.

The script is location-independent: it determines all paths relative
to its own file location, so it can be run from anywhere.

Project layout (assumed):
    polyp_sam_less/                  <- project root
    ├── data/
    │   ├── download_datasets.py
    │   ├── generate_split.py        <- this file
    │   ├── split.json               <- output
    │   ├── TrainDataset/
    │   └── TestDataset/
    └── ...
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path configuration (relative to this script's location)
# ---------------------------------------------------------------------------

# Directory containing this script: polyp_sam_less/data/
DATA_DIR = Path(__file__).resolve().parent

# Project root: polyp_sam_less/
PROJECT_ROOT = DATA_DIR.parent

# Dataset directories (filesystem paths).
TRAIN_DATASET_DIR = DATA_DIR / "TrainDataset"
TEST_DATASET_DIR = DATA_DIR / "TestDataset"

# Output file for the split JSON.
OUTPUT_SPLIT_PATH = DATA_DIR / "split.json"

# Prefix used inside the JSON for file paths (relative to PROJECT_ROOT).
# This ensures the JSON remains portable regardless of where the project
# is placed on disk.
TRAIN_RELATIVE_PREFIX = "data/TrainDataset"
TEST_RELATIVE_PREFIX = "data/TestDataset"

# ---------------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------------

VAL_RATIO = 0.10
RANDOM_SEED = 42

# Dataset name constants (used consistently in the JSON).
KVASIR_NAME = "Kvasir-SEG"
CLINICDB_NAME = "CVC-ClinicDB"

# Mapping from TestDataset folder name -> dataset_origin name in the JSON.
# This handles the "Kvasir" -> "Kvasir-SEG" rename.
TEST_FOLDER_TO_ORIGIN = {
    "Kvasir": KVASIR_NAME,
    "CVC-ClinicDB": CLINICDB_NAME,
    "CVC-300": "CVC-300",
    "CVC-ColonDB": "CVC-ColonDB",
    "ETIS-LaribPolypDB": "ETIS-LaribPolypDB",
}

# Mapping from TestDataset folder name -> distribution type.
TEST_FOLDER_TO_DISTRIBUTION = {
    "Kvasir": "in_distribution",
    "CVC-ClinicDB": "in_distribution",
    "CVC-300": "out_of_distribution",
    "CVC-ColonDB": "out_of_distribution",
    "ETIS-LaribPolypDB": "out_of_distribution",
}

# Supported image extensions.
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_files(folder: Path) -> List[Path]:
    """Return a sorted list of image files inside a folder."""
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def _relative_path_for_json(file_path: Path) -> str:
    """
    Build a POSIX-style relative path (from PROJECT_ROOT) for the JSON.

    If the file is outside PROJECT_ROOT, fall back to the absolute path.
    This guarantees the JSON is portable when the project is moved.
    """
    try:
        rel = file_path.resolve().relative_to(PROJECT_ROOT.resolve())
        return rel.as_posix()
    except ValueError:
        # File is not under PROJECT_ROOT (unusual, but handled gracefully).
        return file_path.resolve().as_posix()


def _classify_train_source(stem: str) -> Optional[str]:
    """
    Classify a TRAIN image filename by dataset origin.

    Strategy (robust to naming variations):
        - Purely numeric names (e.g., '1', '10', '100') -> CVC-ClinicDB
        - All other names (e.g., 'cju...', 'cjy...',
          'cjz...', 'ck2...')                           -> Kvasir-SEG
    """
    if stem.isdigit():
        return CLINICDB_NAME
    return KVASIR_NAME


# ---------------------------------------------------------------------------
# TRAIN dataset processing
# ---------------------------------------------------------------------------

def _collect_train_pairs() -> Dict[str, List[Tuple[Path, Path]]]:
    """
    Read the TRAIN dataset and group (image, mask) pairs by source.

    Returns a dict mapping dataset_origin -> list of (image_path, mask_path).
    """
    image_dir = TRAIN_DATASET_DIR / "image"
    mask_dir = TRAIN_DATASET_DIR / "masks"

    if not image_dir.exists():
        raise FileNotFoundError(f"Missing TRAIN image folder: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing TRAIN mask folder: {mask_dir}")

    image_files = _list_files(image_dir)
    if not image_files:
        raise RuntimeError(f"No images found in: {image_dir}")

    # Build a mask lookup by stem for fast pairing.
    mask_lookup = {m.stem: m for m in _list_files(mask_dir)}

    grouped: Dict[str, List[Tuple[Path, Path]]] = defaultdict(list)
    missing_masks: List[str] = []

    for img_path in image_files:
        origin = _classify_train_source(img_path.stem)
        mask_path = mask_lookup.get(img_path.stem)
        if mask_path is None:
            missing_masks.append(img_path.name)
            continue
        grouped[origin].append((img_path, mask_path))

    if missing_masks:
        print(f"[WARNING] {len(missing_masks)} images have no matching mask.")
        for name in missing_masks[:5]:
            print(f"          - {name}")

    return grouped


def _split_train_val(
    pairs: List[Tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """Split a list of pairs into (train, val) with a fixed seed."""
    pairs = list(pairs)  # local copy
    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_val = max(1, int(round(len(pairs) * val_ratio)))
    val = pairs[:n_val]
    train = pairs[n_val:]
    return train, val


def _pair_to_record(pair: Tuple[Path, Path], origin: str) -> Dict:
    """Build a JSON record for a TRAIN sample."""
    img_path, mask_path = pair
    return {
        "image": _relative_path_for_json(img_path),
        "mask": _relative_path_for_json(mask_path),
        "dataset_origin": origin,
    }


# ---------------------------------------------------------------------------
# TEST dataset processing
# ---------------------------------------------------------------------------

def _collect_test_records() -> List[Dict]:
    """
    Read the TEST dataset and build a flat list of records.

    For each sub-dataset folder (e.g., CVC-300, Kvasir, ...), the
    'images' and 'masks' sub-folders are read and paired by stem.
    """
    if not TEST_DATASET_DIR.exists():
        raise FileNotFoundError(f"Missing TEST dataset folder: {TEST_DATASET_DIR}")

    records: List[Dict] = []

    for subfolder_name in TEST_FOLDER_TO_ORIGIN.keys():
        sub_dir = TEST_DATASET_DIR / subfolder_name
        image_dir = sub_dir / "images"
        mask_dir = sub_dir / "masks"

        if not image_dir.exists():
            print(f"[WARNING] Missing TEST image folder: {image_dir}")
            continue
        if not mask_dir.exists():
            print(f"[WARNING] Missing TEST mask folder: {mask_dir}")
            continue

        image_files = _list_files(image_dir)
        mask_lookup = {m.stem: m for m in _list_files(mask_dir)}

        origin = TEST_FOLDER_TO_ORIGIN[subfolder_name]
        distribution = TEST_FOLDER_TO_DISTRIBUTION[subfolder_name]

        n_added = 0
        for img_path in image_files:
            mask_path = mask_lookup.get(img_path.stem)
            if mask_path is None:
                continue
            records.append({
                "image": _relative_path_for_json(img_path),
                "mask": _relative_path_for_json(mask_path),
                "dataset_origin": origin,
                "distribution": distribution,
            })
            n_added += 1

        print(f"[INFO] {subfolder_name:20s} -> {n_added:4d} records "
              f"({distribution}, origin='{origin}')")

    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_split(
    output_path: Path = OUTPUT_SPLIT_PATH,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> Dict:
    """
    Generate the split JSON file.

    Returns the split dictionary that was written to disk.
    """
    print("=" * 70)
    print("  GENERATING SPLIT FILE")
    print("=" * 70)
    print(f"[INFO] PROJECT_ROOT     : {PROJECT_ROOT}")
    print(f"[INFO] DATA_DIR         : {DATA_DIR}")
    print(f"[INFO] TRAIN_DATASET_DIR: {TRAIN_DATASET_DIR}")
    print(f"[INFO] TEST_DATASET_DIR : {TEST_DATASET_DIR}")
    print(f"[INFO] OUTPUT           : {output_path}")

    # ------------------------------------------------------------------
    # Step 1: TRAIN dataset -> group by source
    # ------------------------------------------------------------------
    print("\n[STEP 1] Reading TRAIN dataset ...")
    grouped = _collect_train_pairs()
    for origin, pairs in grouped.items():
        print(f"[INFO] {origin:15s} : {len(pairs)} pairs")

    # ------------------------------------------------------------------
    # Step 2: Per-source 90/10 split, then combine
    # ------------------------------------------------------------------
    print("\n[STEP 2] Splitting each source into train/val ...")
    train_records: List[Dict] = []
    val_records: List[Dict] = []

    for idx, origin in enumerate(sorted(grouped.keys())):
        pairs = grouped[origin]
        tr, va = _split_train_val(pairs, val_ratio, seed=seed + idx)
        print(f"[INFO] {origin:15s} : train={len(tr):4d}  val={len(va):3d}  "
              f"(total={len(pairs)})")

        train_records.extend(_pair_to_record(p, origin) for p in tr)
        val_records.extend(_pair_to_record(p, origin) for p in va)

    # Shuffle the combined train and val lists once more.
    rng = random.Random(seed)
    rng.shuffle(train_records)
    rng.shuffle(val_records)

    print(f"\n[INFO] Combined train : {len(train_records)} records")
    print(f"[INFO] Combined val   : {len(val_records)} records")

    # ------------------------------------------------------------------
    # Step 3: TEST dataset -> flat list
    # ------------------------------------------------------------------
    print("\n[STEP 3] Reading TEST dataset ...")
    test_records = _collect_test_records()
    print(f"\n[INFO] Total test     : {len(test_records)} records")

    # ------------------------------------------------------------------
    # Step 4: Assemble the full split dictionary
    # ------------------------------------------------------------------
    split = {
        "fold0": {
            "train": train_records,
            "val": val_records,
        },
        "test": test_records,
    }

    # ------------------------------------------------------------------
    # Step 5: Write to disk
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Split file written to: {output_path}")

    # ------------------------------------------------------------------
    # Step 6: Verification summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  VERIFICATION SUMMARY")
    print("=" * 70)

    # TRAIN breakdown.
    train_by_origin: Dict[str, int] = defaultdict(int)
    for rec in train_records:
        train_by_origin[rec["dataset_origin"]] += 1
    print("\n[fold0.train]")
    for origin, count in sorted(train_by_origin.items()):
        print(f"  {origin:20s} : {count}")

    # VAL breakdown.
    val_by_origin: Dict[str, int] = defaultdict(int)
    for rec in val_records:
        val_by_origin[rec["dataset_origin"]] += 1
    print("\n[fold0.val]")
    for origin, count in sorted(val_by_origin.items()):
        print(f"  {origin:20s} : {count}")

    # TEST breakdown.
    test_by_key: Dict[Tuple[str, str], int] = defaultdict(int)
    for rec in test_records:
        key = (rec["distribution"], rec["dataset_origin"])
        test_by_key[key] += 1
    print("\n[test]")
    for (dist, origin), count in sorted(test_by_key.items()):
        print(f"  [{dist:20s}] {origin:20s} : {count}")

    print("\n[INFO] Totals:")
    print(f"  train = {len(train_records)}")
    print(f"  val   = {len(val_records)}")
    print(f"  test  = {len(test_records)}")
    print("=" * 70)

    return split


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_split(
        output_path=OUTPUT_SPLIT_PATH,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED,
    )