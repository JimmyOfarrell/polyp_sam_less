"""
Polyp Segmentation - Dataset Downloader & Validator
====================================================

This module downloads, extracts, and validates the training and test
datasets for the Polyp Segmentation project. Both datasets are fetched
from Google Drive as zip archives.

The script is location-independent: it determines all paths relative
to its own file location, so it can be run from anywhere.

Project layout (assumed):
    polyp_sam_less/                  <- project root
    ├── data/
    │   ├── download_datasets.py     <- this file
    │   ├── generate_split.py
    │   ├── split.json
    │   ├── TrainDataset/
    │   └── TestDataset/
    └── ...

Expected structure of the TRAIN dataset after extraction:
    TrainDataset/
        image/          -> 1450 PNG images (singular folder name 'image')
            - Numeric names (1.png, 10.png, ...)      : CVC-ClinicDB (~550)
            - Alphanumeric names (cju*, cjy*, cjz*,
              ck2*, ...)                              : Kvasir-SEG    (~900)
        masks/          -> 1450 PNG masks (same naming as images)

Expected structure of the TEST dataset after extraction:
    TestDataset/
        CVC-300/            images/ (60)  masks/ (60)
        CVC-ClinicDB/       images/ (62)  masks/ (62)
        CVC-ColonDB/        images/ (380) masks/ (380)
        ETIS-LaribPolypDB/  images/ (196) masks/ (196)
        Kvasir/             images/ (100) masks/ (100)

Note: the TRAIN dataset uses the singular folder name 'image', while
the TEST dataset uses the plural folder name 'images'. This module
handles both cases transparently.
"""

import random
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gdown
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Path configuration (relative to this script's location)
# ---------------------------------------------------------------------------

# Directory containing this script: polyp_sam_less/data/
DATA_DIR = Path(__file__).resolve().parent

# Project root: polyp_sam_less/
PROJECT_ROOT = DATA_DIR.parent

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

# Google Drive file IDs (or full URLs).
TRAIN_GDRIVE_ID = "1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb"
TEST_GDRIVE_ID = "1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao"

# Dataset folder names.
TRAIN_DATASET_NAME = "TrainDataset"
TEST_DATASET_NAME = "TestDataset"

# Zip archive names.
TRAIN_ZIP_NAME = "TrainDataset.zip"
TEST_ZIP_NAME = "TestDataset.zip"

# Expected counts for the TRAIN dataset.
TRAIN_EXPECTED_TOTAL = 1450
TRAIN_EXPECTED_CLINICDB = 550
TRAIN_EXPECTED_KVASIR = 900

# Expected counts for the TEST dataset (folder name -> expected images).
TEST_EXPECTED_COUNTS = {
    "CVC-300": 60,
    "CVC-ClinicDB": 62,
    "CVC-ColonDB": 380,
    "ETIS-LaribPolypDB": 196,
    "Kvasir": 100,
}

# Folder name variants for images inside a dataset.
IMAGE_FOLDER_NAMES = ("image", "images")
MASK_FOLDER_NAMES = ("masks", "mask")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Sampling configuration for integrity checks.
SAMPLE_SIZE_INTEGRITY = 50


# ===========================================================================
# SECTION 1: SHARED HELPERS (download, extract, security, file utilities)
# ===========================================================================

def _safe_remove_file(path: Path) -> None:
    """Safely remove a file, printing a warning on failure instead of raising."""
    path = Path(path)
    if not path.exists():
        return
    try:
        path.unlink()
        print(f"[INFO] Deleted file: {path}")
    except OSError as exc:
        print(f"[WARNING] Could not delete file {path}: {exc}")


def _list_image_files(folder: Path) -> List[Path]:
    """Return a sorted list of image files inside a folder."""
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def _is_within_directory(base: Path, target: Path) -> bool:
    """Return True if 'target' is inside 'base' (prevents Zip Slip attacks)."""
    base = base.resolve()
    target = target.resolve()
    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False


def _download_from_gdrive(file_id_or_url: str, zip_path: Path) -> bool:
    """Download a file from Google Drive (accepts raw ID or full URL)."""
    if "http" not in file_id_or_url:
        url = f"https://drive.google.com/uc?id={file_id_or_url}"
    else:
        url = file_id_or_url

    print(f"[INFO] Downloading from Google Drive ...")
    print(f"[INFO] URL/ID: {url}")

    try:
        output = gdown.download(url=url, output=str(zip_path), quiet=False, fuzzy=True)
    except Exception as exc:
        print(f"[ERROR] Download failed with exception: {exc}")
        print("[TIP] Ensure the file is shared as 'Anyone with the link'.")
        return False

    if output is None or not zip_path.exists():
        print("[ERROR] Download failed. gdown returned None or file not found.")
        return False

    if zip_path.stat().st_size == 0:
        print("[ERROR] Downloaded file is empty (0 bytes).")
        _safe_remove_file(zip_path)
        return False

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Downloaded: {zip_path} ({size_mb:.2f} MB)")
    return True


def _extract_zip_safely(zip_path: Path, extract_to: Path) -> bool:
    """Extract a zip archive with Zip Slip protection."""
    print(f"[INFO] Extracting archive to: {extract_to}")
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            for member in zf.namelist():
                member_path = extract_to / member
                if not _is_within_directory(extract_to, member_path):
                    print(f"[ERROR] Zip Slip attack detected: {member}")
                    return False
            zf.extractall(str(extract_to))
    except zipfile.BadZipFile:
        print(f"[ERROR] Not a valid zip archive: {zip_path}")
        return False
    except Exception as exc:
        print(f"[ERROR] Extraction failed: {exc}")
        return False

    print("[SUCCESS] Extraction complete.")
    return True


def _flatten_nested_dir(dataset_path: Path, folder_name: str) -> Path:
    """Flatten a nested folder with the same name (e.g., TrainDataset/TrainDataset/)."""
    nested = dataset_path / folder_name
    if nested.exists() and nested.is_dir():
        print(f"[INFO] Flattening nested directory: {nested}")
        for item in nested.iterdir():
            shutil.move(str(item), str(dataset_path))
        try:
            nested.rmdir()
        except OSError:
            print(f"[WARNING] Could not remove nested dir: {nested}")
    return dataset_path


def _find_dataset_root(search_dir: Path, expected_name: str) -> Optional[Path]:
    """Search for the expected dataset folder inside 'search_dir'."""
    direct = search_dir / expected_name
    if direct.exists() and direct.is_dir():
        return direct
    candidates = list(search_dir.glob(f"**/{expected_name}"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _resolve_subfolder(parent: Path, candidates: Tuple[str, ...]) -> Optional[Path]:
    """Return the first existing sub-folder whose name matches any candidate."""
    for name in candidates:
        path = parent / name
        if path.exists() and path.is_dir():
            return path
    return None


def _check_folder_pair(
    parent_dir: Path,
    image_candidates: Tuple[str, ...] = IMAGE_FOLDER_NAMES,
    mask_candidates: Tuple[str, ...] = MASK_FOLDER_NAMES,
) -> Tuple[Optional[Path], Optional[Path], bool]:
    """
    Check for image/mask sub-folders inside 'parent_dir' using flexible
    folder-name candidates. Returns (image_dir, mask_dir, ok).
    """
    image_dir = _resolve_subfolder(parent_dir, image_candidates)
    mask_dir = _resolve_subfolder(parent_dir, mask_candidates)

    ok = True
    if image_dir is None:
        print(f"[ERROR] No image folder found in: {parent_dir} "
              f"(looked for: {image_candidates})")
        ok = False
    else:
        print(f"[OK] Image folder found: {image_dir}")

    if mask_dir is None:
        print(f"[ERROR] No mask folder found in: {parent_dir} "
              f"(looked for: {mask_candidates})")
        ok = False
    else:
        print(f"[OK] Mask folder found : {mask_dir}")

    return image_dir, mask_dir, ok


def _check_filename_pairing(image_files: List[Path], mask_files: List[Path]) -> bool:
    """Verify that each image has a matching mask with the same stem."""
    image_stems = {p.stem for p in image_files}
    mask_stems = {p.stem for p in mask_files}

    missing_masks = image_stems - mask_stems
    missing_images = mask_stems - image_stems

    ok = True
    if missing_masks:
        print(f"[ERROR] {len(missing_masks)} images without matching mask.")
        for name in sorted(missing_masks)[:5]:
            print(f"        - {name}.png")
        ok = False
    if missing_images:
        print(f"[ERROR] {len(missing_images)} masks without matching image.")
        for name in sorted(missing_images)[:5]:
            print(f"        - {name}.png")
        ok = False

    if ok:
        print("[OK] Every image has a matching mask (identical stem).")
    return ok


def _check_integrity(
    image_files: List[Path],
    mask_files: List[Path],
    sample_size: int = SAMPLE_SIZE_INTEGRITY,
    seed: int = 42,
) -> bool:
    """
    Sample images and masks to:
      - verify they are readable (not corrupted)
      - compare image/mask dimensions
      - inspect mask value range (should be binary)
    """
    random.seed(seed)
    n = min(sample_size, len(image_files))
    if n == 0:
        print("[WARNING] No image files to sample.")
        return False

    sampled_images = random.sample(image_files, n)
    mask_lookup = {m.stem: m for m in mask_files}

    corrupt: List[Tuple[str, str]] = []
    dim_mismatch: List[str] = []
    mask_ranges: List[Tuple[int, int]] = []
    unique_counts: List[int] = []

    for img_path in sampled_images:
        mask_path = mask_lookup.get(img_path.stem)
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                img_size = img.size

            if mask_path is not None and mask_path.exists():
                with Image.open(mask_path) as m:
                    m.verify()
                with Image.open(mask_path) as m:
                    mask_size = m.size
                    arr = np.array(m)
                    mask_ranges.append((int(arr.min()), int(arr.max())))
                    unique_counts.append(int(len(np.unique(arr))))

                if img_size != mask_size:
                    dim_mismatch.append(
                        f"{img_path.name}: image {img_size} vs mask {mask_size}"
                    )
        except Exception as exc:
            corrupt.append((img_path.name, str(exc)))

    ok = True

    if corrupt:
        print(f"[ERROR] {len(corrupt)} corrupted files detected (sampled):")
        for name, err in corrupt[:5]:
            print(f"        - {name}: {err}")
        ok = False
    else:
        print(f"[OK] Sampled {n} image/mask pairs loaded successfully.")

    if dim_mismatch:
        print(f"[ERROR] {len(dim_mismatch)} image/mask dimension mismatches:")
        for entry in dim_mismatch[:5]:
            print(f"        - {entry}")
        ok = False
    elif mask_ranges:
        print("[OK] Image and mask dimensions match for all sampled pairs.")

    if mask_ranges:
        vmin = min(r[0] for r in mask_ranges)
        vmax = max(r[1] for r in mask_ranges)
        print(f"[INFO] Mask value range : [{vmin}, {vmax}]")
        if vmax in (1, 255) and vmin == 0:
            print("[OK] Masks appear binary (values in {0,1} or {0,255}).")
        else:
            print("[WARNING] Mask values are not clearly binary. Please verify.")
            ok = False

    return ok


def _download_extract_pipeline(
    gdrive_id_or_url: str,
    output_dir: Path,
    dataset_name: str,
    zip_name: str,
    force_redownload: bool = False,
    force_reextract: bool = False,
    clean_existing: bool = False,
    remove_zip_after_extraction: bool = True,
) -> Optional[Path]:
    """
    Shared pipeline: download the zip, extract it, flatten if needed,
    and return the path to the dataset root (or None on failure).
    """
    output_dir = Path(output_dir).resolve()
    dataset_root = output_dir / dataset_name
    zip_path = output_dir / zip_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional cleanup.
    if clean_existing and dataset_root.exists():
        print(f"[WARNING] Removing existing dataset at: {dataset_root}")
        shutil.rmtree(dataset_root)
        dataset_root = output_dir / dataset_name

    if force_redownload and zip_path.exists():
        _safe_remove_file(zip_path)

    if force_reextract and dataset_root.exists():
        print(f"[WARNING] Removing existing dataset for re-extraction: {dataset_root}")
        shutil.rmtree(dataset_root)

    # --- Step 1: Download ---
    print("\n" + "-" * 70)
    print(f"  STEP 1: DOWNLOAD ({dataset_name})")
    print("-" * 70)

    need_download = (not dataset_root.exists()) or force_redownload
    if need_download:
        if not zip_path.exists():
            if not _download_from_gdrive(gdrive_id_or_url, zip_path):
                return None
        else:
            print(f"[INFO] Using existing zip: {zip_path}")
    else:
        print(f"[INFO] Dataset already exists at: {dataset_root}")
        print("[INFO] Skipping download. Set force_redownload=True to re-download.")

    # --- Step 2: Extract ---
    print("\n" + "-" * 70)
    print(f"  STEP 2: EXTRACT ({dataset_name})")
    print("-" * 70)

    if not dataset_root.exists():
        if not zip_path.exists():
            print(f"[ERROR] Zip not found and dataset folder missing: {zip_path}")
            return None
        if not _extract_zip_safely(zip_path, output_dir):
            return None
        dataset_root = _find_dataset_root(output_dir, dataset_name)
        if dataset_root is None:
            print(f"[ERROR] Could not locate '{dataset_name}' after extraction.")
            return None
        dataset_root = _flatten_nested_dir(dataset_root, dataset_name)

        if remove_zip_after_extraction:
            _safe_remove_file(zip_path)
    else:
        print(f"[INFO] Dataset folder already exists: {dataset_root}")
        if remove_zip_after_extraction and zip_path.exists():
            _safe_remove_file(zip_path)

    return dataset_root


# ===========================================================================
# SECTION 2: TRAIN DATASET VALIDATION
# ===========================================================================

def _classify_train_sources(image_files: List[Path]) -> Dict[str, List[str]]:
    """
    Classify TRAIN image filenames by dataset origin.

    Strategy (robust to naming variations):
        - Purely numeric names (e.g., '1', '10', '100') -> CVC-ClinicDB
        - All other names (e.g., 'cju...', 'cjy...',
          'cjz...', 'ck2...')                           -> Kvasir-SEG
    """
    clinicdb, kvasir, unknown = [], [], []
    for path in image_files:
        stem = path.stem
        if re.match(r"^\d+$", stem):
            clinicdb.append(stem)
        else:
            kvasir.append(stem)
    return {"clinicdb": clinicdb, "kvasir": kvasir, "unknown": unknown}


def _check_train_source_distribution(image_files: List[Path]) -> bool:
    """Verify counts per source dataset in the TRAIN set."""
    dist = _classify_train_sources(image_files)
    n_c = len(dist["clinicdb"])
    n_k = len(dist["kvasir"])

    print(f"[INFO] CVC-ClinicDB-like (numeric)    : {n_c} (expected ~{TRAIN_EXPECTED_CLINICDB})")
    print(f"[INFO] Kvasir-SEG-like (non-numeric)  : {n_k} (expected ~{TRAIN_EXPECTED_KVASIR})")

    ok = True
    if n_c != TRAIN_EXPECTED_CLINICDB:
        print(f"[WARNING] CVC-ClinicDB count off (expected {TRAIN_EXPECTED_CLINICDB}).")
        ok = False
    if n_k != TRAIN_EXPECTED_KVASIR:
        print(f"[WARNING] Kvasir-SEG count off (expected {TRAIN_EXPECTED_KVASIR}).")
        ok = False

    if dist["kvasir"]:
        sample = sorted(dist["kvasir"])[:5]
        print(f"[INFO] Sample Kvasir-SEG names: {sample}")

    return ok


def validate_train_dataset(dataset_root: Path) -> Dict:
    """Run a full validation pipeline on the TRAIN dataset."""
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_root}")
        return _build_result(False, dataset_root, None, None, [], [], {})

    print("\n" + "=" * 70)
    print(f"  VALIDATING TRAIN DATASET: {dataset_root}")
    print("=" * 70)

    checks: Dict[str, bool] = {}

    # 1. Folder structure (image/ or images/ ; masks/ or mask/).
    image_dir, mask_dir, structure_ok = _check_folder_pair(dataset_root)
    checks["directory_structure"] = structure_ok
    if not structure_ok:
        return _build_result(False, dataset_root, image_dir, mask_dir, [], [], checks)

    # 2. Collect files.
    image_files = _list_image_files(image_dir)
    mask_files = _list_image_files(mask_dir)
    if not image_files or not mask_files:
        print("[ERROR] No images or masks found in the TRAIN dataset.")
        return _build_result(False, dataset_root, image_dir, mask_dir,
                             image_files, mask_files, checks)

    # 3. Counts.
    print("\n--- Counts ---")
    print(f"[INFO] Images found : {len(image_files)}")
    print(f"[INFO] Masks found  : {len(mask_files)}")
    print(f"[INFO] Expected     : {TRAIN_EXPECTED_TOTAL}")
    counts_ok = (len(image_files) == TRAIN_EXPECTED_TOTAL
                 and len(mask_files) == TRAIN_EXPECTED_TOTAL
                 and len(image_files) == len(mask_files))
    if counts_ok:
        print(f"[OK] Counts match expected value ({TRAIN_EXPECTED_TOTAL}).")
    else:
        print("[WARNING] Count mismatch detected.")
    checks["counts"] = counts_ok

    # 4. Filename pairing.
    print("\n--- Filename pairing ---")
    checks["filename_pairing"] = _check_filename_pairing(image_files, mask_files)

    # 5. Source distribution.
    print("\n--- Source distribution ---")
    checks["source_distribution"] = _check_train_source_distribution(image_files)

    # 6. Integrity (sample).
    print("\n--- Integrity (sample) ---")
    checks["integrity"] = _check_integrity(image_files, mask_files, SAMPLE_SIZE_INTEGRITY)

    all_ok = all(checks.values())
    print("\n" + "=" * 70)
    print("  TRAIN DATASET - VALIDATION SUMMARY")
    print("=" * 70)
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if all_ok:
        print("\n[SUCCESS] All TRAIN checks passed.")
    else:
        print("\n[WARNING] Some TRAIN checks failed.")
    print("=" * 70)

    return _build_result(all_ok, dataset_root, image_dir, mask_dir,
                         image_files, mask_files, checks)


def download_and_validate_train_dataset(
    gdrive_file_id_or_url: str = TRAIN_GDRIVE_ID,
    output_dir: Path = DATA_DIR,
    force_redownload: bool = False,
    force_reextract: bool = False,
    clean_existing: bool = False,
    remove_zip_after_extraction: bool = True,
) -> Dict:
    """
    Download, extract, and validate the TRAIN dataset from Google Drive.

    Returns a dict with keys: success, dataset_root, image_dir, mask_dir,
    image_files, mask_files, checks.
    """
    print("\n" + "#" * 70)
    print("#  TRAIN DATASET PIPELINE")
    print("#" * 70)

    dataset_root = _download_extract_pipeline(
        gdrive_id_or_url=gdrive_file_id_or_url,
        output_dir=output_dir,
        dataset_name=TRAIN_DATASET_NAME,
        zip_name=TRAIN_ZIP_NAME,
        force_redownload=force_redownload,
        force_reextract=force_reextract,
        clean_existing=clean_existing,
        remove_zip_after_extraction=remove_zip_after_extraction,
    )

    if dataset_root is None:
        return _failure_result(Path(output_dir) / TRAIN_DATASET_NAME)

    return validate_train_dataset(dataset_root)


# ===========================================================================
# SECTION 3: TEST DATASET VALIDATION
# ===========================================================================

def validate_test_dataset(dataset_root: Path) -> Dict:
    """
    Run a full validation pipeline on the TEST dataset.

    The TEST dataset contains multiple sub-datasets, each with its own
    'images' and 'masks' folders. For each sub-dataset we verify:
      - folder structure (images/ and masks/)
      - file counts against the expected values
      - filename pairing between images and masks
      - integrity of a small sample of files

    Returns a dict with per-subset results.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_root}")
        return {
            "success": False,
            "dataset_root": dataset_root,
            "subsets": {},
            "checks": {},
        }

    print("\n" + "=" * 70)
    print(f"  VALIDATING TEST DATASET: {dataset_root}")
    print("=" * 70)

    subsets_result: Dict[str, Dict] = {}
    overall_ok = True

    # List of sub-datasets to validate.
    subset_names = list(TEST_EXPECTED_COUNTS.keys())

    for subset_name in subset_names:
        subset_dir = dataset_root / subset_name
        expected_count = TEST_EXPECTED_COUNTS[subset_name]

        print("\n" + "-" * 70)
        print(f"  SUBSET: {subset_name}  (expected {expected_count} images)")
        print("-" * 70)

        subset_info: Dict = {
            "expected_count": expected_count,
            "actual_count": 0,
            "checks": {},
        }

        if not subset_dir.exists() or not subset_dir.is_dir():
            print(f"[ERROR] Subset folder not found: {subset_dir}")
            subset_info["checks"]["exists"] = False
            overall_ok = False
            subsets_result[subset_name] = subset_info
            continue
        subset_info["checks"]["exists"] = True

        # Check images/ and masks/ sub-folders.
        image_dir, mask_dir, structure_ok = _check_folder_pair(subset_dir)
        subset_info["checks"]["structure"] = structure_ok
        if not structure_ok:
            overall_ok = False
            subsets_result[subset_name] = subset_info
            continue

        image_files = _list_image_files(image_dir)
        mask_files = _list_image_files(mask_dir)
        subset_info["actual_count"] = len(image_files)

        # Count check.
        count_ok = (len(image_files) == expected_count
                    and len(mask_files) == expected_count)
        if count_ok:
            print(f"[OK] Counts match expected ({expected_count}).")
        else:
            print(f"[WARNING] Counts: images={len(image_files)}, "
                  f"masks={len(mask_files)}, expected={expected_count}.")
        subset_info["checks"]["counts"] = count_ok
        if not count_ok:
            overall_ok = False

        # Filename pairing.
        pair_ok = _check_filename_pairing(image_files, mask_files)
        subset_info["checks"]["filename_pairing"] = pair_ok
        if not pair_ok:
            overall_ok = False

        # Integrity (sample from this subset only).
        integrity_ok = _check_integrity(
            image_files, mask_files,
            sample_size=min(SAMPLE_SIZE_INTEGRITY, len(image_files)) or 1,
            seed=hash(subset_name) % (2**32),
        )
        subset_info["checks"]["integrity"] = integrity_ok
        if not integrity_ok:
            overall_ok = False

        subset_info["image_files"] = image_files
        subset_info["mask_files"] = mask_files
        subset_info["image_dir"] = image_dir
        subset_info["mask_dir"] = mask_dir

        subsets_result[subset_name] = subset_info

    # Overall summary.
    print("\n" + "=" * 70)
    print("  TEST DATASET - VALIDATION SUMMARY")
    print("=" * 70)
    for name, info in subsets_result.items():
        status = "PASS" if all(info["checks"].values()) else "FAIL"
        print(f"  [{status}] {name}: {info['actual_count']} images "
              f"(expected {info['expected_count']})")
    print("=" * 70)
    if overall_ok:
        print("[SUCCESS] All TEST subsets passed validation.")
    else:
        print("[WARNING] Some TEST subsets failed validation.")
    print("=" * 70)

    return {
        "success": overall_ok,
        "dataset_root": dataset_root,
        "subsets": subsets_result,
    }


def download_and_validate_test_dataset(
    gdrive_file_id_or_url: str = TEST_GDRIVE_ID,
    output_dir: Path = DATA_DIR,
    force_redownload: bool = False,
    force_reextract: bool = False,
    clean_existing: bool = False,
    remove_zip_after_extraction: bool = True,
) -> Dict:
    """
    Download, extract, and validate the TEST dataset from Google Drive.

    Returns a dict with keys: success, dataset_root, subsets.
    """
    print("\n" + "#" * 70)
    print("#  TEST DATASET PIPELINE")
    print("#" * 70)

    dataset_root = _download_extract_pipeline(
        gdrive_id_or_url=gdrive_file_id_or_url,
        output_dir=output_dir,
        dataset_name=TEST_DATASET_NAME,
        zip_name=TEST_ZIP_NAME,
        force_redownload=force_redownload,
        force_reextract=force_reextract,
        clean_existing=clean_existing,
        remove_zip_after_extraction=remove_zip_after_extraction,
    )

    if dataset_root is None:
        return {
            "success": False,
            "dataset_root": Path(output_dir) / TEST_DATASET_NAME,
            "subsets": {},
        }

    return validate_test_dataset(dataset_root)


# ===========================================================================
# SECTION 4: RESULT HELPERS
# ===========================================================================

def _build_result(
    success: bool,
    dataset_root: Path,
    image_dir: Optional[Path],
    mask_dir: Optional[Path],
    image_files: List[Path],
    mask_files: List[Path],
    checks: Dict[str, bool],
) -> Dict:
    return {
        "success": success,
        "dataset_root": dataset_root,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "image_files": image_files,
        "mask_files": mask_files,
        "checks": checks,
    }


def _failure_result(dataset_root: Path) -> Dict:
    return {
        "success": False,
        "dataset_root": dataset_root,
        "image_dir": None,
        "mask_dir": None,
        "image_files": [],
        "mask_files": [],
        "checks": {},
    }


# ===========================================================================
# SECTION 5: MAIN ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Log the resolved paths for transparency.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  POLYP SEGMENTATION - DATASET DOWNLOADER & VALIDATOR")
    print("=" * 70)
    print(f"[INFO] Script location : {Path(__file__).resolve()}")
    print(f"[INFO] PROJECT_ROOT    : {PROJECT_ROOT}")
    print(f"[INFO] DATA_DIR        : {DATA_DIR}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # TRAIN dataset
    # ------------------------------------------------------------------
    train_result = download_and_validate_train_dataset(
        gdrive_file_id_or_url=TRAIN_GDRIVE_ID,
        output_dir=DATA_DIR,
        force_redownload=False,
        force_reextract=False,
        clean_existing=False,
        remove_zip_after_extraction=True,
    )

    if train_result["success"]:
        print(f"\n[INFO] TRAIN dataset ready at: {train_result['dataset_root']}")
        print(f"[INFO]   #Images : {len(train_result['image_files'])}")
        print(f"[INFO]   #Masks  : {len(train_result['mask_files'])}")
    else:
        print("\n[WARNING] TRAIN dataset validation failed.")

    # ------------------------------------------------------------------
    # TEST dataset
    # ------------------------------------------------------------------
    test_result = download_and_validate_test_dataset(
        gdrive_file_id_or_url=TEST_GDRIVE_ID,
        output_dir=DATA_DIR,
        force_redownload=False,
        force_reextract=False,
        clean_existing=False,
        remove_zip_after_extraction=True,
    )

    if test_result["success"]:
        print(f"\n[INFO] TEST dataset ready at: {test_result['dataset_root']}")
        for subset_name, info in test_result["subsets"].items():
            print(f"[INFO]   {subset_name}: {info['actual_count']} images")
    else:
        print("\n[WARNING] TEST dataset validation failed.")

    # ------------------------------------------------------------------
    # Final combined status
    # ------------------------------------------------------------------
    if train_result["success"] and test_result["success"]:
        print("\n[SUCCESS] Both TRAIN and TEST datasets are ready for training.")
    else:
        print("\n[WARNING] One or more datasets failed validation. "
              "Please review the log above.")
        raise SystemExit(1)