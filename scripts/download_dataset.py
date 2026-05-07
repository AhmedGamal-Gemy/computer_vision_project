#!/usr/bin/env python3
"""
Dataset Download Script for Chest X-Ray Pneumonia Dataset.

This script downloads the chest X-ray pneumonia dataset from Kaggle and prepares
it for use in the computer vision project. It intelligently checks for existing
data before attempting to download.

Dataset: paultimothymooney/chest-xray-pneumonia
Source: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    KaggleApi = None  # type: ignore


# Configuration
KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "chest_xray"

# Expected directory structure
EXPECTED_SUBDIRS = [
    "train/NORMAL",
    "train/PNEUMONIA",
    "val/NORMAL",
    "val/PNEUMONIA",
    "test/NORMAL",
    "test/PNEUMONIA",
]


def check_existing_data() -> bool:
    """
    Check if the dataset already exists with the correct structure.

    Returns:
        True if data exists with at least train/ and test/ splits containing images,
        False otherwise.
    """
    print(f"Checking for existing dataset at {DATA_DIR}...")

    if not DATA_DIR.exists():
        print("  - Data directory does not exist")
        return False

    # Check for minimum required structure: train and test with both classes
    required_splits = ["train", "test"]
    required_classes = ["NORMAL", "PNEUMONIA"]

    for split in required_splits:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"  - Missing directory: {split}/")
            return False

        for class_name in required_classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"  - Missing directory: {split}/{class_name}/")
                return False

            # Check if directory contains .jpeg files
            jpeg_files = list(class_dir.glob("*.jpeg"))
            if len(jpeg_files) == 0:
                print(f"  - No .jpeg files found in {split}/{class_name}/")
                return False

    print("  + Dataset structure verified!")
    return True


def print_dataset_summary() -> None:
    """Print a summary of the dataset structure and image counts."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    splits = ["train", "val", "test"]
    classes = ["NORMAL", "PNEUMONIA"]

    total_images = 0

    for split in splits:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            continue

        print(f"\n{split.upper()}/:")
        for class_name in classes:
            class_dir = split_dir / class_name
            if class_dir.exists():
                jpeg_files = list(class_dir.glob("*.jpeg"))
                count = len(jpeg_files)
                total_images += count
                print(f"  {class_name}/: {count} images")

    print("\n" + "-" * 60)
    print(f"TOTAL IMAGES: {total_images}")
    print("=" * 60)


def print_kaggle_setup_instructions() -> None:
    """Print instructions for setting up Kaggle API credentials."""
    print("\n" + "!" * 60)
    print("KAGGLE API NOT CONFIGURED")
    print("!" * 60)
    print("""
To download the dataset, you need to configure Kaggle API credentials:

1. CREATE KAGGLE ACCOUNT
   - Go to https://www.kaggle.com/account/login

2. DOWNLOAD API TOKEN
   - Click your profile picture (top right)
   - Select "My Profile" -> "Account"
   - Click "Create New API Token"
   - This downloads kaggle.json to your Downloads folder

3. PLACE KAGGLE.JSON IN CORRECT LOCATION (Windows)
   - Create directory: %USERPROFILE%\\\\.kaggle\\\\
   - Move kaggle.json to: C:\\\\Users\\\\<YourUsername>\\\\.kaggle\\\\kaggle.json

4. SET FILE PERMISSIONS (IMPORTANT)
   - Right-click .kaggle folder -> Properties -> Security
   - Ensure only your user account has access

5. RUN SCRIPT AGAIN
   - uv run python scripts/download_dataset.py

For more details: https://www.kaggle.com/docs/api
""")
    print("!" * 60)


def download_dataset() -> bool:
    """
    Download the dataset from Kaggle.

    Returns:
        True if download succeeded, False otherwise.
    """
    if KaggleApi is None:
        print("ERROR: kaggle package not installed")
        print("Install with: uv add kaggle")
        return False

    try:
        api = KaggleApi()
        api.authenticate()
        print("  + Kaggle API authenticated successfully")
    except Exception as e:
        print(f"  - Kaggle API authentication failed: {e}")
        return False

    # Create temporary download location
    temp_dir = DATA_DIR.parent / "temp_download"
    temp_dir.mkdir(exist_ok=True)

    try:
        print(f"  - Downloading dataset: {KAGGLE_DATASET}")
        print("  - This may take several minutes depending on your connection...")

        # Download dataset
        api.dataset_download_files(KAGGLE_DATASET, path=str(temp_dir), unzip=False)

        # Find the downloaded zip file
        zip_files = list(temp_dir.glob("*.zip"))
        if not zip_files:
            print("  - ERROR: No zip file found after download")
            return False

        zip_path = zip_files[0]
        print(f"  + Downloaded: {zip_path.name}")

        # Extract to data directory
        print("  - Extracting dataset...")
        DATA_DIR.parent.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

        # Clean up temp directory
        shutil.rmtree(temp_dir)
        print("  + Extraction complete, cleaned up temporary files")

        return True

    except Exception as e:
        print(f"  - Download failed: {e}")
        # Clean up on failure
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False


def verify_extracted_structure() -> bool:
    """
    Verify that the extracted dataset has the expected structure.

    Returns:
        True if structure is valid, False otherwise.
    """
    print("\nVerifying extracted dataset structure...")

    all_valid = True

    for subdir in EXPECTED_SUBDIRS:
        full_path = DATA_DIR / subdir
        if not full_path.exists():
            print(f"  [!] Missing: {subdir}/")
            all_valid = False
        else:
            jpeg_count = len(list(full_path.glob("*.jpeg")))
            if jpeg_count == 0:
                print(f"  [!] Empty: {subdir}/ (no .jpeg files)")
                all_valid = False
            else:
                print(f"  [OK] {subdir}/: {jpeg_count} images")

    return all_valid


def main() -> int:
    """
    Main entry point for the dataset download script.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    print("=" * 60)
    print("CHEST X-RAY PNEUMONIA DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Target directory: {DATA_DIR}")
    print(f"Kaggle dataset: {KAGGLE_DATASET}")
    print()

    # Step 1: Check if data already exists
    if check_existing_data():
        print("\n[OK] Dataset already found at data/chest_xray/")
        print_dataset_summary()
        print("\nSkipping download. Ready to use!")
        return 0

    print("\nDataset not found. Proceeding with download...\n")

    # Step 2: Download dataset
    if not download_dataset():
        print_kaggle_setup_instructions()
        return 1

    # Step 3: Verify extracted structure
    if not verify_extracted_structure():
        print("\n[WARNING] Dataset structure may be incomplete")
        print("Please check the data/chest_xray/ directory manually")
        return 1

    # Step 4: Print summary
    print_dataset_summary()

    print("\n[SUCCESS] Dataset download and setup complete!")
    print("Data is ready for use at: data/chest_xray/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
