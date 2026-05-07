"""
Preprocessing module for medical image dataset.
"""

import cv2
import numpy as np
from pathlib import Path

from .config import get_config, PipelineConfig

# Normalization range - scale pixel values to full 8-bit range
# WHY: These are fixed OpenCV normalization bounds for 8-bit images,
# not configurable pipeline parameters.
NORMALIZE_MIN = 0
NORMALIZE_MAX = 255

# Supported image file extensions (lowercase for matching)
# WHY: Tied to what OpenCV can reliably read, not a pipeline parameter.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def preprocess_image(image_path: str, cfg: PipelineConfig | None = None) -> np.ndarray:
    """Load and preprocess a medical image for model input.

    Performs the following steps in order:
    1. Loads the image as grayscale using OpenCV.
    2. Resizes the image to the configured target size.
    3. Applies Gaussian blur with configured kernel and sigma.
    4. Applies a median filter with configured kernel size.
    5. Normalizes pixel values to the range [0, 255] as 8-bit unsigned integers.

    Args:
        image_path: Path to the input image file (string).
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        Preprocessed image as a numpy ndarray.

    Raises:
        FileNotFoundError: If the image cannot be loaded from the given path.
    """
    if cfg is None:
        cfg = get_config()
    pp = cfg.preprocessing

    # Load image as grayscale - medical images are typically single-channel
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image at path: {image_path}")

    # Resize to standard dimensions for consistent model input
    img = cv2.resize(img, pp.target_size)

    # Apply Gaussian blur - smooths image while preserving edge structure better than simple averaging
    img = cv2.GaussianBlur(img, pp.gaussian_kernel, sigmaX=pp.gaussian_sigma)

    # Apply median filter - effective at removing salt-and-pepper noise common in medical imaging
    img = cv2.medianBlur(img, pp.median_kernel)

    # Normalize pixel values to full 8-bit range for optimal contrast
    img = cv2.normalize(
        img,
        img,
        NORMALIZE_MIN,
        NORMALIZE_MAX,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )

    return img


def load_dataset_paths(data_dir: str) -> tuple[list[str], list[int]]:
    """Load image paths and labels from dataset splits.

    Walks through the train, val, and test splits of the dataset, each containing
    NORMAL and PNEUMONIA class subdirectories. Collects all image paths and assigns
    labels (0 for NORMAL, 1 for PNEUMONIA), printing progress for each split/class.

    Args:
        data_dir: Root directory of the dataset (string) containing split folders.

    Returns:
        A tuple of two lists:
        - image_paths: List of string paths to image files.
        - labels: List of integer labels corresponding to each image (0 or 1).
    """
    data_path = Path(data_dir)
    image_paths: list[str] = []
    labels: list[int] = []
    splits = ["train", "val", "test"]
    class_mapping = {"NORMAL": 0, "PNEUMONIA": 1}

    for split in splits:
        split_dir = data_path / split
        if not split_dir.exists():
            continue  # Skip splits that don't exist

        for class_name, label in class_mapping.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue  # Skip classes that don't exist in this split

            # Collect all image files with case-insensitive extension matching
            # Using glob patterns to handle mixed-case extensions on Windows
            image_files: list[Path] = []
            for ext in IMAGE_EXTENSIONS:
                # Match both lowercase and uppercase extensions
                image_files.extend(class_dir.glob(f"*{ext}"))
                image_files.extend(class_dir.glob(f"*{ext.upper()}"))

            # Add to paths and labels
            for img_file in image_files:
                image_paths.append(str(img_file))
                labels.append(label)

            # Print progress as required
            print(f"Found {len(image_files)} images in {split}/{class_name}")

    return (image_paths, labels)
