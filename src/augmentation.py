
"""
Image augmentation pipeline for medical chest X-ray images.

Provides configurable augmentation functions that preserve anatomical
orientation and medical validity. All transformations are constrained
to ranges safe for chest X-ray interpretation.
"""

import cv2
import numpy as np
from pathlib import Path

from .config import get_config, PipelineConfig, AugmentationConfig

# Supported image file extensions (lowercase for matching)
# WHY: Tied to what OpenCV can reliably read, not a pipeline parameter.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Default augmentation config - all transforms enabled
# WHY: This controls which transforms are toggled on/off, separate from
# the parameter ranges defined in AugmentationConfig. It's a UI/pipeline
# concern, not a tunable hyperparameter.
DEFAULT_CONFIG = {
    "rotation": True,
    "brightness": True,
    "contrast": True,
    "noise": True,
    "zoom": True,
    "shear": True,
    "translation": True,
    "horizontal_flip": True,
}


def _apply_rotation(image: np.ndarray, aug_cfg: AugmentationConfig) -> np.ndarray:
    """Rotate image by a random angle within the configured max rotation angle.

    Args:
        image: Grayscale input image as a numpy array.
        aug_cfg: Augmentation configuration with parameter ranges.

    Returns:
        Rotated image with black padding for areas outside original bounds.
    """
    angle = np.random.uniform(-aug_cfg.max_rotation_angle, aug_cfg.max_rotation_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated


def _apply_brightness(image: np.ndarray, aug_cfg: AugmentationConfig) -> np.ndarray:
    """Adjust image brightness by a random multiplier within the configured range.

    Args:
        image: Grayscale input image as a numpy array.
        aug_cfg: Augmentation configuration with parameter ranges.

    Returns:
        Brightness-adjusted image clipped to valid pixel range.
    """
    factor = np.random.uniform(*aug_cfg.brightness_range)
    adjusted = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return adjusted


def _apply_contrast(image: np.ndarray, aug_cfg: AugmentationConfig) -> np.ndarray:
    """Adjust image contrast by a random multiplier within the configured range.

    Args:
        image: Grayscale input image as a numpy array.
        aug_cfg: Augmentation configuration with parameter ranges.

    Returns:
        Contrast-adjusted image clipped to valid pixel range.
    """
    factor = np.random.uniform(*aug_cfg.contrast_range)
    mean = image.mean()
    adjusted = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    return adjusted


def _apply_noise(image: np.ndarray, aug_cfg: AugmentationConfig) -> np.ndarray:
    """Add Gaussian noise with standard deviation sampled from the configured range.

    Args:
        image: Grayscale input image as a numpy array.
        aug_cfg: Augmentation configuration with parameter ranges.

    Returns:
        Noisy image with pixel values clipped to [0, 255].
    """
    std = np.random.uniform(*aug_cfg.noise_std_range) * 255  # Scale to pixel range
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def _apply_zoom(image: np.ndarray, aug_cfg: AugmentationConfig) -> np.ndarray:
    """Zoom into or out of the image by a random factor within the configured range.

    Args:
        image: Grayscale input image as a numpy array.
        aug_cfg: Augmentation configuration with parameter ranges.

    Returns:
        Zoomed image resized back to original dimensions.
    """
    factor = np.random.uniform(*aug_cfg.zoom_range)
    h, w = image.shape[:2]
    # Compute crop dimensions - zoom in means smaller crop, zoom out means larger
    new_h = int(h / factor)
    new_w = int(w / factor)
    # Center crop
    y_start = max(0, (h - new_h) // 2)
    x_start = max(0, (w - new_w) // 2)
    # Handle cases where crop exceeds image bounds
    y_end = min(h, y_start + new_h)
    x_end = min(w, x_start + new_w)
    cropped = image[y_start:y_end, x_start:x_end]
    # Resize back to original dimensions
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed


def _apply_shear(image: np.ndarray, aug_cfg: AugmentationConfig) -> np.ndarray:
    """Apply affine shear transformation within the configured shear range.

    Args:
        image: Grayscale input image as a numpy array.
        aug_cfg: Augmentation configuration with parameter ranges.

    Returns:
        Sheared image with reflection padding for out-of-bounds areas.
    """
    shear_angle = np.random.uniform(*aug_cfg.shear_range)
    h, w = image.shape[:2]
    # Convert shear angle to affine shift
    shear_factor = np.tan(np.radians(shear_angle))
    # Shear matrix: shift pixels horizontally proportional to their row
    matrix = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    sheared = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return sheared


def _apply_translation(image: np.ndarray, aug_cfg: AugmentationConfig) -> np.ndarray:
    """Translate image by a random offset within the configured range of dimensions.

    Args:
        image: Grayscale input image as a numpy array.
        aug_cfg: Augmentation configuration with parameter ranges.

    Returns:
        Translated image with reflection padding for vacated areas.
    """
    h, w = image.shape[:2]
    tx = np.random.uniform(-aug_cfg.translation_range[0], aug_cfg.translation_range[0]) * w
    ty = np.random.uniform(-aug_cfg.translation_range[1], aug_cfg.translation_range[1]) * h
    matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    translated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return translated


def _apply_horizontal_flip(image: np.ndarray, aug_cfg: AugmentationConfig) -> np.ndarray:
    """Flip image horizontally with the configured probability.

    WHY: Chest X-rays exhibit bilateral symmetry, so horizontal flip
    preserves anatomical plausibility and doubles effective training data.

    Args:
        image: Grayscale input image as a numpy array.
        aug_cfg: Augmentation configuration with parameter ranges.

    Returns:
        Horizontally flipped image or original, based on random probability.
    """
    if np.random.random() < aug_cfg.horizontal_flip_prob:
        return cv2.flip(image, 1)
    return image


def augment_image(image: np.ndarray, config: dict | None = None, cfg: PipelineConfig | None = None) -> np.ndarray:
    """Apply a random combination of augmentations based on config.

    Each enabled augmentation is applied sequentially with random parameters
    drawn from their respective configured ranges. Medical augmentations preserve
    anatomical orientation - no vertical flips, limited rotation, small shear.

    Args:
        image: Grayscale input image as a numpy array (expected np.uint8).
        config: Dictionary mapping augmentation names to bools. Keys:
            rotation, brightness, contrast, noise, zoom, shear,
            translation, horizontal_flip. None uses DEFAULT_CONFIG.
        cfg: Optional pipeline configuration for parameter ranges.
            Defaults to global config.

    Returns:
        Augmented image as a numpy array with the same dtype as input.
    """
    if cfg is None:
        cfg = get_config()
    if config is None:
        config = DEFAULT_CONFIG

    aug_cfg = cfg.augmentation
    result = image.copy()

    # Apply each augmentation if enabled in config - order matters for
    # cumulative effect: geometric transforms first, then intensity changes
    if config.get("rotation", False):
        result = _apply_rotation(result, aug_cfg)

    if config.get("zoom", False):
        result = _apply_zoom(result, aug_cfg)

    if config.get("shear", False):
        result = _apply_shear(result, aug_cfg)

    if config.get("translation", False):
        result = _apply_translation(result, aug_cfg)

    if config.get("horizontal_flip", False):
        result = _apply_horizontal_flip(result, aug_cfg)

    if config.get("brightness", False):
        result = _apply_brightness(result, aug_cfg)

    if config.get("contrast", False):
        result = _apply_contrast(result, aug_cfg)

    if config.get("noise", False):
        result = _apply_noise(result, aug_cfg)

    # Preserve original dtype - medical images must stay uint8
    return result.astype(image.dtype)


def augment_batch(
    images: list[np.ndarray],
    labels: list[int],
    config: dict | None = None,
    multiplier: int = 2,
    cfg: PipelineConfig | None = None,
) -> tuple[list[np.ndarray], list[int]]:
    """Generate multiple augmented copies of each image in a batch.

    WHY: Medical datasets are often small and imbalanced; augmentation
    artificially expands training data while preserving label integrity.

    Args:
        images: List of grayscale images as numpy arrays.
        labels: List of integer labels corresponding to each image.
        config: Augmentation config dict (see augment_image). None uses DEFAULT_CONFIG.
        multiplier: Number of augmented copies to generate per image.
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        A tuple of two lists:
            - augmented_images: Original images followed by augmented copies.
            - augmented_labels: Labels repeated to match augmented images.
    """
    augmented_images: list[np.ndarray] = list(images)
    augmented_labels: list[int] = list(labels)

    for _ in range(multiplier):
        for img, label in zip(images, labels):
            aug_img = augment_image(img, config, cfg)
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    return (augmented_images, augmented_labels)


def generate_augmented_dataset(
    input_dir: Path,
    output_dir: Path,
    multiplier: int = 2,
) -> None:
    """Load images from input_dir, augment, and save to output_dir.

    Walks NORMAL/ and PNEUMONIA/ subdirectories in input_dir, generates
    augmented copies of each image, and saves them to output_dir with
    the same directory structure. Original images are also copied.

    Args:
        input_dir: Root directory containing class subdirectories (NORMAL/, PNEUMONIA/).
        output_dir: Root directory where augmented dataset will be written.
        multiplier: Number of augmented copies per original image.

    Returns:
        None. Augmented images are written to output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    class_dirs = ["NORMAL", "PNEUMONIA"]
    total_processed = 0

    for class_name in class_dirs:
        class_input = input_path / class_name
        class_output = output_path / class_name

        if not class_input.exists():
            continue

        class_output.mkdir(parents=True, exist_ok=True)

        # Collect image files with case-insensitive extension matching
        image_files: list[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(class_input.glob(f"*{ext}"))
            image_files.extend(class_input.glob(f"*{ext.upper()}"))

        for img_path in image_files:
            # Load as grayscale - consistent with preprocessing pipeline
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            # Save original image to output directory
            cv2.imwrite(str(class_output / img_path.name), image)

            # Generate and save augmented copies
            stem = img_path.stem
            suffix = img_path.suffix
            for copy_idx in range(1, multiplier + 1):
                aug_image = augment_image(image)
                aug_filename = f"{stem}_aug{copy_idx}{suffix}"
                cv2.imwrite(str(class_output / aug_filename), aug_image)

            total_processed += 1
            if total_processed % 50 == 0:
                print(f"Processed {total_processed} images")

    print(f"Augmentation complete: {total_processed} images processed")
