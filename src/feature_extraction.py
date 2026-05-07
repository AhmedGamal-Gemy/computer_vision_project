"""Feature extraction module for medical image analysis.

Provides functions to extract texture and shape features from medical images
using Gray-Level Co-occurrence Matrix (GLCM) and region properties.
"""

import numpy as np
import skimage.feature
import skimage.measure
from typing import Tuple

from .preprocessing import preprocess_image
from .segmentation import otsu_segmentation
from .config import get_config, PipelineConfig

# Processing parameters
# WHY: Display-only parameter, not a pipeline tuning knob.
PROGRESS_INTERVAL = 100  # Print progress every N images during batch processing

# Feature names for reference and documentation
# WHY: Fixed list matching the 8-dimensional feature vector from the pipeline.
# Not configurable because changing features requires code changes anyway.
FEATURE_NAMES = [
    "Contrast", "Correlation", "Energy", "Homogeneity",
    "Area", "Perimeter", "Eccentricity", "Solidity"
]


def extract_features(image: np.ndarray, mask: np.ndarray, cfg: PipelineConfig | None = None) -> np.ndarray:
    """Extract texture and shape features from an image and its segmentation mask.

    Computes 8 features total:
    - 4 texture features using Gray-Level Co-occurrence Matrix (GLCM):
      contrast, correlation, energy, homogeneity
    - 4 shape features using connected component analysis:
      area (normalized), perimeter, eccentricity, solidity

    Args:
        image: Grayscale input image as a numpy ndarray (uint8, 0-255).
        mask: Binary segmentation mask as a numpy ndarray (uint8, 0 or 255).
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        A numpy array of 8 feature values in the order:
        [contrast, correlation, energy, homogeneity, area, perimeter, eccentricity, solidity].
    """
    if cfg is None:
        cfg = get_config()
    fe = cfg.feature_extraction

    # --- TEXTURE FEATURES (4) using GLCM ---
    # GLCM captures spatial relationships between pixel pairs to quantify texture patterns
    glcm = skimage.feature.graycomatrix(
        image,
        distances=fe.glcm_distances,
        angles=fe.glcm_angles,
        levels=fe.glcm_levels,
        symmetric=True,
        normed=True
    )

    # Contrast: Measures texture variation - pneumonia shows different texture patterns than healthy tissue
    contrast = skimage.feature.graycoprops(glcm, 'contrast')[0, 0]
    # Correlation: Measures linear dependency - healthy vs diseased tissue have different correlation structures
    correlation = skimage.feature.graycoprops(glcm, 'correlation')[0, 0]
    # Energy: Uniformity measure - pneumonia regions have different energy (sum of squared elements) than healthy areas
    energy = skimage.feature.graycoprops(glcm, 'energy')[0, 0]
    # Homogeneity: Local uniformity - diseased areas show different homogeneity (inverse difference moment) patterns
    homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')[0, 0]

    # --- SHAPE FEATURES (4) using regionprops ---
    # Shape features quantify the geometry of segmented regions - pneumonia affects lung structure
    mask_binary = (mask > 0).astype(np.uint8)
    labels = skimage.measure.label(mask_binary)
    regions = skimage.measure.regionprops(labels)

    if len(regions) > 0:
        # Find the largest region by area
        largest_region = max(regions, key=lambda r: r.area)

        image_size = image.shape[0] * image.shape[1]
        # Area: Size of affected region - pneumonia typically affects larger areas than healthy tissue
        area = largest_region.area / image_size  # normalized by image size
        # Perimeter: Boundary complexity - diseased regions have irregular, complex boundaries
        perimeter = largest_region.perimeter
        # Eccentricity: Shape elongation - pneumonia patterns tend to be more irregular (less elliptical)
        eccentricity = largest_region.eccentricity
        # Solidity: Convexity measure - diseased regions are less solid/convex due to patchy infiltration
        solidity = largest_region.solidity
    else:
        # No region found, return zeros for shape features
        area = 0.0
        perimeter = 0.0
        eccentricity = 0.0
        solidity = 0.0

    return np.array([
        contrast, correlation, energy, homogeneity,
        area, perimeter, eccentricity, solidity
    ], dtype=np.float64)


def build_feature_matrix(image_paths: list, labels: list, cfg: PipelineConfig | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Build a feature matrix and label vector from a list of image paths.

    Iterates over all image paths, and for each image:
    1. Preprocesses the image using preprocess_image()
    2. Segments the image using otsu_segmentation()
    3. Extracts features using extract_features()

    Prints progress every 100 images.

    Args:
        image_paths: List of string paths to image files.
        labels: List of integer labels corresponding to each image (0 or 1).
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        A tuple (X, y) where:
        - X is a numpy array of shape (N, 8) containing the feature matrix
        - y is a numpy array of shape (N,) containing the labels
    """
    if cfg is None:
        cfg = get_config()

    features_list = []

    for i, image_path in enumerate(image_paths):
        # Preprocess the image
        processed_image = preprocess_image(image_path, cfg)

        # Segment the image using Otsu
        mask, _ = otsu_segmentation(processed_image, cfg)

        # Extract features
        features = extract_features(processed_image, mask, cfg)
        features_list.append(features)

        # Print progress every PROGRESS_INTERVAL images
        if (i + 1) % PROGRESS_INTERVAL == 0:
            print(f"Processed {i + 1} images")

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels, dtype=np.int64)

    return (X, y)
