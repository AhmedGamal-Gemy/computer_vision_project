"""Image segmentation functions using Otsu thresholding and K-means clustering."""

from typing import Tuple

import cv2
import numpy as np

from .config import get_config, PipelineConfig


def otsu_segmentation(image: np.ndarray, cfg: PipelineConfig | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Otsu thresholding to segment the input image.

    Otsu thresholding automatically selects the optimal threshold by minimizing
    intra-class variance based on the image histogram. This is ideal for medical
    images where tissue intensity distributions are bimodal.

    Args:
        image: Grayscale input image as a numpy array.
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        A tuple containing:
            - binary_mask: The binary mask after morphological opening.
            - segmented_image: The original image with the mask applied.
    """
    if cfg is None:
        cfg = get_config()
    seg = cfg.segmentation

    # Apply Otsu's method - automatically finds optimal threshold from histogram
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (thresh > 0).astype(np.uint8) * 255
    
    # Morphological opening removes small noise while preserving larger structures
    kernel = np.ones(seg.morph_kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=seg.morph_iterations)
    
    # Apply mask to original image to get segmented result
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return (mask, segmented)


def kmeans_segmentation(image: np.ndarray, k: int | None = None, cfg: PipelineConfig | None = None) -> np.ndarray:
    """Apply K-means clustering to segment the input image.

    K-means with k=3 clusters medical images into background, tissue, and
    abnormal regions based on intensity similarity. This unsupervised approach
    works well when tissue types have distinct intensity distributions.

    Args:
        image: Grayscale input image as a numpy array.
        k: Number of clusters for K-means. If None, uses config default.
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        A label map visualized as an image with values scaled to 0-255.
    """
    if cfg is None:
        cfg = get_config()
    seg = cfg.segmentation

    if k is None:
        k = seg.kmeans_k

    # Reshape image to pixel list for clustering
    pixels = image.reshape(-1, 1).astype(np.float32)
    
    # Termination criteria: stop after max iterations OR when centers move less than epsilon
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, seg.kmeans_iterations, seg.kmeans_epsilon)
    best_labels = np.zeros(pixels.shape[0], dtype=np.int32)
    
    # Run K-means with random center initialization
    _, labels, centers = cv2.kmeans(
        pixels, k, best_labels, criteria, seg.kmeans_attempts, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Replace each pixel with its cluster center value
    segmented = centers[labels.flatten()].reshape(image.shape).astype(np.uint8)
    
    # Normalize to 0-255 range for visualization
    segmented = cv2.normalize(segmented, segmented, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return segmented
