"""
Class balancing utilities for medical image datasets.

Provides oversampling, undersampling, and SMOTE-based strategies to
address class imbalance common in medical datasets where pathology
cases are typically rarer than normal cases.
"""

import numpy as np
from sklearn.utils import resample

from .config import get_config, PipelineConfig

# Supported balancing strategies - extensible for future methods
# WHY: Validation list for the strategy parameter, not a tunable parameter.
BALANCE_STRATEGIES = ("oversample", "undersample", "smote")


def oversample_minority(
    X: np.ndarray, y: np.ndarray, cfg: PipelineConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class by duplicating samples with slight noise.

    WHY: Prevents model bias toward majority class in medical datasets
    where pneumonia cases are often underrepresented. Adding Gaussian noise
    to duplicates avoids the model memorizing exact repeated samples.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        A tuple of two numpy arrays:
            - X_balanced: Feature matrix with minority class oversampled.
            - y_balanced: Corresponding label array.
    """
    if cfg is None:
        cfg = get_config()

    classes, counts = np.unique(y, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    majority_count = counts.max()

    X_balanced_parts: list[np.ndarray] = [X[y == classes[0]]]
    y_balanced_parts: list[np.ndarray] = [y[y == classes[0]]]

    # Start from second class since first is already added
    for cls in classes[1:]:
        cls_mask = y == cls
        X_cls = X[cls_mask]
        y_cls = y[cls_mask]
        cls_count = X_cls.shape[0]

        X_balanced_parts.append(X_cls)
        y_balanced_parts.append(y_cls)

        if cls == majority_class:
            continue

        # Oversample minority class to match majority count
        needed = majority_count - cls_count
        if needed <= 0:
            continue

        # Randomly select samples to duplicate
        duplicate_indices = np.random.choice(cls_count, size=needed, replace=True)
        X_duplicates = X_cls[duplicate_indices].copy()

        # Add slight Gaussian noise to avoid exact copies - prevents overfitting
        noise = np.random.normal(0, cfg.balancing.oversample_noise_std, X_duplicates.shape).astype(
            X_duplicates.dtype
        )
        X_duplicates = X_duplicates + noise

        X_balanced_parts.append(X_duplicates)
        y_balanced_parts.append(np.full(needed, cls, dtype=y.dtype))

    X_balanced = np.concatenate(X_balanced_parts, axis=0)
    y_balanced = np.concatenate(y_balanced_parts, axis=0)

    return (X_balanced, y_balanced)


def undersample_majority(
    X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Undersample the majority class by randomly removing samples.

    WHY: Reduces training time and prevents majority class dominance,
    though at the cost of discarding potentially useful majority samples.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
        random_state: Seed for reproducible sampling.

    Returns:
        A tuple of two numpy arrays:
            - X_balanced: Feature matrix with majority class undersampled.
            - y_balanced: Corresponding label array.
    """
    classes, counts = np.unique(y, return_counts=True)
    minority_count = counts.min()
    minority_class = classes[np.argmin(counts)]

    X_balanced_parts: list[np.ndarray] = []
    y_balanced_parts: list[np.ndarray] = []

    for cls in classes:
        cls_mask = y == cls
        X_cls: np.ndarray = X[cls_mask]
        y_cls: np.ndarray = y[cls_mask]

        if cls == minority_class:
            X_balanced_parts.append(X_cls)
            y_balanced_parts.append(y_cls)
        else:
            # Downsample majority class to match minority count
            X_downsampled = np.asarray(
                resample(
                    X_cls,
                    replace=False,
                    n_samples=minority_count,
                    random_state=random_state,
                )
            )
            y_downsampled = np.asarray(
                resample(
                    y_cls,
                    replace=False,
                    n_samples=minority_count,
                    random_state=random_state,
                )
            )
            X_balanced_parts.append(X_downsampled)
            y_balanced_parts.append(y_downsampled)

    X_balanced = np.concatenate(X_balanced_parts, axis=0)
    y_balanced = np.concatenate(y_balanced_parts, axis=0)

    return (X_balanced, y_balanced)


def _apply_smote(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE-like oversampling by interpolating between minority samples.

    WHY: SMOTE generates synthetic minority samples along line segments
    between nearest neighbors, creating more diverse training data than
    simple duplication with noise.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).

    Returns:
        A tuple of two numpy arrays:
            - X_balanced: Feature matrix with synthetic minority samples.
            - y_balanced: Corresponding label array.
    """
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()
    minority_class = classes[np.argmin(counts)]

    X_balanced_parts: list[np.ndarray] = [X]
    y_balanced_parts: list[np.ndarray] = [y]

    cls_mask = y == minority_class
    X_minority = X[cls_mask]
    minority_count = X_minority.shape[0]
    needed = majority_count - minority_count

    if needed <= 0:
        return (X.copy(), y.copy())

    # Generate synthetic samples by interpolating between random minority pairs
    for _ in range(needed):
        idx_a = np.random.randint(0, minority_count)
        idx_b = np.random.randint(0, minority_count)
        # Interpolation factor - random blend between two minority samples
        alpha = np.random.random()
        synthetic = X_minority[idx_a] + alpha * (X_minority[idx_b] - X_minority[idx_a])
        X_balanced_parts.append(synthetic.reshape(1, -1))
        y_balanced_parts.append(np.array([minority_class], dtype=y.dtype))

    X_balanced = np.concatenate(X_balanced_parts, axis=0)
    y_balanced = np.concatenate(y_balanced_parts, axis=0)

    return (X_balanced, y_balanced)


def balance_dataset(
    X: np.ndarray, y: np.ndarray, strategy: str | None = None, cfg: PipelineConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Balance dataset classes using the specified strategy.

    Dispatches to the appropriate balancing method and prints class
    distribution before and after balancing for verification.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
        strategy: Balancing method - one of BALANCE_STRATEGIES
            ("oversample", "undersample", "smote"). If None, uses config default.
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        A tuple of two numpy arrays:
            - X_balanced: Feature matrix after balancing.
            - y_balanced: Corresponding label array after balancing.

    Raises:
        ValueError: If strategy is not in BALANCE_STRATEGIES.
    """
    if cfg is None:
        cfg = get_config()
    if strategy is None:
        strategy = cfg.balancing.strategy

    if strategy not in BALANCE_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Must be one of {BALANCE_STRATEGIES}"
        )

    # Print class distribution before balancing
    classes_before, counts_before = np.unique(y, return_counts=True)
    dist_before = dict(zip(classes_before.tolist(), counts_before.tolist()))
    print(f"Class distribution before balancing: {dist_before}")

    # Dispatch to the selected balancing method
    if strategy == "oversample":
        X_balanced, y_balanced = oversample_minority(X, y, cfg)
    elif strategy == "undersample":
        X_balanced, y_balanced = undersample_majority(X, y)
    elif strategy == "smote":
        X_balanced, y_balanced = _apply_smote(X, y)

    # Print class distribution after balancing
    classes_after, counts_after = np.unique(y_balanced, return_counts=True)
    dist_after = dict(zip(classes_after.tolist(), counts_after.tolist()))
    print(f"Class distribution after balancing ({strategy}): {dist_after}")

    return (X_balanced, y_balanced)
