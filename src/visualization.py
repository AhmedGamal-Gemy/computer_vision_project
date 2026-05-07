"""
Pipeline-wide plotting utilities for medical image classification.

Provides standardized visualization functions for model evaluation metrics,
feature analysis, and comparison plots. All functions save figures to disk
and close them to prevent memory leaks.

Uses matplotlib for all plotting and sklearn.metrics for metric computations.
Consistent styling across all plots ensures professional, publication-ready
output for clinical and research use.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

from .config import get_config, PipelineConfig


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Path,
    title: str = "ROC Curve",
    cfg: PipelineConfig | None = None,
) -> None:
    """Plot and save a Receiver Operating Characteristic (ROC) curve.

    Computes the ROC curve and AUC score, then plots the curve with a
    diagonal reference line representing a random classifier. The AUC
    value is displayed in the legend.

    WHY: ROC curves show the trade-off between sensitivity and specificity
    at different thresholds, which is critical for medical diagnosis where
    the optimal threshold depends on clinical context (e.g., screening
    vs. confirmatory testing).

    Args:
        y_true: True binary labels of shape (n_samples,), 0 for NORMAL, 1 for PNEUMONIA.
        y_scores: Predicted probabilities for the positive class (PNEUMONIA),
            shape (n_samples,).
        save_path: Path where the figure will be saved.
        title: Plot title string.
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        None. Figure is saved to save_path and closed.
    """
    if cfg is None:
        cfg = get_config()
    viz = cfg.visualization

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=viz.figure_size_standard)
    ax.plot(fpr, tpr, color=viz.bar_color, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random classifier")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=viz.figure_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Path,
    title: str = "Precision-Recall Curve",
    cfg: PipelineConfig | None = None,
) -> None:
    """Plot and save a Precision-Recall curve.

    Computes the PR curve and average precision score, then plots the curve
    with the average precision displayed in the legend.

    WHY: PR curves are more informative than ROC when classes are imbalanced
    (common in medical datasets where pathology is rarer than normal). They
    show the trade-off between detecting true positives and avoiding false
    alarms, which directly maps to clinical sensitivity vs. over-referral.

    Args:
        y_true: True binary labels of shape (n_samples,), 0 for NORMAL, 1 for PNEUMONIA.
        y_scores: Predicted probabilities for the positive class (PNEUMONIA),
            shape (n_samples,).
        save_path: Path where the figure will be saved.
        title: Plot title string.
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        None. Figure is saved to save_path and closed.
    """
    if cfg is None:
        cfg = get_config()
    viz = cfg.visualization

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=viz.figure_size_standard)
    ax.plot(recall, precision, color=viz.bar_color, lw=2,
            label=f"PR curve (AP = {avg_precision:.3f})")
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=viz.figure_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_detailed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str = "Confusion Matrix",
    cfg: PipelineConfig | None = None,
) -> None:
    """Plot and save a detailed confusion matrix with counts and percentages.

    Creates a heatmap visualization of the confusion matrix with each cell
    annotated with both the raw count and the percentage of total samples.

    WHY: Confusion matrices reveal the types of errors a model makes.
    In medical diagnosis, false negatives (missing pneumonia) are far more
    dangerous than false positives (over-referral), so understanding the
    error distribution is essential for clinical deployment decisions.

    Args:
        y_true: True binary labels of shape (n_samples,).
        y_pred: Predicted binary labels of shape (n_samples,).
        class_names: List of class name strings for axis labels
            (e.g., ["NORMAL", "PNEUMONIA"]).
        save_path: Path where the figure will be saved.
        title: Plot title string.
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        None. Figure is saved to save_path and closed.
    """
    if cfg is None:
        cfg = get_config()
    viz = cfg.visualization

    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()

    fig, ax = plt.subplots(figsize=viz.figure_size_square)
    im = ax.imshow(cm, interpolation="nearest", cmap=viz.cmap_confusion)
    ax.figure.colorbar(im, ax=ax)

    # Annotate each cell with count and percentage
    n_classes = len(class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i, j]
            pct = count / total * 100 if total > 0 else 0.0
            # Use white text on dark cells, black on light cells
            text_color = "white" if im.norm(cm[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", color=text_color, fontsize=12)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=viz.figure_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(
    metrics: dict[str, dict[str, float]],
    save_path: Path,
    cfg: PipelineConfig | None = None,
) -> None:
    """Plot and save a grouped bar chart comparing models across metrics.

    Creates a side-by-side bar chart comparing SVM and Random Forest
    performance across all provided metrics (accuracy, precision, recall, etc.).

    WHY: Direct visual comparison of models helps select the best classifier
    for clinical use. Different metrics matter differently in medical contexts:
    recall is critical for screening (catch all cases), while precision
    matters for confirmatory testing (avoid false alarms).

    Args:
        metrics: Dictionary mapping model name to its metrics dict.
            Example: {"SVM": {"accuracy": 0.9, "precision": 0.88, ...},
                       "RF": {"accuracy": 0.85, "precision": 0.82, ...}}
        save_path: Path where the figure will be saved.
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        None. Figure is saved to save_path and closed.
    """
    if cfg is None:
        cfg = get_config()
    viz = cfg.visualization

    model_names = list(metrics.keys())
    # Use metric keys from the first model (assumes all models share same metrics)
    metric_names = list(metrics[model_names[0]].keys())

    n_metrics = len(metric_names)
    n_models = len(model_names)

    # Bar positioning for grouped bars
    x = np.arange(n_metrics)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=viz.figure_size_wide)

    # Assign distinct colors to each model
    colors = [viz.bar_color, viz.color_pneumonia]

    for i, model_name in enumerate(model_names):
        values = [metrics[model_name][m] for m in metric_names]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=model_name,
               color=colors[i % len(colors)], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    ax.set_ylim((0.0, 1.05))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=viz.figure_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distribution(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    save_path: Path,
    cfg: PipelineConfig | None = None,
) -> None:
    """Plot and save feature distributions for each class.

    Creates a grid of subplots (one per feature) showing the distribution
    of feature values for NORMAL vs PNEUMONIA classes using histograms.
    This reveals which features separate the classes and how much overlap
    exists between them.

    WHY: Understanding feature distributions helps validate that the
    extracted features actually differentiate healthy from diseased tissue.
    Features with well-separated distributions are more reliable predictors,
    while overlapping distributions may indicate noisy or uninformative features.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label array of shape (n_samples,), 0 for NORMAL, 1 for PNEUMONIA.
        feature_names: List of feature name strings for subplot titles.
        save_path: Path where the figure will be saved.
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        None. Figure is saved to save_path and closed.
    """
    if cfg is None:
        cfg = get_config()
    viz = cfg.visualization

    n_features = X.shape[1]
    # Layout: 2 columns, enough rows for all features
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    axes_flat = axes.flatten() if n_features > 1 else [axes]

    normal_mask = y == 0
    pneumonia_mask = y == 1

    for i, name in enumerate(feature_names):
        ax = axes_flat[i]

        # Histogram for NORMAL class - green indicates healthy tissue
        ax.hist(X[normal_mask, i], bins=30, alpha=0.6, color=viz.color_normal,
                label="NORMAL", density=True)
        # Histogram for PNEUMONIA class - red indicates pathology
        ax.hist(X[pneumonia_mask, i], bins=30, alpha=0.6, color=viz.color_pneumonia,
                label="PNEUMONIA", density=True)

        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    # Hide unused subplots if n_features doesn't fill the grid
    for j in range(n_features, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions by Class", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=viz.figure_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_bar(
    importance: dict[str, float],
    save_path: Path,
    title: str = "Feature Importance",
    cfg: PipelineConfig | None = None,
) -> None:
    """Plot and save a horizontal bar chart of feature importance scores.

    Creates a horizontal bar chart with features sorted by importance
    (highest at top). This provides a clear ranking of which features
    the model relies on most.

    WHY: Feature importance ranking helps clinicians understand what the
    model considers most diagnostic. If the top features align with known
    radiological markers (e.g., contrast for texture changes in pneumonia),
    it validates the model's clinical reasoning.

    Args:
        importance: Dictionary mapping feature name to importance score.
        save_path: Path where the figure will be saved.
        title: Plot title string.
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        None. Figure is saved to save_path and closed.
    """
    if cfg is None:
        cfg = get_config()
    viz = cfg.visualization

    # Sort by importance (ascending for horizontal bars - highest at top)
    sorted_items = sorted(importance.items(), key=lambda x: x[1])
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=viz.figure_size_standard)
    y_pos = np.arange(len(names))

    ax.barh(y_pos, values, color=viz.bar_color, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels at the end of each bar
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=viz.figure_dpi, bbox_inches="tight")
    plt.close(fig)
