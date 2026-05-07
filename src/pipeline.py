"""Main training pipeline for medical image pneumonia detection.

Orchestrates the full workflow: load data, extract features, train models,
evaluate performance, and save results/visualizations.
"""

import sys
from pathlib import Path

# Add project root to sys.path so the script can resolve src/ imports
# when run directly via `uv run python src/pipeline.py`.
# WHY: Python adds the script's directory (src/) to sys.path, not the
# project root. This ensures all src.* imports resolve correctly.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.classification import train_and_evaluate, save_models
from src.feature_extraction import build_feature_matrix
from src.preprocessing import load_dataset_paths
from src.visualization import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix_detailed,
    plot_model_comparison,
    plot_feature_distribution,
    plot_feature_importance_bar,
)
from src.balancing import balance_dataset
from src.explainability import compute_permutation_importance
from src.config import get_config, PipelineConfig

# =============================================================================
# Fixed Constants (not configurable - defined by pipeline structure)
# =============================================================================

# Feature names matching the 8-dimensional feature vector from feature_extraction
# WHY: Fixed list matching the pipeline's feature vector. Not configurable
# because changing features requires code changes anyway.
# Order: 4 GLCM texture features + 4 shape features
FEATURE_NAMES = ["Contrast", "Correlation", "Energy", "Homogeneity", "Area", "Perimeter", "Eccentricity", "Solidity"]

# Visualization output filenames
# WHY: File naming convention, not a pipeline parameter.
CONFUSION_MATRIX_FILE = "confusion_matrix.png"
FEATURE_IMPORTANCE_FILE = "feature_importance.png"
ROC_CURVE_FILE = "roc_curve.png"
PR_CURVE_FILE = "pr_curve.png"
FEATURE_DISTRIBUTION_FILE = "feature_distribution.png"
MODEL_COMPARISON_FILE = "model_comparison.png"
FEATURE_IMPORTANCE_BAR_FILE = "feature_importance_bar.png"
CONFUSION_MATRIX_DETAILED_FILE = "confusion_matrix_detailed.png"


def save_confusion_matrices(results: dict, output_path: Path, cfg: PipelineConfig | None = None) -> None:
    """Save confusion matrix heatmaps for SVM and Random Forest side by side.

    Args:
        results: Dictionary returned by train_and_evaluate() containing confusion matrices.
        output_path: Path where the figure will be saved (models/confusion_matrix.png).
        cfg: Optional pipeline configuration. Defaults to global config.
    
    Why confusion matrix visualization:
        In medical diagnosis, understanding misclassification patterns is critical.
        False negatives (missing pneumonia) are more dangerous than false positives.
        The confusion matrix reveals these patterns at a glance, helping clinicians
        assess model safety for real-world deployment.
    """
    if cfg is None:
        cfg = get_config()
    viz = cfg.visualization

    fig, axes = plt.subplots(1, 2, figsize=viz.figure_size_confusion)

    # SVM confusion matrix
    svm_cm = results['svm']['confusion_matrix']
    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
    axes[0].set_title('SVM Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Random Forest confusion matrix
    rf_cm = results['rf']['confusion_matrix']
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
    axes[1].set_title('Random Forest Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(output_path, dpi=viz.figure_dpi, bbox_inches='tight')
    plt.close(fig)


def save_feature_importance(results: dict, output_path: Path, cfg: PipelineConfig | None = None) -> None:
    """Save feature importance bar chart from Random Forest model.

    Args:
        results: Dictionary returned by train_and_evaluate() containing the RF model.
        output_path: Path where the figure will be saved (models/feature_importance.png).
        cfg: Optional pipeline configuration. Defaults to global config.
    
    Why feature importance chart:
        Medical domain interpretability is essential for clinical adoption.
        Doctors need to understand which image features (texture, shape) drive predictions.
        This visualization builds trust by showing the model's decision criteria,
        enabling domain experts to validate that the model uses medically relevant features.
    """
    if cfg is None:
        cfg = get_config()
    viz = cfg.visualization

    rf_model = results['rf']['model']
    feature_names = ['Contrast', 'Correlation', 'Energy', 'Homogeneity',
                     'Area', 'Perimeter', 'Eccentricity', 'Solidity']

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=viz.figure_size_importance)
    ax.barh(range(len(importances)), importances[indices], align='center')
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Feature Importance')

    plt.tight_layout()
    plt.savefig(output_path, dpi=viz.figure_dpi, bbox_inches='tight')
    plt.close(fig)


def print_comparison_table(results: dict) -> None:
    """Print a clean comparison table of classifier performance metrics.

    Args:
        results: Dictionary returned by train_and_evaluate() containing metrics.
    """
    header = f"{'Classifier':<15} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}"
    separator = '-' * len(header)

    print(header)
    print(separator)

    for name, key in [('SVM', 'svm'), ('Random Forest', 'rf')]:
        metrics = results[key]
        row = f"{name:<15} | {metrics['accuracy']:>8.2f} | {metrics['precision']:>9.2f} | {metrics['recall']:>6.2f} | {metrics['f1_score']:>6.2f}"
        print(row)


def main() -> None:
    """Run the full training pipeline."""
    cfg = get_config()
    data_dir = Path(cfg.data_dir)
    models_dir = Path(cfg.models_dir)

    # Step 1: Load dataset paths from configured data directory
    all_paths, all_labels = load_dataset_paths(str(data_dir))

    # Separate train and test splits
    # Why separate train/test splits:
    #   Evaluating generalization is critical in medical ML. Training data
    #   performance is optimistic; test data reveals how the model performs
    #   on unseen patients. This prevents overfitting and ensures the model
    #   will work in real clinical settings with new X-ray images.
    # load_dataset_paths walks through train, val, test directories
    # We need to re-load with split awareness
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    for path, label in zip(all_paths, all_labels):
        path_obj = Path(path)
        # Check if path contains 'train' or 'test' in its parts
        parts = path_obj.parts
        if 'train' in parts:
            train_paths.append(path)
            train_labels.append(label)
        elif 'test' in parts:
            test_paths.append(path)
            test_labels.append(label)

    print(f"Train set: {len(train_paths)} images")
    print(f"Test set: {len(test_paths)} images")

    # Step 2: Build feature matrix for train set
    print("Building feature matrix for training set...")
    X_train, y_train = build_feature_matrix(train_paths, train_labels, cfg)

    # Step 3: Build feature matrix for test set
    print("Building feature matrix for test set...")
    X_test, y_test = build_feature_matrix(test_paths, test_labels, cfg)

    # Step 4: Train and evaluate
    print("Training and evaluating models...")
    results = train_and_evaluate(X_train, y_train, X_test, y_test, cfg)

    # Step 5: Print comparison table
    print("\n=== Model Comparison ===")
    print_comparison_table(results)

    # Step 6: Save models to configured models directory
    print(f"\nSaving models to {models_dir}/...")
    save_models(results, str(models_dir))

    # Step 7: Save confusion matrix figure
    confusion_path = models_dir / CONFUSION_MATRIX_FILE
    print(f"Saving confusion matrix to {confusion_path}...")
    save_confusion_matrices(results, confusion_path, cfg)

    # Step 8: Save feature importance bar chart
    importance_path = models_dir / FEATURE_IMPORTANCE_FILE
    print(f"Saving feature importance to {importance_path}...")
    save_feature_importance(results, importance_path, cfg)

    # -------------------------------------------------------------------------
    # Step 9: Apply class balancing and retrain (if enabled)
    # Why class balancing: Medical datasets often have class imbalance where
    # pneumonia cases outnumber normal cases (or vice versa). Balancing ensures
    # the model doesn't become biased toward the majority class, which would
    # lead to missed diagnoses for the underrepresented class.
    # -------------------------------------------------------------------------
    if cfg.balancing.apply_balancing:
        print("\nApplying class balancing...")
        X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, cfg=cfg)
        # Retrain with balanced data
        results_balanced = train_and_evaluate(X_train_balanced, y_train_balanced, X_test, y_test, cfg)
        print("\n=== Model Comparison (Balanced) ===")
        print_comparison_table(results_balanced)

    # -------------------------------------------------------------------------
    # Step 10: Generate advanced visualizations
    # Why advanced visualizations: ROC and PR curves reveal threshold-dependent
    # performance critical for clinical decision-making. Detailed confusion
    # matrices show error patterns. Feature distributions validate that
    # extracted features actually separate disease from healthy tissue.
    # -------------------------------------------------------------------------
    # Prepare scaled test data and model outputs for visualization
    X_test_scaled = results['scaler'].transform(X_test)
    svm_proba = results['svm']['model'].predict_proba(X_test_scaled)
    svm_preds = results['svm']['model'].predict(X_test_scaled)

    # ROC curve - shows sensitivity/specificity trade-off across thresholds
    roc_path = models_dir / ROC_CURVE_FILE
    print(f"Saving ROC curve to {roc_path}...")
    plot_roc_curve(y_test, svm_proba[:, 1], roc_path, cfg=cfg)

    # Precision-Recall curve - more informative than ROC for imbalanced data
    pr_path = models_dir / PR_CURVE_FILE
    print(f"Saving PR curve to {pr_path}...")
    plot_precision_recall_curve(y_test, svm_proba[:, 1], pr_path, cfg=cfg)

    # Detailed confusion matrix with counts and percentages
    cm_detailed_path = models_dir / CONFUSION_MATRIX_DETAILED_FILE
    print(f"Saving detailed confusion matrix to {cm_detailed_path}...")
    plot_confusion_matrix_detailed(y_test, svm_preds, ["NORMAL", "PNEUMONIA"], cm_detailed_path, cfg=cfg)

    # Model comparison bar chart - direct visual comparison across metrics
    # Extract only float metrics (exclude model object and confusion matrix)
    svm_metrics = {
        'accuracy': results['svm']['accuracy'],
        'precision': results['svm']['precision'],
        'recall': results['svm']['recall'],
        'f1_score': results['svm']['f1_score'],
    }
    rf_metrics = {
        'accuracy': results['rf']['accuracy'],
        'precision': results['rf']['precision'],
        'recall': results['rf']['recall'],
        'f1_score': results['rf']['f1_score'],
    }
    comparison_path = models_dir / MODEL_COMPARISON_FILE
    print(f"Saving model comparison to {comparison_path}...")
    plot_model_comparison({"SVM": svm_metrics, "RF": rf_metrics}, comparison_path, cfg=cfg)

    # Feature distribution by class - validates feature discriminative power
    dist_path = models_dir / FEATURE_DISTRIBUTION_FILE
    print(f"Saving feature distribution to {dist_path}...")
    plot_feature_distribution(X_test, y_test, FEATURE_NAMES, dist_path, cfg=cfg)

    # Feature importance bar chart from Random Forest feature_importances_
    importance = dict(zip(FEATURE_NAMES, results['rf']['model'].feature_importances_))
    importance_bar_path = models_dir / FEATURE_IMPORTANCE_BAR_FILE
    print(f"Saving feature importance bar chart to {importance_bar_path}...")
    plot_feature_importance_bar(importance, importance_bar_path, cfg=cfg)

    # -------------------------------------------------------------------------
    # Step 11: Compute permutation importance for both models
    # Why permutation importance: Unlike RF's built-in feature_importances_,
    # permutation importance is model-agnostic and directly measures how much
    # each feature contributes to accuracy. Computing it for both SVM and RF
    # reveals whether both models rely on the same features, which strengthens
    # clinical confidence in the identified diagnostic markers.
    # -------------------------------------------------------------------------
    print("\n=== SVM Permutation Importance ===")
    svm_importance = compute_permutation_importance(results['svm']['model'], X_test_scaled, y_test, cfg=cfg)
    for name, score in sorted(svm_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score:.4f}")

    print("\n=== Random Forest Permutation Importance ===")
    rf_importance = compute_permutation_importance(results['rf']['model'], X_test_scaled, y_test, cfg=cfg)
    for name, score in sorted(rf_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score:.4f}")

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
