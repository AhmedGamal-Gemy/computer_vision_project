"""Streamlit web application for medical image analysis.

Provides an interactive interface for uploading chest X-ray images,
running preprocessing, segmentation, feature extraction, and classification
using trained SVM or Random Forest models.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path so Streamlit can resolve src/ imports.
# WHY: Streamlit runs the script from the app/ directory context, which
# means the project root (where src/ lives) is not on the Python path.
# This is a standard pattern for Streamlit apps in subdirectories.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.classification import load_best_model, predict_single
from src.explainability import explain_prediction, generate_prediction_report
from src.feature_extraction import build_feature_matrix
from src.preprocessing import load_dataset_paths, preprocess_image
from src.segmentation import otsu_segmentation, kmeans_segmentation
from src.config import get_config, PipelineConfig

# =============================================================================
# FIXED CONSTANTS (not configurable - defined by pipeline structure)
# =============================================================================

# Streamlit default port for local development
STREAMLIT_PORT = 8501

# Feature names for extracted GLCM and shape features - must match
# the order returned by the feature extraction pipeline
# WHY: Fixed list matching the pipeline's feature vector.
FEATURE_NAMES = [
    "Contrast", "Correlation", "Energy", "Homogeneity",
    "Area", "Perimeter", "Eccentricity", "Solidity",
]

# Preset augmentation configurations for preview display
# WHY: UI-specific presets that control which transforms are toggled on/off.
# These are separate from the parameter ranges in AugmentationConfig.
AUGMENTATION_PRESETS = [
    {"rotation": True, "brightness": False, "contrast": False, "noise": False, "zoom": False, "shear": False, "translation": False, "horizontal_flip": False},
    {"rotation": False, "brightness": True, "contrast": True, "noise": False, "zoom": False, "shear": False, "translation": False, "horizontal_flip": False},
    {"rotation": False, "brightness": False, "contrast": False, "noise": True, "zoom": False, "shear": False, "translation": False, "horizontal_flip": False},
    {"rotation": False, "brightness": False, "contrast": False, "noise": False, "zoom": True, "shear": False, "translation": False, "horizontal_flip": True},
]

# Labels corresponding to each augmentation preset
AUGMENTATION_LABELS = ["Rotation", "Brightness+Contrast", "Noise", "Zoom+Flip"]

# Directory where trained models and visualizations are stored
# WHY: Centralized path for all model artifacts - used by both model loading
# and the Model Report tab to display saved visualization images.
MODELS_DIR = Path("models")

# Visualization files with human-readable captions for the Model Report tab
# WHY: Mapping keeps file names and display captions decoupled. Adding or
# removing visualizations only requires updating this dict, not the UI code.
VISUALIZATION_FILES = {
    "confusion_matrix.png": "Confusion Matrices (SVM vs Random Forest)",
    "confusion_matrix_detailed.png": "Detailed Confusion Matrix (Counts & Percentages)",
    "roc_curve.png": "ROC Curve (Sensitivity vs 1-Specificity)",
    "pr_curve.png": "Precision-Recall Curve",
    "model_comparison.png": "Model Performance Comparison",
    "feature_importance.png": "Random Forest Feature Importance",
    "feature_importance_bar.png": "Feature Importance Ranking",
    "feature_distribution.png": "Feature Distributions (NORMAL vs PNEUMONIA)",
}


@st.cache_data
def compute_model_metrics(_scaler, _svm_model, _rf_model, cfg):
    """Compute and return metrics for both models on the test set.

    WHY: Streamlit reruns the entire script on every interaction. Without
    caching, metrics would be recomputed from scratch on each tab switch or
    widget change, causing unnecessary delays. @st.cache_data stores the
    result keyed by the config object, so recomputation only happens when
    config changes. Leading underscores on model/scaler args tell Streamlit
    not to hash them (sklearn objects are not hashable).

    Args:
        _scaler: Fitted StandardScaler for feature scaling (not hashed).
        _svm_model: Trained SVM classifier (not hashed).
        _rf_model: Trained Random Forest classifier (not hashed).
        cfg: Pipeline configuration with data_dir path.

    Returns:
        Dictionary with "SVM" and "Random Forest" keys, each mapping to
        a dict of accuracy, precision, recall, f1, total_samples,
        normal_count, and pneumonia_count.
    """
    # Load test set paths
    all_paths, all_labels = load_dataset_paths(str(Path(cfg.data_dir)))
    test_paths, test_labels = [], []
    for path, label in zip(all_paths, all_labels):
        if "test" in Path(path).parts:
            test_paths.append(path)
            test_labels.append(label)

    # Build features (show progress)
    with st.spinner("Computing test features..."):
        X_test, y_test = build_feature_matrix(test_paths, test_labels)

    X_test_scaled = _scaler.transform(X_test)

    # Compute metrics for both models
    metrics = {}
    for name, model in [("SVM", _svm_model), ("Random Forest", _rf_model)]:
        preds = model.predict(X_test_scaled)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "total_samples": len(y_test),
            "normal_count": int(sum(1 for y in y_test if y == 0)),
            "pneumonia_count": int(sum(1 for y in y_test if y == 1)),
        }
    return metrics


def run_pipeline(image_path: str, scaler, model, cfg: PipelineConfig | None = None):
    """Run the full analysis pipeline on a single image.

    Preprocesses the image, runs Otsu and K-Means segmentation,
    extracts features, and predicts the class.

    Args:
        image_path: Path to the uploaded image file.
        scaler: Fitted StandardScaler for feature scaling.
        model: Trained classifier (SVM or Random Forest).
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        A dictionary containing:
        - original: The original loaded image (grayscale).
        - preprocessed: The preprocessed image.
        - mask: The Otsu segmentation binary mask.
        - kmeans_result: The K-Means segmentation result.
        - prediction: 'NORMAL' or 'PNEUMONIA'.
        - confidence: Prediction confidence as a float [0, 1].
        - features: Numpy array of 8 extracted feature values.
    """
    if cfg is None:
        cfg = get_config()

    # Load original image
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    # Preprocess
    preprocessed = preprocess_image(image_path, cfg)

    # Otsu segmentation
    mask, _ = otsu_segmentation(preprocessed, cfg)

    # K-Means segmentation with configured k for visualizing lung regions,
    # background, and other anatomical structures
    kmeans_result = kmeans_segmentation(preprocessed, k=cfg.segmentation.kmeans_k, cfg=cfg)

    # Prediction
    result = predict_single(image_path, scaler, model, cfg)

    return {
        "original": original,
        "preprocessed": preprocessed,
        "mask": mask,
        "kmeans_result": kmeans_result,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "features": result["features"],
    }


def main() -> None:
    """Main Streamlit application entry point."""
    cfg = get_config()

    # --- Page Configuration ---
    # Browser tab shows medical context with lung emoji for quick visual identification
    st.set_page_config(
        page_title=cfg.app.page_title,
        page_icon=cfg.app.page_icon,
        layout="wide",
    )

    # --- Header ---
    st.title(f"{cfg.app.page_icon} Medical Image Analysis Assistant")
    st.subheader("Chest X-Ray Classification — Normal vs Pneumonia")

    # --- Sidebar ---
    model_choice = st.sidebar.radio("Select Model", ["SVM", "Random Forest"])
    st.sidebar.info(
        "Pipeline: Gaussian + Median Filtering → Otsu Segmentation → "
        "GLCM + Shape Features → SVM/RF Classification"
    )

    # Sidebar toggles for optional UI sections
    show_explainability = st.sidebar.checkbox(
        "Show Explainability", value=cfg.app.show_explainability_default
    )
    show_augmentation = st.sidebar.checkbox(
        "Show Augmentation Preview", value=cfg.app.show_augmentation_default
    )

    # --- Load Models (cached) ---
    # Cache models in memory to avoid reloading on every user interaction.
    # Streamlit reruns the script on each widget change, so caching prevents
    # expensive model loading from disk multiple times.
    @st.cache_resource
    def load_models():
        try:
            return load_best_model(str(Path("models")))
        except FileNotFoundError:
            return None

    loaded = load_models()
    if loaded is None:
        st.warning("Run `uv run python src/pipeline.py` first to train the models.")
        st.stop()

    scaler, svm_model, rf_model = loaded

    # Select model based on sidebar choice
    model = svm_model if model_choice == "SVM" else rf_model

    # --- Tab Navigation ---
    # Convert from single-page to tabbed layout. The Analysis tab preserves
    # all existing functionality. The Model Report tab adds a dashboard for
    # model performance metrics and visualization gallery.
    tab_analysis, tab_report = st.tabs(["🔬 Analysis", "📊 Model Report"])

    # =========================================================================
    # TAB 1: Analysis (existing functionality preserved)
    # =========================================================================
    with tab_analysis:
        # --- Main Area: File Uploader ---
        # Accept only common image formats that OpenCV can process reliably
        uploaded_file = st.file_uploader("Upload Chest X-Ray", type=cfg.app.accepted_file_types)

        if uploaded_file is not None:
            # Save uploaded file to a temporary location.
            # Streamlit's file_uploader returns bytes, but OpenCV's imread()
            # requires a file path, so we write to a temp file first.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Run the full pipeline
                results = run_pipeline(tmp_path, scaler, model, cfg)

                # --- Two-Column Results Layout ---
                # 2-column layout separates visual outputs (left) from numerical results (right).
                # This improves UX by allowing users to compare images and metrics side-by-side
                # without scrolling, making it easier to correlate visual changes with predictions.
                col1, col2 = st.columns(2)

                # Left column: Visual outputs
                with col1:
                    st.image(results["original"], caption="Original Image")
                    st.image(results["preprocessed"], caption="Preprocessed (Gaussian + Median)")
                    # clamp=True ensures binary mask (0-1 range) displays correctly
                    # without color distortion from out-of-range values
                    st.image(results["mask"], caption="Otsu Segmentation Mask", clamp=True)
                    st.image(results["kmeans_result"], caption=f"K-Means Segmentation (k={cfg.segmentation.kmeans_k})")

                # Right column: Prediction results
                with col2:
                    # Prediction badge
                    if results["prediction"] == "NORMAL":
                        st.success("✅ NORMAL")
                    else:
                        st.error("🔴 PNEUMONIA")

                    # Confidence metric
                    st.metric("Confidence", f"{results['confidence']:.2%}")

                    # Feature table
                    feature_df = pd.DataFrame({
                        "Feature": FEATURE_NAMES,
                        "Value": results["features"],
                    })
                    st.table(feature_df)

                    # Bar chart
                    chart_df = pd.DataFrame(
                        {"Value": results["features"]},
                        index=pd.Index(FEATURE_NAMES),
                    )
                    st.bar_chart(chart_df)

                    # Explainability panel - shows WHY the model made its prediction
                    # using perturbation-based feature contribution analysis
                    if show_explainability:
                        with st.expander("🔍 Model Explainability"):
                            explanation = explain_prediction(
                                model, scaler, results["features"]
                            )
                            report = generate_prediction_report(explanation, cfg)
                            st.text(report)

                            # Show feature contributions as a horizontal bar chart
                            contrib_df = pd.DataFrame(
                                {"Contribution": list(explanation["feature_contributions"].values())},
                                index=pd.Index(explanation["feature_contributions"].keys()),
                            )
                            st.bar_chart(contrib_df, horizontal=True)

            finally:
                # Clean up temporary file
                Path(tmp_path).unlink(missing_ok=True)

            # Data Augmentation Preview - shows how the uploaded image would look
            # under different augmentation transforms used during training
            if show_augmentation:
                from src.augmentation import augment_image

                with st.expander("🔄 Data Augmentation Preview"):
                    st.caption(
                        "Preview of augmentation transforms applied to the uploaded image"
                    )
                    # Show 4 augmented versions in a 2x2 grid
                    aug_col1, aug_col2, aug_col3, aug_col4 = st.columns(4)
                    for col, config, label in zip(
                        [aug_col1, aug_col2, aug_col3, aug_col4],
                        AUGMENTATION_PRESETS,
                        AUGMENTATION_LABELS,
                    ):
                        with col:
                            aug_img = augment_image(results["preprocessed"], config, cfg)
                            st.image(aug_img, caption=label)

    # =========================================================================
    # TAB 2: Model Report (new dashboard)
    # =========================================================================
    with tab_report:
        st.header("Model Performance Dashboard")

        # --- Section A: Model Performance Metrics ---
        st.subheader("📈 Performance Metrics")

        metrics = compute_model_metrics(scaler, svm_model, rf_model, cfg)

        # SVM metrics row
        st.markdown("#### SVM Classifier")
        svm_cols = st.columns(4)
        svm_cols[0].metric("Accuracy", f"{metrics['SVM']['accuracy']:.4f}")
        svm_cols[1].metric("Precision", f"{metrics['SVM']['precision']:.4f}")
        svm_cols[2].metric("Recall", f"{metrics['SVM']['recall']:.4f}")
        svm_cols[3].metric("F1 Score", f"{metrics['SVM']['f1']:.4f}")

        # Random Forest metrics row
        st.markdown("#### Random Forest Classifier")
        rf_cols = st.columns(4)
        rf_cols[0].metric("Accuracy", f"{metrics['Random Forest']['accuracy']:.4f}")
        rf_cols[1].metric("Precision", f"{metrics['Random Forest']['precision']:.4f}")
        rf_cols[2].metric("Recall", f"{metrics['Random Forest']['recall']:.4f}")
        rf_cols[3].metric("F1 Score", f"{metrics['Random Forest']['f1']:.4f}")

        # --- Section B: Dataset Statistics ---
        st.subheader("📋 Dataset Statistics")

        # Compute dataset stats from the metrics (test set breakdown)
        test_total = metrics["SVM"]["total_samples"]
        test_normal = metrics["SVM"]["normal_count"]
        test_pneumonia = metrics["SVM"]["pneumonia_count"]

        dataset_cols = st.columns(4)
        dataset_cols[0].metric("Test Samples", test_total)
        dataset_cols[1].metric("NORMAL (Test)", test_normal)
        dataset_cols[2].metric("PNEUMONIA (Test)", test_pneumonia)
        dataset_cols[3].metric("Features", len(FEATURE_NAMES))

        # Feature names table
        st.caption(f"Feature vector ({len(FEATURE_NAMES)} dimensions): {', '.join(FEATURE_NAMES)}")

        # --- Section C: Visualization Gallery ---
        st.subheader("🖼️ Visualization Gallery")

        for filename, caption in VISUALIZATION_FILES.items():
            img_path = MODELS_DIR / filename
            if img_path.exists():
                st.image(str(img_path), caption=caption)
            else:
                st.warning(f"⚠️ Visualization not found: `{filename}`")

        # --- Section D: Feature Importance Table ---
        st.subheader("🔬 Feature Importance (Random Forest)")

        importances = rf_model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        st.dataframe(
            importance_df.style.format({"Importance": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

    # --- Footer ---
    st.markdown("---")
    st.caption(
        "Pipeline: Gaussian + Median Filtering → Otsu Segmentation → "
        "GLCM + Shape Features → SVM/RF Classification"
    )


if __name__ == "__main__":
    main()
