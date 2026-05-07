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

from src.classification import load_best_model, predict_single
from src.explainability import explain_prediction, generate_prediction_report
from src.preprocessing import preprocess_image
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

    # --- Footer ---
    st.markdown("---")
    st.caption(
        "Pipeline: Gaussian + Median Filtering → Otsu Segmentation → "
        "GLCM + Shape Features → SVM/RF Classification"
    )


if __name__ == "__main__":
    main()
