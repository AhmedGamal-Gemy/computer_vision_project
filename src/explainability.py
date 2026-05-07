"""
Feature-level model explainability for medical image classification.

Provides perturbation-based and permutation-based feature importance analysis
for SVM and Random Forest models trained on GLCM texture features. This
approach is appropriate for traditional ML models where gradient-based
methods (Grad-CAM) are not applicable.

Doctors need to understand WHICH features drove a prediction, not just the
output label. This module makes model decisions interpretable at the
feature level.
"""

import numpy as np
from sklearn.metrics import accuracy_score

from .config import get_config, PipelineConfig

# Feature names matching the 8-dimensional feature vector from feature_extraction
# WHY: Fixed list matching the pipeline's feature vector. Not configurable
# because changing features requires code changes anyway.
# Order: 4 GLCM texture features + 4 shape features
FEATURE_NAMES = [
    "Contrast", "Correlation", "Energy", "Homogeneity",
    "Area", "Perimeter", "Eccentricity", "Solidity"
]

# Class label mapping for human-readable prediction output
# WHY: Fixed mapping - 0 = NORMAL, 1 = PNEUMONIA. Defined by dataset.
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


def explain_prediction(
    model,
    scaler,
    features: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict:
    """Explain a single prediction by measuring per-feature contribution.

    Uses a perturbation approach: for each feature, replace it with the
    training-set mean (from scaler.mean_) and measure how much the predicted
    probability changes. A larger change means that feature was more
    influential for this specific prediction.

    WHY: Doctors need to understand WHY the model made a prediction, not just
    the output. Showing which texture/shape features drove the decision builds
    trust and enables clinical validation of the model's reasoning.

    Args:
        model: Trained sklearn classifier with predict() and predict_proba()
            methods (SVC with probability=True or RandomForestClassifier).
        scaler: Fitted StandardScaler used during training.
        features: Single feature vector of shape (8,) with raw (unscaled) values.
        feature_names: Optional list of feature names. Defaults to FEATURE_NAMES.

    Returns:
        A dictionary with the following structure:
        {
            "prediction": "NORMAL" or "PNEUMONIA",
            "confidence": float (0.0-1.0, max probability),
            "feature_contributions": dict mapping feature name to contribution
                score (absolute probability change when feature is replaced),
            "top_features": list of top 3 most influential feature names,
        }
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    # Scale the original feature vector
    features_2d = features.reshape(1, -1)
    features_scaled = scaler.transform(features_2d)

    # Get baseline prediction and probability
    pred_class = model.predict(features_scaled)[0]
    pred_proba = model.predict_proba(features_scaled)[0]
    confidence = float(np.max(pred_proba))

    # Baseline probability for the predicted class
    baseline_prob = pred_proba[pred_class]

    # Compute per-feature contribution using perturbation
    # Replace each feature with the training mean and measure probability drop
    contributions: dict[str, float] = {}
    for i in range(features.shape[0]):
        perturbed = features_scaled.copy()
        # Replace feature with the mean value from training data (scaler.mean_)
        # This approximates "what if this feature had no informative value"
        perturbed[0, i] = 0.0  # After scaling, mean maps to 0

        perturbed_proba = model.predict_proba(perturbed)[0]
        # Contribution = how much the predicted class probability drops
        # when this feature is replaced with its mean
        contribution = abs(baseline_prob - perturbed_proba[pred_class])
        contributions[feature_names[i]] = float(contribution)

    # Identify top 3 most influential features
    sorted_features = sorted(
        contributions.keys(), key=lambda f: contributions[f], reverse=True
    )
    top_features = sorted_features[:3]

    return {
        "prediction": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "feature_contributions": contributions,
        "top_features": top_features,
    }


def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    cfg: PipelineConfig | None = None,
) -> dict[str, float]:
    """Compute permutation-based feature importance for a trained model.

    For each feature, shuffles its values across samples N_REPEATS times and
    measures the accuracy drop compared to the baseline. A larger drop means
    the model relies more on that feature. This works for any sklearn model
    including SVM and Random Forest.

    WHY: Permutation importance is model-agnostic and directly measures how
    much each feature contributes to prediction accuracy. Unlike coefficient-
    based importance, it captures non-linear dependencies and interactions.

    Args:
        model: Trained sklearn classifier with predict() method.
        X: Feature matrix of shape (n_samples, n_features), already scaled.
        y: True labels of shape (n_samples,).
        feature_names: Optional list of feature names. Defaults to FEATURE_NAMES.
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        A dictionary mapping feature name to its average importance score
        (mean accuracy drop across permutation repeats).
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
    if cfg is None:
        cfg = get_config()

    # Baseline accuracy with unpermuted features
    baseline_accuracy = accuracy_score(y, model.predict(X))

    importance: dict[str, float] = {}

    for i, name in enumerate(feature_names):
        accuracy_drops: list[float] = []

        for _ in range(cfg.explainability.permutation_n_repeats):
            # Shuffle the i-th feature column, breaking its relationship with y
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

            # Measure accuracy drop - larger drop = more important feature
            permuted_accuracy = accuracy_score(y, model.predict(X_permuted))
            drop = baseline_accuracy - permuted_accuracy
            accuracy_drops.append(max(0.0, drop))  # Drop can't be negative

        # Average drop across repeats for stable estimate
        importance[name] = float(np.mean(accuracy_drops))

    return importance


def generate_prediction_report(explanation: dict, cfg: PipelineConfig | None = None) -> str:
    """Generate a human-readable report from a prediction explanation.

    Takes the output of explain_prediction() and formats it as a clear
    text report suitable for clinical review, showing the prediction,
    confidence, and top contributing features with influence levels.

    WHY: Raw numbers are hard for clinicians to interpret. A structured
    text report with influence labels (high/moderate/low) makes model
    reasoning accessible without requiring statistical expertise.

    Args:
        explanation: Dictionary returned by explain_prediction(), containing
            "prediction", "confidence", "feature_contributions", and "top_features".
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        A formatted string report with prediction, confidence percentage,
        and top 3 contributing features with influence classification.
    """
    if cfg is None:
        cfg = get_config()
    expl = cfg.explainability

    prediction = explanation["prediction"]
    confidence_pct = explanation["confidence"] * 100

    lines: list[str] = []
    lines.append(f"Prediction: {prediction} (confidence: {confidence_pct:.1f}%)")
    lines.append("Top contributing features:")

    contributions = explanation["feature_contributions"]
    top_features = explanation["top_features"]

    for rank, feature_name in enumerate(top_features, start=1):
        score = contributions[feature_name]

        # Classify influence level based on configured thresholds
        if score >= expl.high_influence_threshold:
            influence = "high influence"
        elif score >= expl.moderate_influence_threshold:
            influence = "moderate influence"
        else:
            influence = "low influence"

        lines.append(f"  {rank}. {feature_name}: {score:.2f} ({influence})")

    return "\n".join(lines)
