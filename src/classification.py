"""Classification module for medical image pneumonia detection.

Provides functions to train, evaluate, save, load, and use machine learning models
(SVM and Random Forest) for pneumonia classification from extracted features.
"""

import joblib
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from .preprocessing import preprocess_image
from .segmentation import otsu_segmentation
from .feature_extraction import extract_features
from .config import get_config, PipelineConfig


# Class label mapping for interpretation
# WHY: Fixed mapping - 0 = NORMAL (healthy lung), 1 = PNEUMONIA (infected lung).
# Not configurable because label semantics are defined by the dataset.
CLASS_LABELS = {0: "NORMAL", 1: "PNEUMONIA"}


def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       cfg: PipelineConfig | None = None) -> dict:
    """Train SVM and Random Forest classifiers and evaluate their performance.

    Scales the input features using StandardScaler, trains both SVM (RBF kernel)
    and Random Forest classifiers, and computes evaluation metrics for each.

    Args:
        X_train: Training feature matrix of shape (N_train, 8).
        y_train: Training labels of shape (N_train,) with values 0 (NORMAL) or 1 (PNEUMONIA).
        X_test: Test feature matrix of shape (N_test, 8).
        y_test: Test labels of shape (N_test,) with values 0 (NORMAL) or 1 (PNEUMONIA).
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        A dictionary with the following structure:
        {
            'svm': {
                'model': SVC object,
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1_score': float,
                'confusion_matrix': np.ndarray
            },
            'rf': {
                'model': RandomForestClassifier object,
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1_score': float,
                'confusion_matrix': np.ndarray
            },
            'scaler': StandardScaler object used for feature scaling
        }
    """
    if cfg is None:
        cfg = get_config()
    clf = cfg.classification

    # Scale features using StandardScaler
    # Features have different scales (e.g., intensity vs texture metrics)
    # SVM and RF both benefit from normalized features for better convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM with RBF kernel
    # RBF kernel handles non-linear separation of features in medical imaging
    svm = SVC(
        kernel=clf.svm_kernel,
        C=clf.svm_c,
        gamma=clf.svm_gamma,
        probability=True,
        random_state=clf.svm_random_state
    )
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)

    # Train Random Forest classifier
    # Ensemble method is robust for medical data where prediction stability is critical
    rf = RandomForestClassifier(
        n_estimators=clf.rf_n_estimators,
        random_state=clf.rf_random_state
    )
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)

    # Compute metrics for SVM
    svm_metrics = {
        'model': svm,
        'accuracy': float(accuracy_score(y_test, svm_pred)),
        'precision': float(precision_score(y_test, svm_pred)),
        'recall': float(recall_score(y_test, svm_pred)),
        'f1_score': float(f1_score(y_test, svm_pred)),
        'confusion_matrix': confusion_matrix(y_test, svm_pred)
    }

    # Compute metrics for Random Forest
    rf_metrics = {
        'model': rf,
        'accuracy': float(accuracy_score(y_test, rf_pred)),
        'precision': float(precision_score(y_test, rf_pred)),
        'recall': float(recall_score(y_test, rf_pred)),
        'f1_score': float(f1_score(y_test, rf_pred)),
        'confusion_matrix': confusion_matrix(y_test, rf_pred)
    }

    return {
        'svm': svm_metrics,
        'rf': rf_metrics,
        'scaler': scaler
    }


def save_models(results: dict, models_dir: str) -> None:
    """Save trained models and scaler to disk using joblib.

    Creates the output directory if it doesn't exist and saves the scaler,
    SVM model, and Random Forest model as separate pickle files.

    Args:
        results: Dictionary returned by train_and_evaluate() containing models and scaler.
        models_dir: Path to the directory where models will be saved (string).

    Returns:
        None
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(results['scaler'], models_path / 'scaler.pkl')
    joblib.dump(results['svm']['model'], models_path / 'svm_model.pkl')
    joblib.dump(results['rf']['model'], models_path / 'rf_model.pkl')


def load_best_model(models_dir: str) -> Tuple[StandardScaler, SVC, RandomForestClassifier]:
    """Load saved scaler and models from disk.

    Loads the scaler, SVM model, and Random Forest model from the specified directory.
    Uses try/except to handle missing or corrupted model files gracefully.

    Args:
        models_dir: Path to the directory containing saved models (string).

    Returns:
        A tuple containing:
        - scaler: Loaded StandardScaler object
        - svm_model: Loaded SVM model (SVC object)
        - rf_model: Loaded Random Forest model (RandomForestClassifier object)

    Raises:
        FileNotFoundError: If any of the model files cannot be loaded.
    """
    models_path = Path(models_dir)

    try:
        scaler = joblib.load(models_path / 'scaler.pkl')
        svm_model = joblib.load(models_path / 'svm_model.pkl')
        rf_model = joblib.load(models_path / 'rf_model.pkl')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load model files from {models_path}: {e}")

    return (scaler, svm_model, rf_model)


def predict_single(image_path: str, scaler: StandardScaler, model: Union[SVC, RandomForestClassifier], cfg: PipelineConfig | None = None) -> dict:
    """Run prediction on a single medical image.

    Preprocesses the image, segments it using Otsu thresholding, extracts features,
    scales the features using the provided scaler, and runs prediction with the model.

    Args:
        image_path: Path to the input image file (string).
        scaler: Fitted StandardScaler object for feature scaling.
        model: Trained classifier with predict() and predict_proba() methods.
            Can be either SVC (SVM) or RandomForestClassifier.
        cfg: Optional pipeline configuration. Defaults to global config.

    Returns:
        A dictionary with the following structure:
        {
            'prediction': 'NORMAL' if predicted class is 0, 'PNEUMONIA' if 1,
            'confidence': float (maximum probability from predict_proba),
            'features': np.ndarray of 8 extracted feature values
        }
    """
    if cfg is None:
        cfg = get_config()

    # Preprocess the image
    img = preprocess_image(image_path, cfg)

    # Segment the image
    mask, _ = otsu_segmentation(img, cfg)

    # Extract features
    features = extract_features(img, mask, cfg)

    # Scale features
    features_scaled = scaler.transform([features])[0]

    # Predict
    pred = model.predict([features_scaled])[0]
    prob = model.predict_proba([features_scaled])[0]

    return {
        'prediction': CLASS_LABELS[pred],
        'confidence': float(max(prob)),
        'features': features
    }
