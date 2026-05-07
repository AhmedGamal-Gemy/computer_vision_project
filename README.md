# Medical Image Analysis Assistant

A computer vision pipeline for preprocessing, segmenting, and classifying chest X-ray images to assist in medical diagnosis.

## Requirements

- Python 3.10+
- uv package manager
- Kaggle account (for manual dataset download)

## Setup Steps

1. Clone the repository
2. Initialize the project with uv:
   ```bash
   uv init medical-cv && cd medical-cv
   ```
3. Install dependencies:
   ```bash
   uv add opencv-python scikit-image scikit-learn streamlit joblib matplotlib seaborn pandas numpy
   ```
4. Download the dataset from Kaggle:
   - Go to [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   - Download the dataset manually (Kaggle CLI is not configured in this project)
   - Extract the contents into the `data/` directory
5. Train the models:
   ```bash
   uv run python src/pipeline.py
   ```
6. Launch the web application:
   ```bash
   uv run streamlit run app/app.py
   ```

## Pipeline Diagram

```
+-------------+     +-------------+     +-------------+     +-------------+     +-------------+     +-------------+
|  Augment    |---->|  Preprocess  |---->|  Segment    |---->|  Feature    |---->|  Balance    |---->|  Classify   |
|  (Rotation,  |     |  (Gaussian+ |     |  (Otsu+     |     |  (GLCM+     |     |  (Over/Under|     |  (SVM/RF)   |
|   Flip, etc) |     |   Median)    |     |  K-Means)   |     |  Shape)     |     |   sample)   |     |             |
+-------------+     +-------------+     +-------------+     +-------------+     +-------------+     +-------------+
                                                                                      |
                                                                                      v
                                                                              +-------------+
                                                                              |  Explain    |
                                                                              |  (Feature   |
                                                                              |  Importance)|
                                                                              +-------------+
```

## Feature List

The pipeline extracts the following features from each image:

- **Contrast (GLCM)** - Measures local variations in pixel intensity
- **Correlation (GLCM)** - Measures linear dependency of gray levels
- **Energy (GLCM)** - Sum of squared elements in GLCM
- **Homogeneity (GLCM)** - Measures closeness of distribution to diagonal
- **Area (Shape, normalized)** - Normalized area of largest segmented region
- **Perimeter (Shape)** - Perimeter of largest segmented region
- **Eccentricity (Shape)** - Ratio of major to minor axis length
- **Solidity (Shape)** - Ratio of region area to convex hull area

## Data Augmentation

The pipeline supports configurable image augmentation to expand training data:

- **Geometric**: Rotation (+-15 degrees), horizontal flip, zoom (0.9-1.1x), shear (+-5 degrees), translation (10%)
- **Intensity**: Brightness (0.8-1.2x), contrast (0.8-1.2x), Gaussian noise
- **Medical-aware**: No vertical flips (anatomically invalid), limited rotation angles

## Class Balancing

Address dataset imbalance with three strategies:

- **Oversampling**: Duplicate minority class with slight Gaussian noise
- **Undersampling**: Randomly sample majority class to match minority
- **SMOTE**: Generate synthetic minority samples via interpolation

Toggle balancing in `src/pipeline.py`: `APPLY_BALANCING = True`

## Model Explainability

Understand WHY the model made each prediction:

- **Perturbation-based analysis**: Measures how much each feature contributes to a specific prediction
- **Permutation importance**: Model-agnostic feature importance via accuracy drop on shuffled features
- **Human-readable reports**: Top contributing features with influence labels (high/moderate/low)

## Visualization Outputs

The pipeline generates the following visualizations in `models/`:

- `confusion_matrix.png` - Side-by-side SVM vs RF confusion matrices
- `confusion_matrix_detailed.png` - Detailed confusion matrix with counts and percentages
- `roc_curve.png` - ROC curve with AUC score
- `pr_curve.png` - Precision-Recall curve with average precision
- `model_comparison.png` - Grouped bar chart comparing SVM vs RF metrics
- `feature_importance.png` - Random Forest feature importance
- `feature_importance_bar.png` - Feature importance bar chart with values
- `feature_distribution.png` - Feature distributions for NORMAL vs PNEUMONIA classes

## Streamlit App Features

The web application (`app/app.py`) includes:

- **Model selection**: Choose between SVM and Random Forest
- **Visual pipeline**: Original, preprocessed, Otsu mask, K-Means segmentation
- **Prediction display**: Class badge, confidence score, feature table, bar chart
- **Explainability panel**: Per-prediction feature contribution analysis (toggleable)
- **Augmentation preview**: See how transforms affect the uploaded image (toggleable)

## Project Structure

```
medical-cv/
├── scripts/
|   └── download_dataset.py      # Smart dataset downloader
├── src/
|   ├── preprocessing.py         # Image loading, resizing, filtering, normalization
|   ├── segmentation.py          # Otsu thresholding, K-Means clustering
|   ├── feature_extraction.py    # GLCM texture + shape features
|   ├── classification.py        # SVM/RF training, evaluation, prediction
|   ├── pipeline.py              # End-to-end training orchestration
|   ├── augmentation.py          # Image augmentation pipeline
|   ├── balancing.py             # Class balancing utilities
|   ├── explainability.py        # Feature-level model explainability
|   └── visualization.py         # ROC, PR, confusion matrix, comparison plots
├── app/
|   └── app.py                   # Streamlit web application
├── models/                      # Trained models and visualization outputs
├── data/                        # Dataset (chest_xray/)
└── docs/
    └── TEAM_WORK_SPLIT.md       # 8-member team responsibility breakdown
```
