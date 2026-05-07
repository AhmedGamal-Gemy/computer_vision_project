# Medical Image Analysis Assistant

A computer vision pipeline for preprocessing, segmenting, and classifying chest X-ray images to assist in medical diagnosis.

## Requirements

- Python 3.10+
- uv package manager
- Kaggle account (for dataset download)

## Setup Steps

1. Clone the repository and enter the project directory:
   ```bash
   cd medical-cv
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Download the dataset (auto-checks if already exists):
   ```bash
   uv run python scripts/download_dataset.py
   ```

4. Train the models:
   ```bash
   uv run python src/pipeline.py
   ```

5. Launch the web application:
   ```bash
   uv run streamlit run app/app.py
   ```

## Documentation

- **Learning Guide**: [`docs/LEARNING_GUIDE.md`](docs/LEARNING_GUIDE.md) - Comprehensive guide covering all 8 pipeline stages with detailed explanations of concepts, implementation, and design decisions
- **Team Work Split**: [`docs/TEAM_WORK_SPLIT.md`](docs/TEAM_WORK_SPLIT.md) - 8-member team responsibility breakdown with dependency chain

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

## Configuration

All pipeline parameters are configurable via `config.yaml` - no hardcoded constants:

- **Preprocessing**: Image size, filter kernels, normalization settings
- **Segmentation**: Otsu thresholding, K-Means clustering parameters
- **Feature Extraction**: GLCM distances, angles, levels
- **Classification**: SVM kernel/C/gamma, Random Forest estimators
- **Augmentation**: Rotation angles, brightness/contrast ranges, flip probabilities
- **Balancing**: Strategy selection (oversample/undersample/SMOTE), noise parameters
- **Explainability**: Permutation repeats, influence thresholds
- **Visualization**: DPI, figure sizes, color schemes

Configuration uses a two-layer system:
1. **Python dataclasses** (`src/config.py`) define defaults with type safety
2. **YAML file** (`config.yaml`) overrides any default without code changes

Example override in `config.yaml`:
```yaml
classification:
  svm_c: 15.0
  rf_n_estimators: 300

augmentation:
  max_rotation_angle: 20.0
```

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

The web application (`app/app.py`) includes two tabs:

### Tab 1: Analysis
- **Model selection**: Choose between SVM and Random Forest
- **Visual pipeline**: Original, preprocessed, Otsu mask, K-Means segmentation
- **Prediction display**: Class badge, confidence score, feature table, bar chart
- **Explainability panel**: Per-prediction feature contribution analysis (toggleable)
- **Augmentation preview**: See how transforms affect the uploaded image (toggleable)

### Tab 2: Model Report
- **Performance metrics dashboard**: Accuracy, precision, recall, F1 for both models
- **Dataset statistics**: Test sample count, class distribution, feature count
- **Visualization gallery**: All generated plots from training
- **Feature importance table**: Sorted Random Forest importances with 4 decimal precision

## Project Structure

```
medical-cv/
├── scripts/
│   └── download_dataset.py      # Smart dataset downloader with existence check
├── src/
│   ├── config.py                # Centralized configuration system
│   ├── preprocessing.py         # Image loading, resizing, filtering, normalization
│   ├── segmentation.py          # Otsu thresholding, K-Means clustering
│   ├── feature_extraction.py    # GLCM texture + shape features
│   ├── classification.py        # SVM/RF training, evaluation, prediction
│   ├── pipeline.py              # End-to-end training orchestration
│   ├── augmentation.py          # Image augmentation pipeline
│   ├── balancing.py             # Class balancing utilities
│   ├── explainability.py        # Feature-level model explainability
│   └── visualization.py         # ROC, PR, confusion matrix, comparison plots
├── app/
│   └── app.py                   # Streamlit web application (Analysis + Model Report tabs)
├── models/                      # Trained models and visualization outputs
├── data/                        # Dataset (chest_xray/)
├── docs/
│   ├── LEARNING_GUIDE.md        # Comprehensive pipeline learning guide
│   └── TEAM_WORK_SPLIT.md       # 8-member team responsibility breakdown
├── config.yaml                  # All configurable parameters
├── pyproject.toml               # Project dependencies
└── README.md                    # This file
```
