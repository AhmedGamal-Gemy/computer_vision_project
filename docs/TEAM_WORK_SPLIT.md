# Team Work Split - Medical Image Analysis Assistant

## Project Overview

The Medical Image Analysis Assistant is a Python-based computer vision pipeline that analyzes chest X-ray images to detect pneumonia. The system processes medical images through preprocessing, segmentation, feature extraction, and machine learning classification to provide diagnostic assistance.

**Tech Stack:**
- Python 3.12
- uv (package manager)
- OpenCV (image processing)
- scikit-image (image analysis)
- scikit-learn (machine learning)
- Streamlit (web interface)

**Dataset:** Chest X-Ray Pneumonia from Kaggle (5,856 images)

**Models:** SVM (RBF kernel) and Random Forest (200 estimators)

---

## Team Structure (8 Members)

### Member 1: Data Pipeline Lead

**Assigned Files:**
- `scripts/download_dataset.py` - Dataset download and verification
- `src/preprocessing.py` - Image preprocessing (load, resize, blur, normalize)
- `src/__init__.py` - Package initialization

**Responsibilities:**
- Data loading and path handling
- Image I/O operations
- Dataset verification and integrity checks
- Image preprocessing pipeline (resizing, blurring, normalization)
- Package structure and exports

**Deliverables:**
- Working dataset download script
- Robust preprocessing functions with error handling
- Clean package initialization

---

### Member 2: Segmentation Specialist

**Assigned Files:**
- `src/segmentation.py` - Otsu and K-Means segmentation

**Responsibilities:**
- Image masking and morphological operations
- Otsu thresholding implementation
- K-Means clustering for segmentation
- Mask generation and refinement
- All segmentation logic

**Deliverables:**
- Segmentation functions that accept preprocessed images
- Binary masks for region analysis
- Clean, well-documented segmentation code

---

### Member 3: Feature Engineering Lead

**Assigned Files:**
- `src/feature_extraction.py` - GLCM texture features + shape features

**Responsibilities:**
- GLCM (Gray Level Co-occurrence Matrix) texture feature computation
- Shape feature extraction using regionprops
- Feature matrix building with progress tracking
- All feature extraction logic

**Deliverables:**
- Feature extraction pipeline that accepts segmented images
- Standardized feature matrix output
- Progress tracking for batch processing

**Dependencies:**
- Member 1 (preprocessing needed for feature extraction)
- Member 2 (segmentation needed for feature extraction)

---

### Member 4: ML Model Engineer

**Assigned Files:**
- `src/classification.py` - SVM and Random Forest training

**Responsibilities:**
- SVM classifier training (RBF kernel)
- Random Forest classifier training (200 estimators)
- Model evaluation metrics (accuracy, precision, recall, F1-score)
- Model serialization (save/load with joblib)
- All ML logic, model training, and evaluation

**Deliverables:**
- Trained SVM and Random Forest models
- Evaluation metrics reporting
- Model save/load functionality
- Clean separation of training and prediction logic

**Dependencies:**
- Member 3 (features needed for classification)

---

### Member 5: Pipeline Orchestrator

**Assigned Files:**
- `src/pipeline.py` - Main training pipeline

**Responsibilities:**
- End-to-end training workflow
- Data splitting (train/test)
- Feature matrix building coordination
- Comparison table generation (SVM vs Random Forest)
- Visualization (confusion matrix, feature importance)
- Pipeline execution and error handling

**Deliverables:**
- Complete training pipeline script
- Model comparison outputs
- Visualization plots
- Pipeline logging and status reporting

**Dependencies:**
- Member 4 (models needed for pipeline)

---

### Member 6: Frontend/UI Developer

**Assigned Files:**
- `app/app.py` - Streamlit web application

**Responsibilities:**
- Streamlit web interface design
- UI layout and navigation
- File upload component
- Prediction display and results visualization
- Model caching with `st.cache_resource`
- All UI/UX, Streamlit components, user interaction

**Deliverables:**
- Functional Streamlit application
- Intuitive user interface
- Real-time prediction display
- Responsive design

**Dependencies:**
- Member 4 (models needed for app)
- Member 5 (pipeline outputs used by app)

---

### Member 7: Visualization & Explainability Lead

**Assigned Files:**
- `src/explainability.py` - Grad-CAM, heatmaps, saliency maps
- `src/visualization.py` - Pipeline-wide plotting utilities

**Responsibilities:**
- Grad-CAM implementation for model interpretability
- Heatmap overlay generation on original X-ray images
- Saliency map computation showing pixel-level importance
- Visualization utilities shared across pipeline (confusion matrix, feature importance, ROC curves)
- Per-image visual explanations of *why* the model predicted pneumonia vs normal
- Medical-grade output formatting (clear, interpretable visuals for clinical context)

**Deliverables:**
- `generate_gradcam(model, image)` → heatmap overlay on original X-ray
- `plot_saliency_map(image, predictions)` → pixel importance visualization
- `plot_roc_curve(y_true, y_scores)` → ROC curve with AUC annotation
- `plot_precision_recall_curve(y_true, y_scores)` → PR curve
- `plot_feature_distribution(features, labels)` → feature space visualization
- All visualization functions accept `pathlib.Path` outputs, return `matplotlib.figure.Figure`

**Dependencies:**
- Member 4 (trained models needed for Grad-CAM)
- Member 5 (pipeline integration for visualization generation)

---

### Member 8: Data Augmentation & Enhancement Specialist

**Assigned Files:**
- `src/augmentation.py` - Image augmentation pipeline
- `src/balancing.py` - Class balancing utilities

**Responsibilities:**
- Geometric augmentations: rotation, horizontal/vertical flip, zoom, shear, translation
- Intensity augmentations: brightness adjustment, contrast variation, Gaussian noise injection
- Medical-aware augmentations (preserve anatomical validity - no extreme rotations that flip lung orientation)
- Class balancing: oversampling minority class, SMOTE for feature space, undersampling majority class
- On-the-fly augmentation during training vs pre-computed augmentation for dataset expansion
- Augmentation configuration via constants (angles, ranges, probabilities)

**Deliverables:**
- `augment_image(image, config)` → augmented image with configurable transforms
- `augment_batch(images, labels, config)` → batch augmentation with label preservation
- `balance_dataset(X, y, strategy)` → balanced feature matrix and labels
- `generate_augmented_dataset(input_dir, output_dir, multiplier)` → expanded dataset on disk
- All augmentation functions accept `numpy.ndarray`, return `numpy.ndarray`
- Constants for augmentation ranges (e.g., `MAX_ROTATION_ANGLE = 15`, `BRIGHTNESS_RANGE = (0.8, 1.2)`)

**Dependencies:**
- Member 1 (preprocessing pipeline - augmentation happens after loading, before segmentation)

---

## SOLID Principles Checklist

For each module, verify the following before marking complete:

### Single Responsibility Principle (S)
- [ ] Each module has one reason to change
- [ ] Functions perform a single, well-defined task
- [ ] No module handles unrelated concerns

### Open/Closed Principle (O)
- [ ] Modules are open for extension
- [ ] Modules are closed for modification
- [ ] New functionality can be added without changing existing code

### Liskov Substitution Principle (L)
- [ ] Subtypes are substitutable for their base types
- [ ] Inheritance hierarchies are properly designed
- [ ] No unexpected behavior when substituting subclasses

### Interface Segregation Principle (I)
- [ ] Many client-specific interfaces are preferred over one general-purpose interface
- [ ] No module depends on methods it does not use
- [ ] Interfaces are focused and cohesive

### Dependency Inversion Principle (D)
- [ ] High-level modules do not depend on low-level modules
- [ ] Both depend on abstractions
- [ ] Abstractions do not depend on details
- [ ] Details depend on abstractions

### Module Review Status

| Module | S | O | L | I | D | Reviewer |
|--------|---|---|---|---|---|----------|
| `src/preprocessing.py` | [ ] | [ ] | [ ] | [ ] | [ ] | TBD |
| `src/segmentation.py` | [ ] | [ ] | [ ] | [ ] | [ ] | TBD |
| `src/feature_extraction.py` | [ ] | [ ] | [ ] | [ ] | [ ] | TBD |
| `src/classification.py` | [ ] | [ ] | [ ] | [ ] | [ ] | TBD |
| `src/pipeline.py` | [ ] | [ ] | [ ] | [ ] | [ ] | TBD |
| `app/app.py` | [ ] | [ ] | [ ] | [ ] | [ ] | TBD |

---

## Development Workflow

1. **Module Development**
   - Each member works on their assigned module
   - Follow SOLID principles checklist
   - Write clean, documented code

2. **Local Testing**
   - Use `uv run python` for all Python commands
   - Test individual modules in isolation
   - Run `uv run python src/pipeline.py` to test full pipeline
   - Run `uv run streamlit run app/app.py` to test UI

3. **Code Review**
   - Each pull request reviewed by at least 2 other members
   - Verify SOLID principles checklist
   - Check for code quality and documentation

4. **Integration**
   - Merge only after pipeline runs successfully
   - Verify all tests pass
   - Confirm documentation is updated

5. **Final Verification**
   - Run complete pipeline end-to-end
   - Test Streamlit application
   - Review all documentation

---

## Dependencies Between Members

```
Member 1 (Data Pipeline)
    |
    v
Member 2 (Segmentation) -----> Member 3 (Feature Engineering)
    |                              |
    |                              v
    |                         Member 4 (ML Models)
    |                              |
    |                              v
    |                         Member 5 (Pipeline)
    |                              |
    |                              v
    +------------------------> Member 6 (Frontend)

Member 8 (Augmentation) -----> Member 1 (feeds into preprocessing)
    |
    v
Member 4 (ML Models) ---------> Member 7 (Explainability)
```

**Dependency Details:**

| From | To | Reason |
|------|-----|--------|
| Member 1 | Member 3 | Preprocessing output needed for feature extraction |
| Member 2 | Member 3 | Segmentation masks needed for feature extraction |
| Member 3 | Member 4 | Feature matrix needed for model training |
| Member 4 | Member 5 | Trained models needed for pipeline |
| Member 4 | Member 6 | Trained models needed for app predictions |
| Member 4 | Member 7 | Trained models needed for Grad-CAM/explainability |
| Member 5 | Member 6 | Pipeline outputs used by app |
| Member 5 | Member 7 | Pipeline integration for visualization generation |
| Member 8 | Member 1 | Augmented images feed into preprocessing pipeline |
| Member 8 | Member 4 | Balanced dataset improves model training |

---

## File Structure Reference

```
medical-cv/
├── scripts/
│   └── download_dataset.py      <- Member 1
├── src/
│   ├── __init__.py              <- Member 1
│   ├── preprocessing.py         <- Member 1
│   ├── segmentation.py          <- Member 2
│   ├── feature_extraction.py    <- Member 3
│   ├── classification.py        <- Member 4
│   ├── pipeline.py              <- Member 5
│   ├── explainability.py        <- Member 7
│   ├── visualization.py         <- Member 7
│   ├── augmentation.py          <- Member 8
│   └── balancing.py             <- Member 8
├── app/
│   └── app.py                   <- Member 6
└── docs/
    └── TEAM_WORK_SPLIT.md
```

---

## Communication Guidelines

- Use pull requests for all code changes
- Tag relevant members in PR reviews based on dependencies
- Update this document when responsibilities change
- Member 7 coordinates visualization standards and explainability output format
- Member 8 coordinates augmentation configuration and dataset versioning

---

*Last updated: 2026-05-07*
