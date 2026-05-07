"""Centralized configuration for the Medical Image Analysis Assistant.

All pipeline parameters, model hyperparameters, visualization settings,
and UI defaults are defined here as dataclass defaults. Override any value
by creating a `config.yaml` file in the project root.

Usage:
    from src.config import get_config
    cfg = get_config()  # loads defaults + config.yaml if present
    cfg.classification.svm_c  # access any parameter
"""

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# =============================================================================
# Sub-configuration dataclasses
# =============================================================================

@dataclass
class PreprocessingConfig:
    """Image preprocessing parameters."""
    target_size: tuple[int, int] = (256, 256)
    gaussian_kernel: tuple[int, int] = (5, 5)
    gaussian_sigma: float = 1.0
    median_kernel: int = 5

@dataclass
class SegmentationConfig:
    """Image segmentation parameters."""
    kmeans_k: int = 3
    morph_kernel_size: tuple[int, int] = (3, 3)
    morph_iterations: int = 2
    # WHY: K-means convergence criteria control how long clustering runs.
    # Medical images need enough iterations for stable tissue separation,
    # but not so many that runtime becomes prohibitive for large datasets.
    kmeans_iterations: int = 10
    kmeans_epsilon: float = 1.0
    kmeans_attempts: int = 10

@dataclass
class FeatureExtractionConfig:
    """Feature extraction parameters."""
    glcm_distances: list[int] = field(default_factory=lambda: [1])
    glcm_angles: list[int] = field(default_factory=lambda: [0])
    glcm_levels: int = 256

@dataclass
class ClassificationConfig:
    """ML classifier hyperparameters."""
    # WHY: RBF kernel handles non-linear feature separation common in
    # medical imaging where decision boundaries are complex.
    svm_kernel: str = "rbf"
    svm_c: float = 10.0
    svm_gamma: str = "scale"
    svm_random_state: int = 42
    rf_n_estimators: int = 200
    rf_random_state: int = 42

@dataclass
class AugmentationConfig:
    """Image augmentation parameters - medical-aware constraints."""
    max_rotation_angle: float = 15.0
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    noise_std_range: tuple[float, float] = (0.01, 0.05)
    zoom_range: tuple[float, float] = (0.9, 1.1)
    shear_range: tuple[float, float] = (-5.0, 5.0)
    translation_range: tuple[float, float] = (0.1, 0.1)
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0

@dataclass
class BalancingConfig:
    """Class balancing parameters."""
    strategy: str = "oversample"
    apply_balancing: bool = True
    oversample_noise_std: float = 0.01

@dataclass
class VisualizationConfig:
    """Plotting and visualization parameters."""
    figure_dpi: int = 150
    figure_size_standard: tuple[float, float] = (10, 6)
    figure_size_wide: tuple[float, float] = (14, 6)
    figure_size_square: tuple[float, float] = (8, 8)
    # WHY: Pipeline-specific figure sizes differ from the standard sizes
    # because side-by-side confusion matrices need a wide but short layout,
    # and feature importance bars need a compact vertical layout.
    figure_size_confusion: tuple[float, float] = (10, 4)
    figure_size_importance: tuple[float, float] = (8, 5)
    cmap_heatmap: str = "Blues"
    cmap_confusion: str = "viridis"
    color_normal: str = "#2ecc71"
    color_pneumonia: str = "#e74c3c"
    bar_color: str = "#3498db"

@dataclass
class ExplainabilityConfig:
    """Model explainability parameters."""
    permutation_n_repeats: int = 10
    high_influence_threshold: float = 0.3
    moderate_influence_threshold: float = 0.1

@dataclass
class AppConfig:
    """Streamlit application parameters."""
    accepted_file_types: list[str] = field(default_factory=lambda: ["jpg", "jpeg", "png"])
    show_explainability_default: bool = True
    show_augmentation_default: bool = False
    page_title: str = "Medical Image Analysis"
    page_icon: str = "🫁"

# =============================================================================
# Root configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Root configuration containing all sub-configs."""
    data_dir: str = "data/chest_xray"
    models_dir: str = "models"
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    balancing: BalancingConfig = field(default_factory=BalancingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    app: AppConfig = field(default_factory=AppConfig)


# =============================================================================
# Config loading
# =============================================================================

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dataclass_to_dict(dc) -> dict:
    """Convert dataclass to nested dict, handling tuples as lists."""
    result = {}
    for f in dc.__dataclass_fields__:
        val = getattr(dc, f)
        if hasattr(val, '__dataclass_fields__'):
            result[f] = _dataclass_to_dict(val)
        elif isinstance(val, tuple):
            result[f] = list(val)
        else:
            result[f] = val
    return result


def _dict_to_dataclass(dc_class, data: dict):
    """Convert nested dict back to dataclass, handling list->tuple conversion."""
    field_types = {f.name: f.type for f in dc_class.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key not in field_types:
            continue
        ft = field_types[key]
        # Check if it's a nested dataclass
        if hasattr(ft, '__dataclass_fields__'):
            kwargs[key] = _dict_to_dataclass(ft, value)
        # Check if it expects a tuple
        elif 'tuple' in str(ft).lower():
            kwargs[key] = tuple(value) if isinstance(value, list) else value
        else:
            kwargs[key] = value
    return dc_class(**kwargs)


def load_config(config_path: Path | str | None = None) -> PipelineConfig:
    """Load configuration from defaults, optionally overridden by YAML file.

    WHY: Centralized configuration allows tuning pipeline behavior without
    modifying source code. Medical teams can adjust parameters for different
    datasets, hardware constraints, or clinical requirements.

    Args:
        config_path: Path to YAML config file. If None, looks for
            'config.yaml' in the project root (parent of src/).

    Returns:
        PipelineConfig with defaults merged with YAML overrides.
    """
    # Start with default config
    default_cfg = PipelineConfig()
    default_dict = _dataclass_to_dict(default_cfg)

    # Load YAML overrides if file exists
    if config_path is None:
        # Auto-detect: project root is parent of src/
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_dict = yaml.safe_load(f) or {}
        merged = _deep_merge(default_dict, yaml_dict)
    else:
        merged = default_dict

    return _dict_to_dataclass(PipelineConfig, merged)


def save_config_template(output_path: Path | str = "config.example.yaml") -> None:
    """Save a YAML template with all default values and documentation.

    Args:
        output_path: Where to write the template file.
    """
    cfg = PipelineConfig()
    template = _dataclass_to_dict(cfg)
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)


# Module-level singleton - lazy loaded on first access
_config: PipelineConfig | None = None


def get_config() -> PipelineConfig:
    """Get the global configuration singleton.

    Returns:
        PipelineConfig instance (cached after first call).
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the global config singleton (useful for testing)."""
    global _config
    _config = None
