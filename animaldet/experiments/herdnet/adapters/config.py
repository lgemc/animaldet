"""Configuration dataclasses for HerdNet experiments."""

from dataclasses import dataclass, field
from typing import Any, Optional
from omegaconf import MISSING

from animaldet.config import ExperimentMetadata


@dataclass
class LossConfig:
    """Configuration for a single loss function."""
    name: str = MISSING  # 'focal_loss' or 'ce_loss'
    idx: int = MISSING  # Output index
    idy: int = MISSING  # Target index
    lambda_: float = 1.0  # Loss weight
    weight: Optional[list[float]] = None  # Class weights (for CE loss)


@dataclass
class ModelConfig:
    """HerdNet model configuration."""
    num_classes: int = 7
    down_ratio: int = 2
    losses: list[LossConfig] = field(default_factory=lambda: [
        LossConfig(name='focal_loss', idx=0, idy=0, lambda_=1.0),
        LossConfig(name='ce_loss', idx=1, idy=1, lambda_=1.0,
                   weight=[0.1, 1.0, 2.0, 1.0, 6.0, 12.0, 1.0])
    ])


@dataclass
class DataConfig:
    """Dataset configuration for HerdNet."""
    patch_size: int = 512
    train_csv: str = MISSING
    train_root: str = MISSING
    val_csv: str = MISSING
    val_root: str = MISSING
    test_csv: Optional[str] = None
    test_root: Optional[str] = None

    # Augmentation probabilities
    vertical_flip: float = 0.5
    horizontal_flip: float = 0.5
    rotate90: float = 0.5
    brightness_contrast: float = 0.2
    blur: float = 0.2

    # Transform params
    radius: int = 2  # For PointsToMask
    mask_down_ratio: int = 32  # patch_size // 16


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    name: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-3


@dataclass
class TrainerConfig:
    """Training configuration."""
    name: str = "HerdNetTrainer"  # Trainer registry name
    num_epochs: int = 100
    batch_size: int = 4
    warmup_iters: int = 100
    checkpoints: str = "best"  # 'best', 'all', or 'last'
    select: str = "max"  # 'max' or 'min'
    validate_on: str = "f1_score"
    work_dir: str = "./output"


@dataclass
class EvaluatorConfig:
    """Evaluation configuration."""
    metrics_radius: int = 20  # For PointsMetrics
    stitcher_overlap: int = 160
    stitcher_reduction: str = "mean"


@dataclass
class HerdNetExperimentConfig:
    """Complete HerdNet experiment configuration."""
    experiment: ExperimentMetadata = field(default_factory=ExperimentMetadata)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    integrations: dict[str, Any] = field(default_factory=dict)
    seed: int = 9292
