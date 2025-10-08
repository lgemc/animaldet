"""Configuration dataclasses for RF-DETR experiments."""

from dataclasses import dataclass, field
from typing import Optional, Literal
from omegaconf import MISSING


@dataclass
class ModelConfig:
    """RF-DETR model configuration."""
    # Model variant
    variant: Literal["nano", "small", "medium", "base", "large"] = "medium"

    # Model architecture params
    num_classes: int = MISSING
    resolution: int = 576  # Will be set based on variant

    # Encoder settings
    encoder: str = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 2

    # Decoder settings
    dec_layers: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300

    # Other architecture params
    projector_scale: list[str] = field(default_factory=lambda: ["P4"])
    out_feature_indexes: list[int] = field(default_factory=lambda: [3, 6, 9, 12])
    positional_encoding_size: int = 36
    group_detr: int = 13

    # Training settings
    two_stage: bool = True
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    gradient_checkpointing: bool = False

    # Loss settings
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0

    # Pretrained weights
    pretrain_weights: Optional[str] = None
    device: str = "cuda"


@dataclass
class DataConfig:
    """Dataset configuration for RF-DETR."""
    # Dataset paths
    dataset_file: Literal["coco", "o365", "roboflow"] = "roboflow"
    dataset_dir: str = MISSING
    train_annotation: Optional[str] = None  # For COCO format
    val_annotation: Optional[str] = None

    # Data augmentation
    multi_scale: bool = True
    expanded_scales: bool = True
    do_random_resize_via_padding: bool = False
    square_resize_div_64: bool = True

    # Class names
    class_names: Optional[list[str]] = None


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    weight_decay: float = 1e-4
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    drop_path: float = 0.0

    # EMA settings
    use_ema: bool = True
    ema_decay: float = 0.993
    ema_tau: int = 100


@dataclass
class TrainerConfig:
    """Training configuration."""
    name: str = "RFDETRTrainer"  # Trainer registry name

    # Training params
    epochs: int = 100
    batch_size: int = 4
    grad_accum_steps: int = 4
    num_workers: int = 2

    # Learning rate schedule
    warmup_epochs: float = 0.0
    lr_drop: int = 100

    # Checkpointing
    checkpoint_interval: int = 10
    work_dir: str = "./output"

    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = False

    # Logging
    tensorboard: bool = True
    wandb: bool = False
    project: Optional[str] = None
    run: Optional[str] = None

    # Evaluation
    run_test: bool = True


@dataclass
class EvaluatorConfig:
    """Evaluation configuration."""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    eval_on_ema: bool = True


@dataclass
class RFDETRExperimentConfig:
    """Complete RF-DETR experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    seed: int = 9292
