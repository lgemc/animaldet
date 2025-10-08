"""RF-DETR experiment adapters."""

from .config import (
    RFDETRExperimentConfig,
    ModelConfig,
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    EvaluatorConfig,
)
from .model import build_model, get_model_config_params
from .dataset import (
    build_train_dataset,
    build_val_dataset,
    build_test_dataset,
    build_dataloaders,
)
from .trainer import (
    build_optimizer,
    build_ema_model,
    build_lr_scheduler,
    build_criterion_and_postprocessors,
    prepare_training_args,
)
from .evaluator import build_evaluator, evaluate

__all__ = [
    # Config
    "RFDETRExperimentConfig",
    "ModelConfig",
    "DataConfig",
    "OptimizerConfig",
    "TrainerConfig",
    "EvaluatorConfig",
    # Model builders
    "build_model",
    "get_model_config_params",
    # Dataset builders
    "build_train_dataset",
    "build_val_dataset",
    "build_test_dataset",
    "build_dataloaders",
    # Training utilities
    "build_optimizer",
    "build_ema_model",
    "build_lr_scheduler",
    "build_criterion_and_postprocessors",
    "prepare_training_args",
    # Evaluation
    "build_evaluator",
    "evaluate",
]