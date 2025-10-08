"""RF-DETR experiment module.

This module provides a complete interface for running RF-DETR experiments
with the animaldet framework, including dataset builders, model builders,
and a PyTorch Lightning-style trainer with hook support.
"""

from .adapters.config import (
    RFDETRExperimentConfig,
    ModelConfig,
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    EvaluatorConfig,
)
from .adapters.model import build_model, get_model_config_params
from .adapters.dataset import (
    build_train_dataset,
    build_val_dataset,
    build_test_dataset,
    build_dataloaders,
)
from .adapters.trainer import (
    build_optimizer,
    build_ema_model,
    build_lr_scheduler,
    build_criterion_and_postprocessors,
    prepare_training_args,
)
from .adapters.evaluator import build_evaluator, evaluate
from .trainer import RFDETRTrainer
from .builder import build_rfdetr_trainer

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
    # Trainer
    "RFDETRTrainer",
    "build_rfdetr_trainer",
]
