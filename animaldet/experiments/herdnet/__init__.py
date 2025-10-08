"""HerdNet experiment module.

This module provides a complete interface for running HerdNet experiments
with the animaldet framework, including dataset builders, model builders,
and a PyTorch Lightning-style trainer with hook support.
"""

from .adapters.config import (
    HerdNetExperimentConfig,
    ModelConfig,
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    EvaluatorConfig,
    LossConfig
)
from .adapters.model import build_model
from .adapters.dataset import (
    build_train_dataset,
    build_val_dataset,
    build_test_dataset,
    build_dataloaders
)
from .adapters.trainer import build_trainer, build_optimizer
from .adapters.evaluator import build_evaluator, evaluate
from .trainer import HerdNetTrainer

__all__ = [
    # Config
    "HerdNetExperimentConfig",
    "ModelConfig",
    "DataConfig",
    "OptimizerConfig",
    "TrainerConfig",
    "EvaluatorConfig",
    "LossConfig",
    # Builders
    "build_model",
    "build_train_dataset",
    "build_val_dataset",
    "build_test_dataset",
    "build_dataloaders",
    "build_trainer",
    "build_optimizer",
    "build_evaluator",
    # Utilities
    "evaluate",
    # Trainer
    "HerdNetTrainer",
]