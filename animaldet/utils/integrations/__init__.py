"""Integration utilities for external services like wandb, tensorboard, etc."""

from animaldet.utils.integrations.tensorboard import TensorBoardLogger
from animaldet.utils.integrations.wandb import WandbLogger

__all__ = ["WandbLogger", "TensorBoardLogger"]
