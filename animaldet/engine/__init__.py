"""Engine module for training and evaluation."""

from animaldet.engine.registry import TRAINER_BUILDERS, INFERENCE_BUILDERS, HOOKS
from animaldet.engine.loggers import ConsoleLogger
from animaldet.engine.metric_hooks import F1MetricHook
from animaldet.engine.inference import BaseInference

# Import integrations to trigger their registration decorators
# TensorBoardLogger and WandbLogger will auto-register via @HOOKS.register()
from animaldet.utils.integrations import TensorBoardLogger, WandbLogger  # noqa: F401

# Register core engine hooks
HOOKS.register("console")(ConsoleLogger)
HOOKS.register("f1_metric")(F1MetricHook)

__all__ = [
    "TRAINER_BUILDERS",
    "INFERENCE_BUILDERS",
    "HOOKS",
    "BaseInference",
    "ConsoleLogger",
    "TensorBoardLogger",
    "WandbLogger",
    "F1MetricHook",
]