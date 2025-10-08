"""Registries for engine components.

This module provides global registries for trainers and other engine components.
"""

from animaldet.utils.registry import Registry

# Global registry for trainer builder functions
TRAINER_BUILDERS = Registry("trainer_builders")

# Global registry for inference builder functions
INFERENCE_BUILDERS = Registry("inference_builders")

# Global registry for hooks (loggers, callbacks, etc.)
HOOKS = Registry("hooks")

# Global registry for models
MODELS = Registry("models")

# Global registry for optimizers
OPTIMIZERS = Registry("optimizers")

# Global registry for datasets
DATASETS = Registry("datasets")

__all__ = ["TRAINER_BUILDERS", "INFERENCE_BUILDERS", "HOOKS", "MODELS", "OPTIMIZERS", "DATASETS"]