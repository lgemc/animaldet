"""Registries for engine components.

This module provides global registries for trainers and other engine components.
"""

from animaldet.utils.registry import Registry

# Global registry for trainer builder functions
TRAINER_BUILDERS = Registry("trainer_builders")

# Global registry for hooks (loggers, callbacks, etc.)
HOOKS = Registry("hooks")

__all__ = ["TRAINER_BUILDERS", "HOOKS"]