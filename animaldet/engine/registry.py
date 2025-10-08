"""Registries for engine components.

This module provides global registries for trainers and other engine components.
"""

from animaldet.utils.registry import Registry

# Global registry for trainers
TRAINERS = Registry("trainers")

__all__ = ["TRAINERS"]