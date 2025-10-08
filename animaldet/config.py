"""Shared configuration dataclasses for animaldet experiments.

This module provides base configuration classes that are used across
different experiments and integrations.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentMetadata:
    """Experiment metadata configuration.

    Used for tracking experiment information across integrations
    like WandB, TensorBoard, etc.
    """
    name: str = "experiment"
    tags: list[str] = field(default_factory=list)
