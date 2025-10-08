"""Builder function for HerdNetInference from configuration.

This module provides a factory function to build a HerdNetInference instance
from a configuration dictionary, following the same pattern as the trainer builder.
"""

from typing import Dict, Any
from omegaconf import OmegaConf

from animaldet.engine.registry import INFERENCE_BUILDERS
from animaldet.experiments.herdnet.inference import HerdNetInference
from animaldet.experiments.herdnet.adapters.model import build_model
from animaldet.experiments.herdnet.adapters.config import HerdNetExperimentConfig


@INFERENCE_BUILDERS.register("HerdNetInference")
def build_herdnet_inference(cfg: Dict[str, Any]) -> HerdNetInference:
    """Build HerdNetInference from configuration.

    This function handles all HerdNet-specific setup for inference including:
    - Model instantiation
    - Checkpoint loading
    - Device configuration

    Args:
        cfg: Configuration dictionary from YAML

    Returns:
        Configured HerdNetInference instance
    """
    # Parse config
    herdnet_cfg = OmegaConf.structured(HerdNetExperimentConfig)
    herdnet_cfg = OmegaConf.merge(herdnet_cfg, cfg)

    # Build model (without moving to device, inference class handles this)
    model = build_model(herdnet_cfg.model, device="cpu")

    # Get inference config
    inference_cfg = herdnet_cfg.get('inference', {})

    # Create inference instance
    inference = HerdNetInference(
        model=model,
        device=inference_cfg.get('device', 'cuda'),
        checkpoint_path=inference_cfg.get('checkpoint_path', None),
        down_ratio=herdnet_cfg.model.down_ratio,
        threshold=inference_cfg.get('threshold', 0.5)
    )

    return inference
