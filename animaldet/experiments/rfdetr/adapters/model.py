"""RF-DETR model builders."""

import torch
from typing import Dict, Any
import sys
from pathlib import Path

# Add rf-detr to path
rfdetr_path = Path("/home/lmanrique/Do/rf-detr")
if str(rfdetr_path) not in sys.path:
    sys.path.insert(0, str(rfdetr_path))

from rfdetr.detr import (
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRBase,
    RFDETRLarge
)

from .config import ModelConfig
from animaldet.engine.registry import MODELS


MODEL_VARIANTS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "medium": RFDETRMedium,
    "base": RFDETRBase,
    "large": RFDETRLarge
}


def build_model(cfg: ModelConfig, device: str = "cuda") -> torch.nn.Module:
    """
    Build RF-DETR model from configuration.

    Args:
        cfg: Model configuration
        device: Device to place model on

    Returns:
        RF-DETR model instance (PyTorch nn.Module)
    """
    # Get the appropriate model class
    model_class = MODEL_VARIANTS.get(cfg.variant.lower())
    if model_class is None:
        raise ValueError(
            f"Unknown model variant: {cfg.variant}. "
            f"Available: {list(MODEL_VARIANTS.keys())}"
        )

    # Build model kwargs from config
    model_kwargs: Dict[str, Any] = {
        "num_classes": cfg.num_classes,
        "patch_size": cfg.patch_size,
        "num_windows": cfg.num_windows,
        "hidden_dim": cfg.hidden_dim,
        "dec_layers": cfg.dec_layers,
        "sa_nheads": cfg.sa_nheads,
        "ca_nheads": cfg.ca_nheads,
        "dec_n_points": cfg.dec_n_points,
        "num_queries": cfg.num_queries,
        "num_select": cfg.num_select,
        "projector_scale": cfg.projector_scale,
        "out_feature_indexes": cfg.out_feature_indexes,
        "positional_encoding_size": cfg.positional_encoding_size,
        "resolution": cfg.resolution,
    }

    # Add optional pretrained weights
    if cfg.pretrain_weights:
        model_kwargs["pretrain_weights"] = cfg.pretrain_weights

    # Create model instance (this is a wrapper: RFDETR class)
    rfdetr_wrapper = model_class(**model_kwargs)

    # Reinitialize detection head with correct number of classes
    # The RF-DETR wrapper may have loaded pretrained weights with different num_classes
    # We need to ensure the head matches our dataset's num_classes
    print(f"Restarting number of classes to {cfg.num_classes}")
    rfdetr_wrapper.model.reinitialize_detection_head(cfg.num_classes+1)  # +1 for background class

    # Extract the actual PyTorch model from the nested wrappers
    # rfdetr_wrapper.model is a Model instance
    # rfdetr_wrapper.model.model is the actual PyTorch LWDETR module
    # Note: The model is already moved to the correct device in Model.__init__
    model = rfdetr_wrapper.model.model

    return model


def get_model_config_params(cfg: ModelConfig) -> Dict[str, Any]:
    """
    Extract RF-DETR specific model parameters from config.

    This is useful for passing to rfdetr's training functions.

    Args:
        cfg: Model configuration

    Returns:
        Dictionary of model parameters
    """
    return {
        "encoder": cfg.encoder,
        "hidden_dim": cfg.hidden_dim,
        "patch_size": cfg.patch_size,
        "num_windows": cfg.num_windows,
        "dec_layers": cfg.dec_layers,
        "sa_nheads": cfg.sa_nheads,
        "ca_nheads": cfg.ca_nheads,
        "dec_n_points": cfg.dec_n_points,
        "num_queries": cfg.num_queries,
        "num_select": cfg.num_select,
        "projector_scale": cfg.projector_scale,
        "out_feature_indexes": cfg.out_feature_indexes,
        "positional_encoding_size": cfg.positional_encoding_size,
        "group_detr": cfg.group_detr,
        "two_stage": cfg.two_stage,
        "bbox_reparam": cfg.bbox_reparam,
        "lite_refpoint_refine": cfg.lite_refpoint_refine,
        "layer_norm": cfg.layer_norm,
        "amp": cfg.amp,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "ia_bce_loss": cfg.ia_bce_loss,
        "cls_loss_coef": cfg.cls_loss_coef,
        "device": cfg.device,
        "resolution": cfg.resolution,
        "num_classes": cfg.num_classes,
    }


@MODELS.register("RFDETR")
def build_rfdetr_model(cfg):
    """Registered RF-DETR model builder for generic trainer.

    Args:
        cfg: Model configuration (should have structure of ModelConfig)

    Returns:
        RF-DETR model instance
    """
    return build_model(cfg)