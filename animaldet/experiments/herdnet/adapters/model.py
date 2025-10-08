"""HerdNet model builders."""

import torch
from animaloc.models import HerdNet, LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss

from .config import ModelConfig
from animaldet.engine.registry import MODELS


def build_model(cfg: ModelConfig, device: str = "cuda") -> LossWrapper:
    """
    Build HerdNet model with loss wrapper.

    Args:
        cfg: Model configuration
        device: Device to place model on

    Returns:
        HerdNet model wrapped with losses
    """
    # Create base model
    model = HerdNet(
        num_classes=cfg.num_classes,
        down_ratio=cfg.down_ratio
    )

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.to(device)

    # Build losses
    losses = []
    for loss_cfg in cfg.losses:
        if loss_cfg.name == "focal_loss":
            loss_fn = FocalLoss(reduction="mean")
        elif loss_cfg.name == "ce_loss":
            weight = None
            if loss_cfg.weight is not None:
                weight = torch.Tensor(loss_cfg.weight)
                if device == "cuda" and torch.cuda.is_available():
                    weight = weight.cuda()
                else:
                    weight = weight.to(device)
            loss_fn = CrossEntropyLoss(reduction="mean", weight=weight)
        else:
            raise ValueError(f"Unknown loss: {loss_cfg.name}")

        losses.append({
            "loss": loss_fn,
            "idx": loss_cfg.idx,
            "idy": loss_cfg.idy,
            "lambda": loss_cfg.lambda_,
            "name": loss_cfg.name
        })

    # Wrap model with losses
    model = LossWrapper(model, losses=losses)

    return model


@MODELS.register("HerdNet")
def build_herdnet_model(cfg):
    """Registered HerdNet model builder for generic trainer.

    Args:
        cfg: Model configuration (should have structure of ModelConfig)

    Returns:
        HerdNet model wrapped with losses
    """
    return build_model(cfg)
