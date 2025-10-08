"""Optimizer builders for animaldet.

This module provides registered optimizer builders that can be instantiated
from configuration files using the OPTIMIZERS registry.
"""

import torch
from typing import Any, Dict, Iterable
from animaldet.engine.registry import OPTIMIZERS


@OPTIMIZERS.register("adam")
def build_adam_optimizer(params: Iterable, cfg: Dict[str, Any]) -> torch.optim.Adam:
    """Build Adam optimizer from configuration.

    Args:
        params: Model parameters
        cfg: Optimizer configuration dict with keys:
            - lr: Learning rate
            - weight_decay: Weight decay (L2 penalty)
            - betas: Coefficients for computing running averages (optional)
            - eps: Term added for numerical stability (optional)

    Returns:
        Adam optimizer instance
    """
    return torch.optim.Adam(
        params,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
        betas=cfg.get("betas", (0.9, 0.999)),
        eps=cfg.get("eps", 1e-8)
    )


@OPTIMIZERS.register("adamw")
def build_adamw_optimizer(params: Iterable, cfg: Dict[str, Any]) -> torch.optim.AdamW:
    """Build AdamW optimizer from configuration.

    Args:
        params: Model parameters
        cfg: Optimizer configuration dict with keys:
            - lr: Learning rate
            - weight_decay: Weight decay (L2 penalty)
            - betas: Coefficients for computing running averages (optional)
            - eps: Term added for numerical stability (optional)

    Returns:
        AdamW optimizer instance
    """
    return torch.optim.AdamW(
        params,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.01),
        betas=cfg.get("betas", (0.9, 0.999)),
        eps=cfg.get("eps", 1e-8)
    )


@OPTIMIZERS.register("sgd")
def build_sgd_optimizer(params: Iterable, cfg: Dict[str, Any]) -> torch.optim.SGD:
    """Build SGD optimizer from configuration.

    Args:
        params: Model parameters
        cfg: Optimizer configuration dict with keys:
            - lr: Learning rate
            - momentum: Momentum factor (optional)
            - weight_decay: Weight decay (L2 penalty) (optional)
            - dampening: Dampening for momentum (optional)
            - nesterov: Whether to use Nesterov momentum (optional)

    Returns:
        SGD optimizer instance
    """
    return torch.optim.SGD(
        params,
        lr=cfg.get("lr", 1e-3),
        momentum=cfg.get("momentum", 0.0),
        weight_decay=cfg.get("weight_decay", 0.0),
        dampening=cfg.get("dampening", 0.0),
        nesterov=cfg.get("nesterov", False)
    )