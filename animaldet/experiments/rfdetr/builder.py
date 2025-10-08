"""Builder function for RFDETRTrainer from configuration.

This module provides a factory function to build a RFDETRTrainer instance
from a configuration dictionary, handling all RF-DETR-specific setup.
"""

import torch
from typing import Dict, Any
from omegaconf import OmegaConf
from pathlib import Path

from animaldet.engine.registry import TRAINER_BUILDERS
from animaldet.experiments.rfdetr.trainer import RFDETRTrainer
from animaldet.experiments.rfdetr.adapters.config import RFDETRExperimentConfig
from animaldet.experiments.rfdetr.adapters.model import build_model
from animaldet.experiments.rfdetr.adapters.dataset import build_dataloaders
from animaldet.experiments.rfdetr.adapters.trainer import (
    build_optimizer,
    build_ema_model,
    build_lr_scheduler,
    build_criterion_and_postprocessors,
    prepare_training_args,
)


@TRAINER_BUILDERS.register("RFDETRTrainer")
def build_rfdetr_trainer(cfg: Dict[str, Any]) -> RFDETRTrainer:
    """Build RFDETRTrainer from configuration.

    This function handles all RF-DETR-specific setup including:
    - Dataset creation
    - Model instantiation
    - Optimizer setup
    - EMA model (if enabled)
    - Learning rate scheduler
    - Criterion and postprocessors

    Args:
        cfg: Configuration dictionary from YAML

    Returns:
        Configured RFDETRTrainer instance
    """
    # Parse config
    rfdetr_cfg = OmegaConf.structured(RFDETRExperimentConfig)
    rfdetr_cfg = OmegaConf.merge(rfdetr_cfg, cfg)

    # Build model
    model = build_model(rfdetr_cfg.model)
    device = torch.device(rfdetr_cfg.model.device)
    model = model.to(device)

    # Build criterion and postprocessors
    criterion, postprocessors = build_criterion_and_postprocessors(rfdetr_cfg.model)
    criterion = criterion.to(device)

    # Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        data_cfg=rfdetr_cfg.data,
        model_cfg=rfdetr_cfg.model,
        batch_size=rfdetr_cfg.trainer.batch_size,
        num_workers=rfdetr_cfg.trainer.num_workers,
    )

    # Build optimizer
    optimizer = build_optimizer(
        model=model,
        optimizer_cfg=rfdetr_cfg.optimizer,
        model_cfg=rfdetr_cfg.model
    )

    # Build learning rate scheduler
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        trainer_cfg=rfdetr_cfg.trainer,
        steps_per_epoch=len(train_loader)
    )

    # Build EMA model (optional)
    ema_model = build_ema_model(
        model=model,
        optimizer_cfg=rfdetr_cfg.optimizer
    )

    # Prepare training arguments namespace
    args = prepare_training_args(
        model_cfg=rfdetr_cfg.model,
        data_cfg=rfdetr_cfg.data,
        trainer_cfg=rfdetr_cfg.trainer,
        optimizer_cfg=rfdetr_cfg.optimizer
    )

    # Create work directory
    work_dir = Path(rfdetr_cfg.trainer.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build hooks (if integrations are configured)
    hooks = []
    if hasattr(rfdetr_cfg, 'integrations') and rfdetr_cfg.integrations:
        from animaldet.utils.integrations.wandb import WandbHook

        if (hasattr(rfdetr_cfg.integrations, 'wandb') and
            rfdetr_cfg.integrations.wandb.get('enabled', False)):
            wandb_hook = WandbHook(
                project=rfdetr_cfg.integrations.wandb.get('project', 'animaldet'),
                name=rfdetr_cfg.integrations.wandb.get('name', 'rfdetr_experiment'),
                config=OmegaConf.to_container(rfdetr_cfg, resolve=True),
                tags=rfdetr_cfg.integrations.wandb.get('tags', []),
            )
            hooks.append(wandb_hook)

    # Create and return RFDETRTrainer
    return RFDETRTrainer(
        model=model,
        criterion=criterion,
        postprocessors=postprocessors,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        ema_model=ema_model,
        device=rfdetr_cfg.model.device,
        args=args,
        hooks=hooks,
        work_dir=str(work_dir)
    )