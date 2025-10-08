"""Builder function for HerdNetTrainer from configuration.

This module provides a factory function to build a HerdNetTrainer instance
from a configuration dictionary, handling all HerdNet-specific setup.
"""

import torch
from typing import Dict, Any
from omegaconf import OmegaConf

from animaloc.train import Trainer as AnimalocTrainer
from torch.optim import Adam

from animaldet.experiments.herdnet.trainer import HerdNetTrainer
from animaldet.experiments.herdnet.adapters.dataset import build_train_dataset, build_val_dataset
from animaldet.experiments.herdnet.adapters.model import build_model
from animaldet.experiments.herdnet.adapters.evaluator import build_evaluator
from animaldet.experiments.herdnet.adapters.config import HerdNetExperimentConfig


def build_herdnet_trainer(cfg: Dict[str, Any]) -> HerdNetTrainer:
    """Build HerdNetTrainer from configuration.

    This function handles all HerdNet-specific setup including:
    - Dataset creation
    - Model instantiation
    - Optimizer setup
    - Creating the underlying animaloc Trainer

    Args:
        cfg: Configuration dictionary from YAML

    Returns:
        Configured HerdNetTrainer instance
    """
    # Parse config
    herdnet_cfg = OmegaConf.structured(HerdNetExperimentConfig)
    herdnet_cfg = OmegaConf.merge(herdnet_cfg, cfg)

    # Build datasets
    train_dataset = build_train_dataset(herdnet_cfg.data, herdnet_cfg.model)
    val_dataset = build_val_dataset(herdnet_cfg.data, herdnet_cfg.model)

    # Build model
    model = build_model(herdnet_cfg.model)

    # Build optimizer
    optimizer = Adam(
        model.parameters(),
        lr=herdnet_cfg.optimizer.lr,
        weight_decay=herdnet_cfg.optimizer.weight_decay
    )

    # Build validation dataloader
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=herdnet_cfg.trainer.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else None
    )

    # Build evaluator
    evaluator = build_evaluator(
        model=model,
        dataloader=val_dataloader,
        work_dir=herdnet_cfg.trainer.work_dir,
        evaluator_cfg=herdnet_cfg.evaluator,
        model_cfg=herdnet_cfg.model,
        data_cfg=herdnet_cfg.data
    )

    # Create animaloc Trainer
    animaloc_trainer = AnimalocTrainer(
        model=model,
        train_dataloader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=herdnet_cfg.trainer.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None
        ),
        optimizer=optimizer,
        num_epochs=herdnet_cfg.trainer.num_epochs,
        evaluator=evaluator,
        work_dir=herdnet_cfg.trainer.work_dir
    )

    # Create and return HerdNetTrainer
    return HerdNetTrainer(trainer=animaloc_trainer)