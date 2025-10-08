"""Builder function for HerdNetTrainer from configuration.

This module provides a factory function to build a HerdNetTrainer instance
from a configuration dictionary, handling all HerdNet-specific setup.
"""

import torch
from typing import Dict, Any
from omegaconf import OmegaConf

from animaloc.train import Trainer as AnimalocTrainer

from animaldet.engine.registry import TRAINER_BUILDERS, MODELS, OPTIMIZERS
from animaldet.engine.hooks_builder import build_hooks_from_config, register_integration_hooks
from animaldet.experiments.herdnet.trainer import HerdNetTrainer
from animaldet.experiments.herdnet.adapters.dataset import build_train_dataset, build_val_dataset
from animaldet.experiments.herdnet.adapters.evaluator import build_evaluator
from animaldet.experiments.herdnet.adapters.config import HerdNetExperimentConfig

# Import to register HerdNet model
from animaldet.experiments.herdnet.adapters import model  # noqa: F401


@TRAINER_BUILDERS.register("HerdNetTrainer")
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

    # Build model using registry
    model = MODELS.build("HerdNet", herdnet_cfg.model)

    # Load checkpoint if resuming
    from pathlib import Path
    checkpoint_path = herdnet_cfg.trainer.resume_from_checkpoint
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print(f"Checkpoint loaded successfully")

    # Build optimizer using registry
    optimizer = OPTIMIZERS.build(
        herdnet_cfg.optimizer.name,
        model.parameters(),
        OmegaConf.to_container(herdnet_cfg.optimizer, resolve=True)
    )

    # Load optimizer state if resuming
    if herdnet_cfg.trainer.resume_from_checkpoint is not None:
        checkpoint = torch.load(herdnet_cfg.trainer.resume_from_checkpoint, map_location='cpu')
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")

    # Build validation dataloader
    # Note: batch_size=1 required for HerdNet evaluator
    # Note: Don't use collate_fn - evaluator expects dict format, not tuple
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
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
    # Note: training dataloader doesn't use collate_fn when MultiTransformsWrapper is used
    # The trainer handles the tuple targets internally
    animaloc_trainer = AnimalocTrainer(
        model=model,
        train_dataloader=torch.utils.data.DataLoader(
            train_dataset,
            batch_size=herdnet_cfg.trainer.batch_size,
            shuffle=True
        ),
        optimizer=optimizer,
        num_epochs=herdnet_cfg.trainer.num_epochs,
        evaluator=evaluator,
        work_dir=herdnet_cfg.trainer.work_dir
    )

    # Register integration hooks
    register_integration_hooks()

    # Build hooks from integration configs
    hooks = build_hooks_from_config(cfg)

    # Create and return HerdNetTrainer with hooks
    return HerdNetTrainer(trainer=animaloc_trainer, hooks=hooks)