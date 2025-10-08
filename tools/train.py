#!/usr/bin/env python3
"""Training script for animaldet experiments.

This script provides a unified entry point for training models across different
experiments (HerdNet, RT-DETR, etc.) using the registry system.

Usage:
    python tools/train.py --config configs/experiment/herdnet.yaml
    python tools/train.py --config configs/experiment/rtdetr.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import OmegaConf

from animaldet.engine.registry import TRAINERS
from animaldet.experiments.herdnet.builder import build_herdnet_trainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train animal detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file (YAML)"
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default=None,
        help="Trainer name from registry (overrides config). Available: "
             f"{', '.join(TRAINERS.registry_names)}"
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Working directory for outputs (overrides config)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_file)
    # Resolve interpolations
    cfg = OmegaConf.to_container(cfg, resolve=True)

    return cfg


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_trainer_from_config(cfg: Dict[str, Any], trainer_name: str = None) -> Any:
    """Build trainer from configuration.

    This function is experiment-agnostic and delegates to the experiment-specific
    builder functions to handle setup.

    Args:
        cfg: Configuration dictionary
        trainer_name: Optional trainer name to use (overrides config)

    Returns:
        Configured trainer instance
    """
    # Determine which trainer to use
    if trainer_name is not None:
        name = trainer_name
    elif "trainer" in cfg and "name" in cfg["trainer"]:
        name = cfg["trainer"]["name"]
    elif len(TRAINERS) == 1:
        name = TRAINERS.registry_names[0]
    else:
        raise ValueError(
            "Please specify a trainer name. "
            f"Available: {TRAINERS.registry_names}"
        )

    # Validate trainer exists
    if name not in TRAINERS:
        raise ValueError(
            f"Trainer '{name}' not found in registry. "
            f"Available: {TRAINERS.registry_names}"
        )

    # Map trainer names to their builder functions
    builder_map = {
        "HerdNetTrainer": build_herdnet_trainer,
    }

    if name not in builder_map:
        raise NotImplementedError(
            f"No builder function registered for trainer '{name}'. "
            f"Please add it to the builder_map in tools/train.py"
        )

    # Use the builder function to create the trainer
    trainer = builder_map[name](cfg)

    return trainer


def main():
    """Main training entry point."""
    args = parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Override config with CLI arguments
    if args.work_dir is not None:
        if "trainer" not in cfg:
            cfg["trainer"] = {}
        cfg["trainer"]["work_dir"] = args.work_dir

    if args.seed is not None:
        cfg["seed"] = args.seed

    # Set random seed
    seed = cfg.get("seed", 42)
    set_random_seed(seed)

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    work_dir = Path(cfg.get("trainer", {}).get("work_dir", "./output"))
    work_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(work_dir / "train.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("animaldet")

    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Available trainers: {TRAINERS.registry_names}")

    # Build trainer
    logger.info("Building trainer from configuration...")
    trainer = build_trainer_from_config(cfg, trainer_name=args.trainer)

    logger.info(f"Using trainer: {trainer.__class__.__name__}")

    # Resume from checkpoint if specified
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        if hasattr(trainer, "load_checkpoint"):
            trainer.load_checkpoint(args.resume)
        else:
            logger.warning("Trainer does not support checkpoint loading")

    # Train
    logger.info("Starting training...")
    try:
        trainer.fit()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        raise

    logger.info("Done!")


if __name__ == "__main__":
    main()
