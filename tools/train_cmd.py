"""Training command implementation."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf

from animaldet.engine.registry import TRAINER_BUILDERS

# Import builders to ensure they are registered
from animaldet.experiments.herdnet import builder as _  # noqa: F401
from animaldet.experiments.rfdetr import builder as __  # noqa: F401


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


def build_trainer_from_config(cfg: Dict[str, Any], trainer_name: Optional[str] = None) -> Any:
    """Build trainer from configuration.

    This function is experiment-agnostic and uses the registry to find
    the appropriate builder function.

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
    elif len(TRAINER_BUILDERS) == 1:
        name = TRAINER_BUILDERS.registry_names[0]
    else:
        raise ValueError(
            "Please specify a trainer name. "
            f"Available: {TRAINER_BUILDERS.registry_names}"
        )

    # Get builder function from registry and build trainer
    builder_fn = TRAINER_BUILDERS[name]
    trainer = builder_fn(cfg)

    return trainer


def train_main(
    config: Optional[str],
    trainer: Optional[str],
    work_dir: Optional[str],
    seed: Optional[int],
    resume: Optional[str],
    device: str,
    debug: bool,
):
    """Main training entry point.

    Args:
        config: Path to configuration file
        trainer: Trainer name from registry
        work_dir: Working directory for outputs
        seed: Random seed
        resume: Path to checkpoint to resume from
        device: Device to use for training
        debug: Enable debug mode
    """
    if config is None:
        raise ValueError("--config is required for training")

    # Load configuration
    cfg = load_config(config)

    # Override config with CLI arguments
    if work_dir is not None:
        if "trainer" not in cfg:
            cfg["trainer"] = {}
        cfg["trainer"]["work_dir"] = work_dir

    if seed is not None:
        cfg["seed"] = seed

    # Set random seed
    seed_value = cfg.get("seed", 42)
    set_random_seed(seed_value)

    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    work_dir_path = Path(cfg.get("trainer", {}).get("work_dir", "./outputs/base_logger"))
    work_dir_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(work_dir_path / "train.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("animaldet")

    logger.info(f"Starting training with config: {config}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {seed_value}")
    logger.info(f"Available trainers: {TRAINER_BUILDERS.registry_names}")

    # Build trainer
    logger.info("Building trainer from configuration...")
    trainer_instance = build_trainer_from_config(cfg, trainer_name=trainer)

    logger.info(f"Using trainer: {trainer_instance.__class__.__name__}")

    # Resume from checkpoint if specified
    if resume is not None:
        logger.info(f"Resuming from checkpoint: {resume}")
        if hasattr(trainer_instance, "load_checkpoint"):
            trainer_instance.load_checkpoint(resume)
        else:
            logger.warning("Trainer does not support checkpoint loading")

    # Train
    logger.info("Starting training...")
    try:
        trainer_instance.fit()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        raise

    logger.info("Done!")