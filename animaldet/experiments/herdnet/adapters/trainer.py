"""Training utilities for HerdNet experiments."""

from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from animaloc.train import Trainer
from animaloc.models import LossWrapper

from .config import OptimizerConfig, TrainerConfig
from .evaluator import build_evaluator


def build_optimizer(model: LossWrapper, cfg: OptimizerConfig):
    """
    Build optimizer from config.

    Args:
        model: Model to optimize
        cfg: Optimizer configuration

    Returns:
        Optimizer instance
    """
    if cfg.name.lower() == "adam":
        return Adam(
            params=model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.name.lower() == "sgd":
        return SGD(
            params=model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.name}")


def build_trainer(
    model: LossWrapper,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_cfg: OptimizerConfig,
    trainer_cfg: TrainerConfig,
    evaluator_cfg,
    model_cfg,
    data_cfg
) -> Trainer:
    """
    Build HerdNet trainer.

    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        optimizer_cfg: Optimizer configuration
        trainer_cfg: Trainer configuration
        evaluator_cfg: Evaluator configuration
        model_cfg: Model configuration
        data_cfg: Data configuration

    Returns:
        Trainer instance
    """
    optimizer = build_optimizer(model, optimizer_cfg)

    evaluator = build_evaluator(
        model=model,
        dataloader=val_loader,
        work_dir=trainer_cfg.work_dir,
        evaluator_cfg=evaluator_cfg,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        header="validation"
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        num_epochs=trainer_cfg.num_epochs,
        evaluator=evaluator,
        work_dir=trainer_cfg.work_dir
    )

    return trainer


def train(
    trainer: Trainer,
    cfg: TrainerConfig
) -> None:
    """
    Start training.

    Args:
        trainer: Trainer instance
        cfg: Training configuration
    """
    trainer.start(
        warmup_iters=cfg.warmup_iters,
        checkpoints=cfg.checkpoints,
        select=cfg.select,
        validate_on=cfg.validate_on
    )