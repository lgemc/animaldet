"""
Weights & Biases (wandb) integration for logging training and evaluation metrics.
"""

import time
from typing import Any, Dict, Optional

from animaldet.engine.hooks import Hook
from animaldet.engine.registry import HOOKS


@HOOKS.register("wandb")
class WandbLogger(Hook):
    """
    Logs training and evaluation metrics to Weights & Biases.

    This hook integrates wandb logging into the training process,
    tracking metrics, hyperparameters, and optionally model checkpoints.

    Args:
        enabled: Enable or disable wandb logging
        project: wandb project name
        name: Run name (auto-generated if None)
        notes: Notes for this run
        tags: List of tags for this run
        entity: wandb entity (team name, uses default user if None)
        dir: Directory to store wandb files (uses default if None)
        mode: Logging mode - 'online', 'offline', or 'disabled'
        log_interval: Number of steps between logging (default: 10)
        log_checkpoints: Whether to log model checkpoints to wandb
        log_gradients: Whether to log gradients
        log_lr: Whether to log learning rate
        init_kwargs: Additional keyword arguments for wandb.init()

    Example:
        >>> from animaldet.utils.integrations import WandbLogger
        >>> logger = WandbLogger(
        ...     enabled=True,
        ...     project="animaldet",
        ...     name="experiment-1",
        ...     tags=["baseline", "herdnet"]
        ... )
        >>> # Add to trainer hooks
        >>> trainer.add_hook(logger)
    """

    def __init__(
        self,
        enabled: bool = False,
        project: str = "animaldet",
        name: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[list[str]] = None,
        entity: Optional[str] = None,
        dir: Optional[str] = None,
        mode: str = "online",
        log_interval: int = 10,
        log_checkpoints: bool = False,
        log_gradients: bool = False,
        log_lr: bool = True,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.enabled = enabled
        self.project = project
        self.name = name
        self.notes = notes
        self.tags = tags or []
        self.entity = entity
        self.dir = dir
        self.mode = mode if enabled else "disabled"
        self.log_interval = log_interval
        self.log_checkpoints = log_checkpoints
        self.log_gradients = log_gradients
        self.log_lr = log_lr
        self.init_kwargs = init_kwargs or {}

        self._wandb = None
        self._run = None
        self._step_count = 0
        self._epoch_start_time: Optional[float] = None

    def _lazy_import_wandb(self):
        """Lazy import wandb to avoid dependency if not enabled."""
        if self._wandb is None and self.enabled:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                print(
                    "Warning: wandb is not installed. "
                    "Install it with: pip install wandb"
                )
                self.enabled = False
        return self._wandb

    def before_train(self, trainer: Any) -> None:
        """Initialize wandb run before training starts."""
        if not self.enabled:
            return

        wandb = self._lazy_import_wandb()
        if wandb is None:
            return

        # Prepare init arguments
        init_args = {
            "project": self.project,
            "name": self.name,
            "notes": self.notes,
            "tags": self.tags,
            "entity": self.entity,
            "dir": self.dir,
            "mode": self.mode,
            **self.init_kwargs,
        }

        # Remove None values
        init_args = {k: v for k, v in init_args.items() if v is not None}

        # Initialize wandb run
        self._run = wandb.init(**init_args)

        # Log trainer configuration if available
        if hasattr(trainer, "cfg"):
            # Convert OmegaConf to dict if needed
            config = trainer.cfg
            if hasattr(config, "to_container"):
                config = config.to_container(resolve=True)
            self._run.config.update(config)

        print(f"wandb: Run initialized - {self._run.url}")

    def after_train(self, trainer: Any) -> None:
        """Finish wandb run after training completes."""
        if not self.enabled or self._run is None:
            return

        wandb = self._lazy_import_wandb()
        if wandb is None:
            return

        # Log final summary metrics if available
        if hasattr(trainer, "metrics"):
            for key, value in trainer.metrics.items():
                self._run.summary[f"final/{key}"] = value

        self._run.finish()
        print("wandb: Run finished")

    def before_epoch(self, trainer: Any, epoch: int) -> None:
        """Log epoch start."""
        if not self.enabled:
            return

        self._epoch_start_time = time.time()

    def after_epoch(self, trainer: Any, epoch: int) -> None:
        """Log epoch metrics."""
        if not self.enabled or self._run is None:
            return

        wandb = self._lazy_import_wandb()
        if wandb is None:
            return

        # Log epoch duration
        if self._epoch_start_time is not None:
            epoch_duration = time.time() - self._epoch_start_time
            self._run.log({"epoch": epoch, "epoch_duration": epoch_duration})

        # Log epoch-level metrics if available
        if hasattr(trainer, "metrics"):
            metrics = {"epoch": epoch}
            for key, value in trainer.metrics.items():
                if isinstance(value, (int, float)):
                    metrics[f"epoch/{key}"] = value
            self._run.log(metrics)

    def after_step(self, trainer: Any, step: int) -> None:
        """Log training metrics at specified intervals."""
        if not self.enabled or self._run is None:
            return

        wandb = self._lazy_import_wandb()
        if wandb is None:
            return

        self._step_count += 1

        if step % self.log_interval == 0:
            metrics = self._get_trainer_metrics(trainer)
            if metrics:
                log_dict = {"step": step}

                # Log metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        log_dict[f"train/{key}"] = value

                # Log learning rate if available and enabled
                if self.log_lr:
                    lr = self._get_learning_rate(trainer)
                    if lr is not None:
                        log_dict["train/lr"] = lr

                self._run.log(log_dict)

        # Log gradients if enabled
        if self.log_gradients and step % self.log_interval == 0:
            self._log_gradients(trainer)

    def after_eval(self, trainer: Any, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics."""
        if not self.enabled or self._run is None:
            return

        wandb = self._lazy_import_wandb()
        if wandb is None:
            return

        # Log evaluation metrics
        log_dict = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_dict[f"eval/{key}"] = value
            elif isinstance(value, dict):
                # Flatten nested dictionaries
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        log_dict[f"eval/{key}/{sub_key}"] = sub_value

        if log_dict:
            self._run.log(log_dict)

    def _get_trainer_metrics(self, trainer: Any) -> Dict[str, Any]:
        """
        Extract current metrics from trainer.

        Override this method if your trainer stores metrics differently.
        """
        if hasattr(trainer, "metrics"):
            return trainer.metrics
        if hasattr(trainer, "get_metrics"):
            return trainer.get_metrics()
        return {}

    def _get_learning_rate(self, trainer: Any) -> Optional[float]:
        """Extract current learning rate from trainer."""
        # Try to get from optimizer
        if hasattr(trainer, "optimizer"):
            optimizer = trainer.optimizer
            if hasattr(optimizer, "param_groups"):
                return optimizer.param_groups[0]["lr"]

        # Try to get from scheduler
        if hasattr(trainer, "scheduler"):
            scheduler = trainer.scheduler
            if hasattr(scheduler, "get_last_lr"):
                lrs = scheduler.get_last_lr()
                return lrs[0] if lrs else None

        return None

    def _log_gradients(self, trainer: Any) -> None:
        """Log gradient statistics to wandb."""
        if not hasattr(trainer, "model"):
            return

        wandb = self._lazy_import_wandb()
        if wandb is None:
            return

        model = trainer.model
        gradients = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[f"gradients/{name}"] = wandb.Histogram(
                    param.grad.cpu().detach().numpy()
                )

        if gradients:
            self._run.log(gradients)

    def log_checkpoint(self, checkpoint_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Manually log a checkpoint to wandb.

        Args:
            checkpoint_path: Path to the checkpoint file
            metadata: Optional metadata to associate with the checkpoint
        """
        if not self.enabled or self._run is None or not self.log_checkpoints:
            return

        wandb = self._lazy_import_wandb()
        if wandb is None:
            return

        artifact = wandb.Artifact(
            name=f"model-{self._run.id}",
            type="model",
            metadata=metadata or {},
        )
        artifact.add_file(checkpoint_path)
        self._run.log_artifact(artifact)
