"""
TensorBoard integration for logging training and evaluation metrics.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

from animaldet.engine.hooks import Hook
from animaldet.engine.registry import HOOKS


@HOOKS.register("tensorboard")
class TensorBoardLogger(Hook):
    """
    Logs training and evaluation metrics to TensorBoard.

    This hook integrates TensorBoard logging into the training process,
    tracking metrics, hyperparameters, and optionally model graphs and histograms.

    Args:
        enabled: Enable or disable tensorboard logging
        log_dir: Directory to store tensorboard logs (relative to work_dir or absolute)
        log_interval: Number of steps between logging (default: 10)
        flush_secs: How often to flush logs to disk in seconds (default: 120)
        log_histograms: Whether to log parameter histograms
        log_graph: Whether to log model graph
        log_gradients: Whether to log gradients
        log_lr: Whether to log learning rate
        log_images: Whether to log images (if supported by trainer)
        max_images: Maximum number of images to log per batch

    Example:
        >>> from animaldet.utils.integrations import TensorBoardLogger
        >>> logger = TensorBoardLogger(
        ...     enabled=True,
        ...     log_dir="runs/experiment-1",
        ...     log_histograms=True
        ... )
        >>> # Add to trainer hooks
        >>> trainer.add_hook(logger)
    """

    def __init__(
        self,
        enabled: bool = False,
        log_dir: str = "tensorboard",
        log_interval: int = 10,
        flush_secs: int = 120,
        log_histograms: bool = False,
        log_graph: bool = False,
        log_gradients: bool = False,
        log_lr: bool = True,
        log_images: bool = False,
        max_images: int = 8,
    ):
        self.enabled = enabled
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.flush_secs = flush_secs
        self.log_histograms = log_histograms
        self.log_graph = log_graph
        self.log_gradients = log_gradients
        self.log_lr = log_lr
        self.log_images = log_images
        self.max_images = max_images

        self._writer = None
        self._step_count = 0
        self._epoch_start_time: Optional[float] = None
        self._graph_logged = False

    def _lazy_import_tensorboard(self):
        """Lazy import tensorboard to avoid dependency if not enabled."""
        if self._writer is None and self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                return SummaryWriter
            except ImportError:
                print(
                    "Warning: tensorboard is not installed. "
                    "Install it with: pip install tensorboard"
                )
                self.enabled = False
                return None
        return None

    def _get_log_dir(self, trainer: Any) -> Path:
        """Get the full log directory path."""
        log_dir = Path(self.log_dir)

        # If relative path, make it relative to work_dir
        if not log_dir.is_absolute():
            if hasattr(trainer, "work_dir"):
                log_dir = Path(trainer.work_dir) / log_dir
            elif hasattr(trainer, "cfg") and hasattr(trainer.cfg.trainer, "work_dir"):
                log_dir = Path(trainer.cfg.trainer.work_dir) / log_dir

        return log_dir

    def before_train(self, trainer: Any) -> None:
        """Initialize TensorBoard writer before training starts."""
        if not self.enabled:
            return

        SummaryWriter = self._lazy_import_tensorboard()
        if SummaryWriter is None:
            return

        log_dir = self._get_log_dir(trainer)
        log_dir.mkdir(parents=True, exist_ok=True)

        self._writer = SummaryWriter(
            log_dir=str(log_dir),
            flush_secs=self.flush_secs,
        )

        # Log hyperparameters if available
        if hasattr(trainer, "cfg"):
            config = trainer.cfg
            if hasattr(config, "to_container"):
                config_dict = config.to_container(resolve=True)
                self._log_hparams(config_dict)

        print(f"TensorBoard: Logging to {log_dir}")

    def after_train(self, trainer: Any) -> None:
        """Close TensorBoard writer after training completes."""
        if not self.enabled or self._writer is None:
            return

        # Log final metrics as hparams
        if hasattr(trainer, "metrics"):
            self._writer.add_hparams(
                hparam_dict={},
                metric_dict={f"final/{k}": v for k, v in trainer.metrics.items()
                            if isinstance(v, (int, float))},
            )

        self._writer.close()
        print("TensorBoard: Logging finished")

    def before_epoch(self, trainer: Any, epoch: int) -> None:
        """Log epoch start."""
        if not self.enabled:
            return

        self._epoch_start_time = time.time()

    def after_epoch(self, trainer: Any, epoch: int) -> None:
        """Log epoch metrics."""
        if not self.enabled or self._writer is None:
            return

        # Log epoch duration
        if self._epoch_start_time is not None:
            epoch_duration = time.time() - self._epoch_start_time
            self._writer.add_scalar("epoch/duration", epoch_duration, epoch)

        # Log epoch-level metrics if available
        if hasattr(trainer, "metrics"):
            for key, value in trainer.metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(f"epoch/{key}", value, epoch)

        # Log histograms if enabled
        if self.log_histograms:
            self._log_model_histograms(trainer, epoch)

        self._writer.flush()

    def after_step(self, trainer: Any, step: int) -> None:
        """Log training metrics at specified intervals."""
        if not self.enabled or self._writer is None:
            return

        self._step_count += 1

        # Log model graph once
        if self.log_graph and not self._graph_logged:
            self._log_model_graph(trainer)
            self._graph_logged = True

        if step % self.log_interval == 0:
            metrics = self._get_trainer_metrics(trainer)
            if metrics:
                # Log metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self._writer.add_scalar(f"train/{key}", value, step)

                # Log learning rate if available and enabled
                if self.log_lr:
                    lr = self._get_learning_rate(trainer)
                    if lr is not None:
                        self._writer.add_scalar("train/lr", lr, step)

            # Log gradients if enabled
            if self.log_gradients:
                self._log_model_gradients(trainer, step)

            # Log images if enabled
            if self.log_images:
                self._log_batch_images(trainer, step)

    def after_eval(self, trainer: Any, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics."""
        if not self.enabled or self._writer is None:
            return

        # Get current step/epoch for x-axis
        global_step = self._step_count

        # Log evaluation metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(f"eval/{key}", value, global_step)
            elif isinstance(value, dict):
                # Flatten nested dictionaries
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        self._writer.add_scalar(
                            f"eval/{key}/{sub_key}", sub_value, global_step
                        )

        self._writer.flush()

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

    def _log_hparams(self, config_dict: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        # Flatten nested config
        flat_config = self._flatten_dict(config_dict)

        # Filter to only include simple types
        hparams = {
            k: v
            for k, v in flat_config.items()
            if isinstance(v, (int, float, str, bool))
        }

        if hparams:
            self._writer.add_hparams(hparam_dict=hparams, metric_dict={})

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _log_model_graph(self, trainer: Any) -> None:
        """Log model graph to TensorBoard."""
        if not hasattr(trainer, "model"):
            return

        try:
            # Need a sample input to trace the model
            if hasattr(trainer, "dataloader"):
                # Try to get a batch from dataloader
                sample_batch = next(iter(trainer.dataloader))
                if isinstance(sample_batch, (list, tuple)):
                    sample_input = sample_batch[0]
                else:
                    sample_input = sample_batch

                self._writer.add_graph(trainer.model, sample_input)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")

    def _log_model_histograms(self, trainer: Any, step: int) -> None:
        """Log model parameter histograms to TensorBoard."""
        if not hasattr(trainer, "model"):
            return

        model = trainer.model
        for name, param in model.named_parameters():
            if param is not None:
                self._writer.add_histogram(f"params/{name}", param.data, step)

    def _log_model_gradients(self, trainer: Any, step: int) -> None:
        """Log gradient histograms to TensorBoard."""
        if not hasattr(trainer, "model"):
            return

        model = trainer.model
        for name, param in model.named_parameters():
            if param.grad is not None:
                self._writer.add_histogram(f"gradients/{name}", param.grad.data, step)

    def _log_batch_images(self, trainer: Any, step: int) -> None:
        """Log batch images to TensorBoard."""
        if not hasattr(trainer, "last_batch"):
            return

        try:
            import torch

            batch = trainer.last_batch
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            # Limit number of images
            if isinstance(images, torch.Tensor):
                images = images[: self.max_images]
                self._writer.add_images("train/images", images, step)
        except Exception as e:
            print(f"Warning: Could not log images: {e}")