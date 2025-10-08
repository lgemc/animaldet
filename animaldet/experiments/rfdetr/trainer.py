"""RF-DETR Trainer with hook support.

This module provides a PyTorch Lightning-style trainer wrapper around RF-DETR's
training functions that integrates with animaldet's hook system for extensible logging.
"""

import torch
from typing import Optional, Dict, Any
from pathlib import Path
from torch.utils.data import DataLoader

from animaldet.engine.hooks import HookManager, Hook
from animaldet.engine.registry import TRAINERS

# Import RF-DETR training utilities
import sys
rfdetr_path = Path("/home/lmanrique/Do/rf-detr")
if str(rfdetr_path) not in sys.path:
    sys.path.insert(0, str(rfdetr_path))

from rfdetr.engine import train_one_epoch, evaluate as rfdetr_evaluate
from rfdetr.util.utils import BestMetricHolder
from collections import defaultdict


@TRAINERS.register()
class RFDETRTrainer:
    """
    PyTorch Lightning-style trainer wrapper for RF-DETR with hook support.

    This class wraps RF-DETR's training loop with animaldet's hook system,
    allowing for extensible logging (W&B, TensorBoard, etc.) and custom callbacks.

    Args:
        model: RF-DETR model
        criterion: Loss criterion
        postprocessors: Output postprocessors
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        train_loader: Training dataloader
        val_loader: Validation dataloader
        ema_model: Optional EMA model wrapper
        device: Device to train on
        args: Training arguments namespace
        hooks: List of hooks to register
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        postprocessors: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        ema_model: Optional[Any] = None,
        device: str = "cuda",
        args: Optional[Any] = None,
        hooks: Optional[list[Hook]] = None,
        work_dir: str = "./output"
    ):
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ema_model = ema_model
        self.device = torch.device(device)
        self.args = args
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Hook management
        self.hook_manager = HookManager(hooks or [])

        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.global_step = 0
        self.metrics = {}

        # Best metric tracking
        self.best_metric_holder = BestMetricHolder(use_ema=False)

    def add_hook(self, hook: Hook) -> None:
        """Add a hook to the trainer."""
        self.hook_manager.add_hook(hook)

    def fit(
        self,
        num_epochs: Optional[int] = None,
        max_norm: float = 0.1,
        checkpoint_interval: int = 10
    ) -> torch.nn.Module:
        """
        Fit the model (PyTorch Lightning-style).

        Args:
            num_epochs: Number of epochs to train (uses args.epochs if None)
            max_norm: Gradient clipping max norm
            checkpoint_interval: Save checkpoint every N epochs

        Returns:
            Trained model
        """
        if num_epochs is None:
            num_epochs = self.args.epochs if self.args else 100

        # Call before_train hooks
        self.hook_manager.before_train(self)

        # Create callbacks dict for RF-DETR's training loop
        callbacks = defaultdict(list)

        # Add hook-based callbacks
        def on_batch_start(callback_dict):
            self.current_step = callback_dict.get("step", 0)
            self.global_step = callback_dict.get("step", 0)
            self.hook_manager.before_step(self, self.global_step)

        def on_batch_end(callback_dict):
            # Update metrics from callback dict if available
            if "metrics" in callback_dict:
                self.metrics.update(callback_dict["metrics"])
            self.hook_manager.after_step(self, self.global_step)

        callbacks["on_train_batch_start"].append(on_batch_start)
        callbacks["on_train_batch_end"].append(on_batch_end)

        # Training loop
        num_training_steps_per_epoch = len(self.train_loader)
        vit_encoder_num_layers = 12  # Default for DINOv2

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Before epoch hook
            self.hook_manager.before_epoch(self, epoch)

            # Train one epoch
            train_stats = train_one_epoch(
                model=self.model,
                criterion=self.criterion,
                lr_scheduler=self.lr_scheduler,
                data_loader=self.train_loader,
                optimizer=self.optimizer,
                device=self.device,
                epoch=epoch,
                batch_size=self.args.batch_size,
                max_norm=max_norm,
                ema_m=self.ema_model,
                schedules={},
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                vit_encoder_num_layers=vit_encoder_num_layers,
                args=self.args,
                callbacks=callbacks
            )

            # Update metrics
            self.metrics.update(train_stats)

            # Validation
            if self.val_loader is not None:
                self.hook_manager.before_validation(self, epoch)

                val_stats = rfdetr_evaluate(
                    model=self.ema_model.module if self.ema_model else self.model,
                    criterion=self.criterion,
                    postprocessors=self.postprocessors,
                    data_loader=self.val_loader,
                    base_ds=None,  # Will be inferred from val_loader
                    device=self.device,
                    output_dir=str(self.work_dir),
                    args=self.args
                )

                self.metrics.update(val_stats)
                self.hook_manager.after_validation(self, epoch)

            # After epoch hook
            self.hook_manager.after_epoch(self, epoch)

            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                self._save_checkpoint(epoch)

        # After train hook
        self.hook_manager.after_train(self)

        return self.model

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.work_dir / f"checkpoint_epoch_{epoch}.pth"

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": self.args,
        }

        if self.ema_model is not None:
            checkpoint["ema_model"] = self.ema_model.module.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def _update_metrics_from_stats(self, stats: Dict[str, Any]) -> None:
        """Update metrics from training/validation stats."""
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.metrics[key] = value