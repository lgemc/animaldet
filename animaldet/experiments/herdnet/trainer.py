"""HerdNet Trainer with hook support.

This module provides a PyTorch Lightning-style trainer wrapper around animaloc's
Trainer that integrates with animaldet's hook system for extensible logging.
"""

import torch
from typing import Optional, Dict, Any
from animaloc.train import Trainer as AnimalocTrainer
from animaloc.models import LossWrapper

from animaldet.engine.hooks import HookManager, Hook


class HerdNetTrainer:
    """
    PyTorch Lightning-style trainer wrapper with hook support.

    This class bridges animaloc's built-in Trainer with animaldet's hook system,
    allowing for extensible logging (W&B, TensorBoard, etc.) and custom callbacks
    during training.

    Args:
        trainer: The underlying animaloc Trainer instance
        hooks: List of hooks to register (or None)
    """

    def __init__(
        self,
        trainer: AnimalocTrainer,
        hooks: Optional[list[Hook]] = None
    ):
        self.trainer = trainer
        self.hook_manager = HookManager(hooks or [])

        # Store training state for hooks
        self.current_epoch = 0
        self.current_step = 0
        self.global_step = 0
        self.metrics = {}

    def add_hook(self, hook: Hook) -> None:
        """Add a hook to the trainer."""
        self.hook_manager.add_hook(hook)

    def fit(
        self,
        warmup_iters: Optional[int] = None,
        checkpoint_mode: str = 'best',
        select_mode: str = 'min',
        monitor: str = 'all'
    ) -> torch.nn.Module:
        """
        Fit the model (PyTorch Lightning-style).

        This wraps the animaloc Trainer's start method and injects hook calls
        at appropriate points during training.

        Args:
            warmup_iters: Number of warmup iterations
            checkpoint_mode: Checkpoint saving mode ('best', 'all')
            select_mode: Best model selection mode ('min', 'max')
            monitor: Metric to monitor for checkpointing

        Returns:
            Trained model
        """
        # Call before_train hooks
        self.hook_manager.before_train(self)

        # Monkey-patch the trainer's _train method to add hooks
        original_train = self.trainer._train
        original_evaluate = self.trainer.evaluate if hasattr(self.trainer, 'evaluate') else None
        original_prepare_evaluator = self.trainer._prepare_evaluator if hasattr(self.trainer, '_prepare_evaluator') else None

        def wrapped_train(epoch, warmup_iters=None, wandb_flag=False):
            self.current_epoch = epoch

            # Before epoch
            self.hook_manager.before_epoch(self, epoch)

            # Wrap the training loop to add step hooks
            original_log_every = self.trainer.train_logger.log_every

            def wrapped_log_every(iterable, print_freq, header):
                for idx, batch in enumerate(original_log_every(iterable, print_freq, header)):
                    self.current_step = idx
                    self.hook_manager.before_step(self, self.global_step)
                    yield batch
                    # Update metrics from logger
                    self._update_metrics_from_logger(self.trainer.train_logger)
                    self.global_step += 1
                    self.hook_manager.after_step(self, self.global_step)

            self.trainer.train_logger.log_every = wrapped_log_every

            # Call original training
            result = original_train(epoch, warmup_iters, wandb_flag)

            # Restore original log_every
            self.trainer.train_logger.log_every = original_log_every

            # After epoch
            self.hook_manager.after_epoch(self, epoch)

            return result

        def wrapped_evaluator_prepare(filename, epoch):
            if original_prepare_evaluator:
                return original_prepare_evaluator(filename, epoch)

        def wrapped_evaluate(epoch, reduction='mean', wandb_flag=False, returns='all'):
            # Before eval
            self.hook_manager.before_eval(self)

            if original_evaluate:
                result = original_evaluate(epoch, reduction, wandb_flag, returns)

                # After eval - gather metrics
                eval_metrics = self._gather_eval_metrics(returns, result)
                self.hook_manager.after_eval(self, eval_metrics)

                return result
            return None

        # Apply monkey patches
        self.trainer._train = wrapped_train
        if original_evaluate:
            self.trainer.evaluate = wrapped_evaluate
        if original_prepare_evaluator:
            self.trainer._prepare_evaluator = wrapped_evaluator_prepare

        # Wrap evaluator if it exists
        if self.trainer.evaluator is not None:
            self._wrap_evaluator()

        try:
            # Fit the model
            model = self.trainer.start(
                warmup_iters=warmup_iters,
                checkpoints=checkpoint_mode,
                select=select_mode,
                validate_on=monitor,
                wandb_flag=False  # Disable built-in W&B, use hooks instead
            )

            # Call after_train hooks
            self.hook_manager.after_train(self)

            return model

        finally:
            # Restore original methods
            self.trainer._train = original_train
            if original_evaluate:
                self.trainer.evaluate = original_evaluate
            if original_prepare_evaluator:
                self.trainer._prepare_evaluator = original_prepare_evaluator

    def validate(self, returns: str = 'f1_score') -> float:
        """
        Run validation (PyTorch Lightning-style).

        Args:
            returns: Metric to return

        Returns:
            Validation metric value
        """
        if self.trainer.evaluator is not None:
            self.hook_manager.before_eval(self)
            result = self.trainer.evaluator.evaluate(returns=returns, viz=False)
            eval_metrics = self._gather_eval_metrics(returns, result)
            self.hook_manager.after_eval(self, eval_metrics)
            return result
        elif hasattr(self.trainer, 'evaluate'):
            self.hook_manager.before_eval(self)
            result = self.trainer.evaluate(
                epoch=self.current_epoch,
                returns=returns,
                wandb_flag=False
            )
            eval_metrics = self._gather_eval_metrics(returns, result)
            self.hook_manager.after_eval(self, eval_metrics)
            return result
        else:
            raise ValueError("No evaluator or evaluate method available")

    def _wrap_evaluator(self) -> None:
        """Wrap the evaluator's evaluate method to add hooks."""
        if self.trainer.evaluator is None:
            return

        original_eval = self.trainer.evaluator.evaluate

        def wrapped_eval(returns='f1_score', wandb_flag=False, viz=False, log_meters=True):
            # Before eval
            self.hook_manager.before_eval(self)

            result = original_eval(returns=returns, wandb_flag=wandb_flag, viz=viz, log_meters=log_meters)

            # Handle None return (can happen when metrics can't be computed)
            if result is None:
                result = 0.0

            # After eval - gather metrics
            eval_metrics = self._gather_eval_metrics(returns, result)
            self.hook_manager.after_eval(self, eval_metrics)

            return result

        self.trainer.evaluator.evaluate = wrapped_eval

    def _update_metrics_from_logger(self, logger) -> None:
        """Extract current metrics from the logger."""
        self.metrics = {}

        # Get metrics from logger meters
        for name, meter in logger.meters.items():
            if hasattr(meter, 'global_avg'):
                self.metrics[name] = meter.global_avg
            elif hasattr(meter, 'avg'):
                self.metrics[name] = meter.avg
            elif hasattr(meter, 'value'):
                self.metrics[name] = meter.value

    def _gather_eval_metrics(self, metric_name: str, metric_value: float) -> Dict[str, Any]:
        """Gather evaluation metrics into a dictionary."""
        metrics = {metric_name: metric_value}

        # Try to get additional metrics from evaluator if available
        if hasattr(self.trainer, 'evaluator') and self.trainer.evaluator is not None:
            if hasattr(self.trainer.evaluator, 'metrics_dict'):
                metrics.update(self.trainer.evaluator.metrics_dict)

        return metrics

    @property
    def model(self) -> LossWrapper:
        """Get the underlying model (PyTorch Lightning-style property)."""
        return self.trainer.model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer (PyTorch Lightning-style property)."""
        return self.trainer.optimizer

    @property
    def current_lr(self) -> float:
        """Get current learning rate."""
        return self.trainer.optimizer.param_groups[0]["lr"]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return self.metrics.copy()

    @property
    def state_dict(self) -> Dict[str, Any]:
        """Get trainer state dictionary."""
        return {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
