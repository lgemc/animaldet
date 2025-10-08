"""
Logger hooks for tracking training and evaluation metrics.
"""

import time
from typing import Any, Dict, Optional

from animaldet.engine.hooks import Hook


class ConsoleLogger(Hook):
    """
    Logs training and evaluation metrics to the console.

    Args:
        log_interval: Number of steps between logging (default: 10)
        metric_format: Format string for metrics (default: ".4f")
    """

    def __init__(self, log_interval: int = 10, metric_format: str = ".4f"):
        self.log_interval = log_interval
        self.metric_format = metric_format
        self._start_time: Optional[float] = None
        self._epoch_start_time: Optional[float] = None
        self._step_times: list[float] = []

    def before_train(self, trainer: Any) -> None:
        """Log training start."""
        self._start_time = time.time()
        print("=" * 80)
        print("Starting training")
        print("=" * 80)

    def after_train(self, trainer: Any) -> None:
        """Log training completion and total time."""
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            print("=" * 80)
            print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
            print("=" * 80)

    def before_epoch(self, trainer: Any, epoch: int) -> None:
        """Log epoch start."""
        self._epoch_start_time = time.time()
        self._step_times = []
        print(f"\nEpoch {epoch}")
        print("-" * 80)

    def after_epoch(self, trainer: Any, epoch: int) -> None:
        """Log epoch completion and average step time."""
        if self._epoch_start_time is not None:
            elapsed = time.time() - self._epoch_start_time
            avg_step_time = sum(self._step_times) / len(self._step_times) if self._step_times else 0
            print(f"Epoch {epoch} completed in {elapsed:.2f}s (avg step: {avg_step_time*1000:.2f}ms)")

    def before_step(self, trainer: Any, step: int) -> None:
        """Record step start time."""
        self._step_start_time = time.time()

    def after_step(self, trainer: Any, step: int) -> None:
        """Log training metrics at specified intervals."""
        if hasattr(self, '_step_start_time'):
            step_time = time.time() - self._step_start_time
            self._step_times.append(step_time)

        if step % self.log_interval == 0:
            metrics = self._get_trainer_metrics(trainer)
            metrics_str = self._format_metrics(metrics)
            print(f"Step {step:6d} | {metrics_str}")

    def before_eval(self, trainer: Any) -> None:
        """Log evaluation start."""
        print("\nStarting evaluation...")

    def after_eval(self, trainer: Any, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics."""
        print("\nEvaluation Results:")
        print("-" * 80)
        metrics_str = self._format_metrics(metrics)
        print(metrics_str)
        print("-" * 80)

    def _get_trainer_metrics(self, trainer: Any) -> Dict[str, Any]:
        """
        Extract current metrics from trainer.

        Override this method if your trainer stores metrics differently.
        """
        if hasattr(trainer, 'metrics'):
            return trainer.metrics
        if hasattr(trainer, 'get_metrics'):
            return trainer.get_metrics()
        return {}

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics dictionary as string."""
        if not metrics:
            return "No metrics available"

        formatted = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted.append(f"{key}: {value:{self.metric_format}}")
            else:
                formatted.append(f"{key}: {value}")

        return " | ".join(formatted)
