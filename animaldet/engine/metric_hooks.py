"""Metric computation hooks.

Hooks that compute additional metrics during evaluation.
"""

from typing import Any, Dict, Optional
from animaldet.engine.hooks import Hook
from animaldet.evaluation.f1_evaluator import F1Evaluator


class F1MetricHook(Hook):
    """
    Computes F1 score metrics after evaluation.

    Args:
        enabled: Whether F1 metric computation is enabled
        center_threshold: Maximum center distance in pixels for matching
        score_threshold: Minimum confidence score for predictions
        verbose: Print detailed debug information
    """

    def __init__(
        self,
        enabled: bool = True,
        center_threshold: float = 50.0,
        score_threshold: float = 0.5,
        verbose: bool = False
    ):
        self.enabled = enabled
        self.evaluator = F1Evaluator(
            center_threshold=center_threshold,
            score_threshold=score_threshold,
            enabled=enabled,
            verbose=verbose
        )

    def after_eval(self, trainer: Any, metrics: Dict[str, Any]) -> None:
        """Compute F1 metrics if predictions and ground truths are available."""
        if not self.enabled:
            return

        # Check if trainer has collected predictions/ground truths
        if not hasattr(trainer, 'eval_predictions') or not hasattr(trainer, 'eval_ground_truths'):
            return

        predictions = trainer.eval_predictions
        ground_truths = trainer.eval_ground_truths

        if not predictions or not ground_truths:
            return

        # Get class names from trainer if available
        class_names = None
        if hasattr(trainer, 'class_names'):
            class_names = trainer.class_names

        # Compute F1 metrics
        f1_metrics = self.evaluator.evaluate(
            predictions=predictions,
            ground_truths=ground_truths,
            class_names=class_names
        )

        if f1_metrics:
            metrics['f1_metrics'] = f1_metrics
            print(f"\n[F1 Metrics] Overall F1: {f1_metrics['overall']['f1_score']:.4f}")
            print(f"[F1 Metrics] Precision: {f1_metrics['overall']['precision']:.4f}")
            print(f"[F1 Metrics] Recall: {f1_metrics['overall']['recall']:.4f}")
