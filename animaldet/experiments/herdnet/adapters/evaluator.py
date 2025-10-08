"""Evaluation utilities for HerdNet experiments."""

from torch.utils.data import DataLoader
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator
from animaloc.models import LossWrapper

from .config import EvaluatorConfig, ModelConfig, DataConfig


def build_evaluator(
    model: LossWrapper,
    dataloader: DataLoader,
    work_dir: str,
    evaluator_cfg: EvaluatorConfig,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    header: str = "validation"
) -> HerdNetEvaluator:
    """
    Build HerdNet evaluator.

    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        work_dir: Working directory for outputs
        evaluator_cfg: Evaluator configuration
        model_cfg: Model configuration
        data_cfg: Data configuration
        header: Header for evaluation outputs

    Returns:
        HerdNetEvaluator instance
    """
    # Build metrics
    metrics = PointsMetrics(
        radius=evaluator_cfg.metrics_radius,
        num_classes=model_cfg.num_classes
    )

    # Build stitcher
    stitcher = HerdNetStitcher(
        model=model,
        size=(data_cfg.patch_size, data_cfg.patch_size),
        overlap=evaluator_cfg.stitcher_overlap,
        down_ratio=model_cfg.down_ratio,
        reduction=evaluator_cfg.stitcher_reduction
    )

    # Build evaluator
    evaluator = HerdNetEvaluator(
        model=model,
        dataloader=dataloader,
        metrics=metrics,
        stitcher=stitcher,
        work_dir=work_dir,
        header=header
    )

    return evaluator


def evaluate(
    evaluator: HerdNetEvaluator,
    returns: str = "f1_score"
) -> float:
    """
    Run evaluation.

    Args:
        evaluator: Evaluator instance
        returns: Metric to return ('f1_score', 'precision', 'recall', etc.)

    Returns:
        Evaluation metric value
    """
    return evaluator.evaluate(returns=returns)