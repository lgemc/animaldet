"""Evaluation utilities for RF-DETR experiments."""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any

# Add rf-detr to path
rfdetr_path = Path("/home/lmanrique/Do/rf-detr")
if str(rfdetr_path) not in sys.path:
    sys.path.insert(0, str(rfdetr_path))

from rfdetr.engine import evaluate as rfdetr_evaluate
from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.datasets.coco_eval import CocoEvaluator

from .config import EvaluatorConfig, ModelConfig


def build_evaluator(
    model: torch.nn.Module,
    dataloader: DataLoader,
    evaluator_cfg: EvaluatorConfig,
    model_cfg: ModelConfig,
    postprocessors: Dict[str, Any],
    device: str = "cuda"
) -> CocoEvaluator:
    """
    Build COCO evaluator for RF-DETR.

    Args:
        model: Model to evaluate
        dataloader: Validation/test dataloader
        evaluator_cfg: Evaluator configuration
        model_cfg: Model configuration
        postprocessors: Postprocessors for model outputs
        device: Device to run evaluation on

    Returns:
        CocoEvaluator instance
    """
    # Get COCO API from dataset
    base_ds = get_coco_api_from_dataset(dataloader.dataset)

    # Create evaluator
    iou_types = ["bbox"]
    evaluator = CocoEvaluator(base_ds, iou_types)

    return evaluator


def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors: Dict[str, Any],
    dataloader: DataLoader,
    evaluator: CocoEvaluator,
    device: torch.device,
    args: Any,
    ema_model: Any = None
) -> Dict[str, float]:
    """
    Run evaluation on a dataset.

    Args:
        model: Model to evaluate
        criterion: Loss criterion
        postprocessors: Output postprocessors
        dataloader: Data loader
        evaluator: COCO evaluator
        device: Device to run on
        args: Training arguments namespace
        ema_model: Optional EMA model

    Returns:
        Dictionary of evaluation metrics
    """
    # Use EMA model if available and configured
    eval_model = model
    if ema_model is not None and hasattr(args, 'eval_on_ema') and args.eval_on_ema:
        eval_model = ema_model.module

    # Run RF-DETR's evaluation
    stats = rfdetr_evaluate(
        model=eval_model,
        criterion=criterion,
        postprocessors=postprocessors,
        data_loader=dataloader,
        base_ds=evaluator.coco_eval["bbox"].coco_gt,
        device=device,
        output_dir=None,
        args=args
    )

    return stats