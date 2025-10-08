"""Training utilities for RF-DETR experiments."""

import sys
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from argparse import Namespace

# Add rf-detr to path
rfdetr_path = Path("/home/lmanrique/Do/rf-detr")
if str(rfdetr_path) not in sys.path:
    sys.path.insert(0, str(rfdetr_path))

from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.utils import ModelEma

from .config import OptimizerConfig, TrainerConfig, ModelConfig


def build_optimizer(
    model: torch.nn.Module,
    optimizer_cfg: OptimizerConfig,
    model_cfg: ModelConfig
) -> torch.optim.Optimizer:
    """
    Build optimizer for RF-DETR model.

    Args:
        model: Model to optimize
        optimizer_cfg: Optimizer configuration
        model_cfg: Model configuration (for architecture-specific params)

    Returns:
        Optimizer instance
    """
    # Create args namespace for get_param_dict
    args = Namespace()
    args.lr = optimizer_cfg.lr
    args.lr_encoder = optimizer_cfg.lr_encoder
    args.weight_decay = optimizer_cfg.weight_decay
    args.lr_vit_layer_decay = optimizer_cfg.lr_vit_layer_decay
    args.lr_component_decay = optimizer_cfg.lr_component_decay
    args.out_feature_indexes = model_cfg.out_feature_indexes

    # Get parameter groups with different learning rates
    # This follows RF-DETR's strategy of different LRs for encoder/decoder
    param_dicts = get_param_dict(args, model)

    optimizer = AdamW(
        param_dicts,
        lr=optimizer_cfg.lr,
        weight_decay=optimizer_cfg.weight_decay
    )

    return optimizer


def build_ema_model(
    model: torch.nn.Module,
    optimizer_cfg: OptimizerConfig
) -> Optional[ModelEma]:
    """
    Build EMA (Exponential Moving Average) model wrapper.

    Args:
        model: Model to track with EMA
        optimizer_cfg: Optimizer config (contains EMA settings)

    Returns:
        ModelEma instance or None if EMA is disabled
    """
    if not optimizer_cfg.use_ema:
        return None

    return ModelEma(
        model,
        decay=optimizer_cfg.ema_decay,
        tau=optimizer_cfg.ema_tau
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    trainer_cfg: TrainerConfig,
    steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        trainer_cfg: Trainer configuration
        steps_per_epoch: Number of training steps per epoch

    Returns:
        Learning rate scheduler
    """
    total_steps = trainer_cfg.epochs * steps_per_epoch
    warmup_steps = int(trainer_cfg.warmup_epochs * steps_per_epoch)

    # Use step LR with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Step decay
            epoch = step // steps_per_epoch
            if epoch >= trainer_cfg.lr_drop:
                return 0.1
            return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

    return scheduler


def build_criterion_and_postprocessors(model_cfg: ModelConfig):
    """
    Build loss criterion and postprocessors for RF-DETR.

    Args:
        model_cfg: Model configuration

    Returns:
        Tuple of (criterion, postprocessors)
    """
    from rfdetr.models import build_criterion_and_postprocessors as build_rfdetr_criterion
    from argparse import Namespace

    # Create args namespace for rfdetr with all required attributes
    args = Namespace()

    # Core settings
    args.device = model_cfg.device
    args.num_classes = model_cfg.num_classes
    args.group_detr = model_cfg.group_detr
    args.num_select = model_cfg.num_select

    # Loss coefficients
    args.cls_loss_coef = model_cfg.cls_loss_coef
    args.bbox_loss_coef = model_cfg.bbox_loss_coef
    args.giou_loss_coef = model_cfg.giou_loss_coef

    # Matcher cost weights
    args.set_cost_class = model_cfg.set_cost_class
    args.set_cost_bbox = model_cfg.set_cost_bbox
    args.set_cost_giou = model_cfg.set_cost_giou
    args.focal_alpha = model_cfg.focal_alpha

    # Auxiliary loss settings
    args.aux_loss = model_cfg.aux_loss
    args.dec_layers = model_cfg.dec_layers
    args.two_stage = model_cfg.two_stage

    # Advanced loss options
    args.sum_group_losses = model_cfg.sum_group_losses
    args.use_varifocal_loss = model_cfg.use_varifocal_loss
    args.use_position_supervised_loss = model_cfg.use_position_supervised_loss
    args.ia_bce_loss = model_cfg.ia_bce_loss

    # Segmentation settings (optional)
    args.segmentation_head = model_cfg.segmentation_head
    if model_cfg.segmentation_head:
        args.mask_ce_loss_coef = model_cfg.mask_ce_loss_coef
        args.mask_dice_loss_coef = model_cfg.mask_dice_loss_coef
        args.mask_point_sample_ratio = model_cfg.mask_point_sample_ratio

    criterion, postprocessors = build_rfdetr_criterion(args)

    return criterion, postprocessors


def prepare_training_args(
    model_cfg: ModelConfig,
    data_cfg,
    trainer_cfg: TrainerConfig,
    optimizer_cfg: OptimizerConfig
) -> Any:
    """
    Prepare argparse-like namespace for RF-DETR's training functions.

    Args:
        model_cfg: Model configuration
        data_cfg: Data configuration
        trainer_cfg: Trainer configuration
        optimizer_cfg: Optimizer configuration

    Returns:
        Namespace with training arguments
    """
    from argparse import Namespace

    args = Namespace()

    # Model params
    args.num_classes = model_cfg.num_classes
    args.resolution = model_cfg.resolution
    args.amp = model_cfg.amp
    args.gradient_checkpointing = model_cfg.gradient_checkpointing
    args.ia_bce_loss = model_cfg.ia_bce_loss
    args.cls_loss_coef = model_cfg.cls_loss_coef
    args.group_detr = model_cfg.group_detr
    args.num_select = model_cfg.num_select
    args.patch_size = model_cfg.patch_size
    args.num_windows = model_cfg.num_windows
    args.device = model_cfg.device

    # Data params
    args.multi_scale = data_cfg.multi_scale
    args.expanded_scales = data_cfg.expanded_scales
    args.do_random_resize_via_padding = data_cfg.do_random_resize_via_padding

    # Training params
    args.epochs = trainer_cfg.epochs
    args.batch_size = trainer_cfg.batch_size
    args.grad_accum_steps = trainer_cfg.grad_accum_steps
    args.lr_drop = trainer_cfg.lr_drop
    args.warmup_epochs = trainer_cfg.warmup_epochs

    # Optimizer params
    args.lr = optimizer_cfg.lr
    args.lr_encoder = optimizer_cfg.lr_encoder
    args.weight_decay = optimizer_cfg.weight_decay
    args.lr_vit_layer_decay = optimizer_cfg.lr_vit_layer_decay
    args.lr_component_decay = optimizer_cfg.lr_component_decay
    args.drop_path = optimizer_cfg.drop_path

    # EMA
    args.use_ema = optimizer_cfg.use_ema
    args.ema_decay = optimizer_cfg.ema_decay
    args.ema_tau = optimizer_cfg.ema_tau

    # Distributed training (default to single GPU)
    args.distributed = False
    args.world_size = 1
    args.rank = 0

    return args