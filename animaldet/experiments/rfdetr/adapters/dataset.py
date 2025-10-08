"""Dataset builders for RF-DETR experiments."""

import sys
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Tuple, Optional

# Add rf-detr to path
rfdetr_path = Path("/home/lmanrique/Do/rf-detr")
if str(rfdetr_path) not in sys.path:
    sys.path.insert(0, str(rfdetr_path))

from rfdetr.datasets import build_dataset

from .config import DataConfig, ModelConfig


def _make_args_namespace(data_cfg: DataConfig, model_cfg: ModelConfig, image_set: str):
    """
    Create an argparse-like namespace for rfdetr's dataset builders.

    Args:
        data_cfg: Data configuration
        model_cfg: Model configuration
        image_set: 'train', 'val', or 'test'

    Returns:
        Namespace object compatible with rfdetr's build_dataset function
    """
    from argparse import Namespace

    args = Namespace()

    # Dataset settings
    args.dataset_file = data_cfg.dataset_file
    args.dataset_dir = data_cfg.dataset_dir

    # For COCO format datasets
    if data_cfg.train_annotation:
        args.train_annotation = data_cfg.train_annotation
    if data_cfg.val_annotation:
        args.val_annotation = data_cfg.val_annotation

    # Augmentation settings
    args.multi_scale = data_cfg.multi_scale
    args.expanded_scales = data_cfg.expanded_scales
    args.do_random_resize_via_padding = data_cfg.do_random_resize_via_padding
    args.square_resize_div_64 = data_cfg.square_resize_div_64

    # Model settings needed for transforms
    args.num_classes = model_cfg.num_classes

    # Class names
    if data_cfg.class_names:
        args.class_names = data_cfg.class_names

    return args


def build_train_dataset(data_cfg: DataConfig, model_cfg: ModelConfig):
    """
    Build training dataset using RF-DETR's dataset builders.

    Args:
        data_cfg: Data configuration
        model_cfg: Model configuration

    Returns:
        Training dataset
    """
    args = _make_args_namespace(data_cfg, model_cfg, "train")
    dataset = build_dataset(
        image_set="train",
        args=args,
        resolution=model_cfg.resolution
    )
    return dataset


def build_val_dataset(data_cfg: DataConfig, model_cfg: ModelConfig):
    """
    Build validation dataset using RF-DETR's dataset builders.

    Args:
        data_cfg: Data configuration
        model_cfg: Model configuration

    Returns:
        Validation dataset
    """
    args = _make_args_namespace(data_cfg, model_cfg, "val")
    dataset = build_dataset(
        image_set="val",
        args=args,
        resolution=model_cfg.resolution
    )
    return dataset


def build_test_dataset(data_cfg: DataConfig, model_cfg: ModelConfig):
    """
    Build test dataset using RF-DETR's dataset builders.

    Args:
        data_cfg: Data configuration
        model_cfg: Model configuration

    Returns:
        Test dataset
    """
    args = _make_args_namespace(data_cfg, model_cfg, "test")
    dataset = build_dataset(
        image_set="test",
        args=args,
        resolution=model_cfg.resolution
    )
    return dataset


def build_dataloaders(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    batch_size: int = 4,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Build train, val, and optionally test dataloaders.

    Args:
        data_cfg: Data configuration
        model_cfg: Model configuration
        batch_size: Training batch size
        num_workers: Number of dataloader workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = build_train_dataset(data_cfg, model_cfg)
    val_dataset = build_val_dataset(data_cfg, model_cfg)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: x  # RF-DETR uses list collation
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x
    )

    # Test loader is optional
    test_loader = None
    try:
        test_dataset = build_test_dataset(data_cfg, model_cfg)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x
        )
    except Exception:
        # Test set may not be available
        pass

    return train_loader, val_loader, test_loader
