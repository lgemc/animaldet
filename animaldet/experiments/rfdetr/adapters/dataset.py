"""Dataset builders for RF-DETR experiments."""

import sys
import random
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional, List

# Add rf-detr to path
rfdetr_path = Path("/home/lmanrique/Do/rf-detr")
if str(rfdetr_path) not in sys.path:
    sys.path.insert(0, str(rfdetr_path))

from rfdetr.datasets import build_dataset
from rfdetr.util.misc import collate_fn

from .config import DataConfig, ModelConfig


class BackgroundFilteredDataset(Dataset):
    """
    Wrapper dataset that filters background images to achieve a target ratio.

    For sparse detection tasks with many background patches, this wrapper
    randomly samples background images to achieve the desired background:foreground ratio.

    Args:
        base_dataset: The underlying COCO dataset
        background_ratio: Target ratio of background:foreground images (e.g., 5.0 for 5:1)
        seed: Random seed for reproducible sampling
    """

    def __init__(self, base_dataset: Dataset, background_ratio: float, seed: int = 42):
        self.base_dataset = base_dataset
        self.background_ratio = background_ratio

        # Get COCO API from base dataset
        self.coco = base_dataset.coco

        # Identify foreground and background images
        img_ids_with_annots = set()
        for ann in self.coco.anns.values():
            img_ids_with_annots.add(ann['image_id'])

        # IMPORTANT: Use base_dataset.ids to ensure index alignment
        # The base dataset uses self.ids[idx] for lookups, so we must enumerate over it
        self.foreground_indices = []
        self.background_indices = []

        for idx, img_id in enumerate(base_dataset.ids):
            if img_id in img_ids_with_annots:
                self.foreground_indices.append(idx)
            else:
                self.background_indices.append(idx)

        # Calculate how many background images to keep
        num_foreground = len(self.foreground_indices)
        num_background_total = len(self.background_indices)
        num_background_target = int(num_foreground * background_ratio)

        # Sample background indices
        rng = random.Random(seed)
        if num_background_target < num_background_total:
            self.sampled_background_indices = rng.sample(
                self.background_indices,
                num_background_target
            )
        else:
            self.sampled_background_indices = self.background_indices

        # Combine and create final index mapping
        self.active_indices = sorted(self.foreground_indices + self.sampled_background_indices)

        # Print statistics
        actual_ratio = len(self.sampled_background_indices) / num_foreground if num_foreground > 0 else 0
        print(f"\n{'='*60}")
        print(f"Background Filtering Statistics:")
        print(f"  Foreground images: {num_foreground:,}")
        print(f"  Background images (original): {num_background_total:,}")
        print(f"  Background images (sampled): {len(self.sampled_background_indices):,}")
        print(f"  Target ratio: {background_ratio:.1f}:1")
        print(f"  Actual ratio: {actual_ratio:.1f}:1")
        print(f"  Total images: {len(self.active_indices):,}")
        print(f"  Background %: {len(self.sampled_background_indices)/len(self.active_indices)*100:.1f}%")
        print(f"{'='*60}\n")

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        # Map filtered index to original dataset index
        original_idx = self.active_indices[idx]
        return self.base_dataset[original_idx]


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

    # RF-DETR expects coco_path for COCO dataset
    args.coco_path = data_cfg.dataset_dir

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
    args.patch_size = model_cfg.patch_size
    args.num_windows = model_cfg.num_windows

    # Class names
    if data_cfg.class_names:
        args.class_names = data_cfg.class_names

    return args


def build_train_dataset(data_cfg: DataConfig, model_cfg: ModelConfig):
    """
    Build training dataset using RF-DETR's dataset builders.

    If background_ratio is specified in data_cfg, wraps the dataset with
    BackgroundFilteredDataset to sample background images.

    Args:
        data_cfg: Data configuration
        model_cfg: Model configuration

    Returns:
        Training dataset (possibly wrapped with background filtering)
    """
    args = _make_args_namespace(data_cfg, model_cfg, "train")
    dataset = build_dataset(
        image_set="train",
        args=args,
        resolution=model_cfg.resolution
    )

    # Apply background filtering if specified
    if data_cfg.background_ratio is not None:
        dataset = BackgroundFilteredDataset(
            base_dataset=dataset,
            background_ratio=data_cfg.background_ratio,
            seed=data_cfg.background_filter_seed
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
        collate_fn=collate_fn,
        drop_last=True  # Drop incomplete batches to ensure batch_size is constant for gradient accumulation
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
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
            collate_fn=collate_fn
        )
    except Exception:
        # Test set may not be available
        pass

    return train_loader, val_loader, test_loader
