"""Dataset builders for HerdNet experiments."""

import warnings
import albumentations as A
from torch.utils.data import DataLoader
from animaloc.datasets import CSVDataset
from animaloc.data.transforms import (
    MultiTransformsWrapper,
    DownSample,
    PointsToMask,
    FIDT
)

from .config import DataConfig, ModelConfig

# Suppress albumentations keypoint warnings for image-only transforms
warnings.filterwarnings('ignore', message='Got processor for keypoints')


def build_train_dataset(data_cfg: DataConfig, model_cfg: ModelConfig) -> CSVDataset:
    """
    Build training dataset with augmentations.

    Args:
        data_cfg: Data configuration
        model_cfg: Model configuration (for num_classes, down_ratio)

    Returns:
        Training dataset
    """
    albu_transforms = [
        A.VerticalFlip(p=data_cfg.vertical_flip),
        A.HorizontalFlip(p=data_cfg.horizontal_flip),
        A.RandomRotate90(p=data_cfg.rotate90),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=data_cfg.brightness_contrast
        ),
        A.Blur(blur_limit=15, p=data_cfg.blur),
        A.Normalize(p=1.0)
    ]

    end_transforms = [MultiTransformsWrapper([
        FIDT(num_classes=model_cfg.num_classes, down_ratio=model_cfg.down_ratio),
        PointsToMask(
            radius=data_cfg.radius,
            num_classes=model_cfg.num_classes,
            squeeze=True,
            down_ratio=data_cfg.mask_down_ratio
        )
    ])]

    dataset = CSVDataset(
        csv_file=data_cfg.train_csv,
        root_dir=data_cfg.train_root,
        albu_transforms=albu_transforms,
        end_transforms=end_transforms
    )

    return dataset


def build_val_dataset(data_cfg: DataConfig, model_cfg: ModelConfig) -> CSVDataset:
    """
    Build validation dataset.

    Args:
        data_cfg: Data configuration
        model_cfg: Model configuration (for down_ratio)

    Returns:
        Validation dataset
    """
    albu_transforms = [A.Normalize(p=1.0)]

    end_transforms = [DownSample(
        down_ratio=model_cfg.down_ratio,
        anno_type='point'
    )]

    dataset = CSVDataset(
        csv_file=data_cfg.val_csv,
        root_dir=data_cfg.val_root,
        albu_transforms=albu_transforms,
        end_transforms=end_transforms
    )

    return dataset


def build_test_dataset(data_cfg: DataConfig, model_cfg: ModelConfig) -> CSVDataset:
    """
    Build test dataset.

    Args:
        data_cfg: Data configuration
        model_cfg: Model configuration (for down_ratio)

    Returns:
        Test dataset
    """
    if data_cfg.test_csv is None or data_cfg.test_root is None:
        raise ValueError("Test CSV and root directory must be specified")

    albu_transforms = [A.Normalize(p=1.0)]

    end_transforms = [DownSample(
        down_ratio=model_cfg.down_ratio,
        anno_type='point'
    )]

    dataset = CSVDataset(
        csv_file=data_cfg.test_csv,
        root_dir=data_cfg.test_root,
        albu_transforms=albu_transforms,
        end_transforms=end_transforms
    )

    return dataset


def build_dataloaders(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    batch_size: int = 4
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    Build train, val, and optionally test dataloaders.

    Args:
        data_cfg: Data configuration
        model_cfg: Model configuration
        batch_size: Training batch size

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = build_train_dataset(data_cfg, model_cfg)
    val_dataset = build_val_dataset(data_cfg, model_cfg)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False
    )

    test_loader = None
    if data_cfg.test_csv is not None and data_cfg.test_root is not None:
        test_dataset = build_test_dataset(data_cfg, model_cfg)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False
        )

    return train_loader, val_loader, test_loader
