#!/usr/bin/env python3
"""Stage 2 HerdNet training script (with Hard Negative Patches).

This script trains with original patches + HNPs using separate datasets
concatenated together for full control and visibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import albumentations as A
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from animaloc.data.transforms import DownSample, FIDT, MultiTransformsWrapper, PointsToMask
from animaloc.datasets import CSVDataset, FolderDataset
from animaloc.eval import HerdNetEvaluator, HerdNetStitcher, PointsMetrics
from animaloc.models import HerdNet, LossWrapper
from animaloc.train import Trainer
from animaloc.train.losses import FocalLoss
from animaloc.utils.seed import set_seed

DEFAULT_CLASS_MAP = {
    1: "Hartebeest",
    2: "Buffalo",
    3: "Kob",
    4: "Warthog",
    5: "Waterbuck",
    6: "Elephant",
}
DEFAULT_CLASS_WEIGHTS = {
    "hartebeest": 1.0,
    "buffalo": 2.0,
    "kob": 1.0,
    "warthog": 6.0,
    "waterbuck": 12.0,
    "elephant": 1.0,
}
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def parse_class_map(value: Optional[str]) -> dict[int, str]:
    if value is None:
        return DEFAULT_CLASS_MAP

    candidate = Path(value)
    if candidate.exists():
        data = json.loads(candidate.read_text())
        return {int(k): str(v) for k, v in data.items()}

    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("Provided class-map string is empty after parsing")

    return {idx + 1: name for idx, name in enumerate(tokens)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2 HerdNet training (original patches + HNPs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage-1 checkpoint to load")
    
    # Training patches (original + HNPs merged)
    parser.add_argument(
        "--train-root",
        type=Path,
        required=True,
        help="Directory with training patches (original + HNPs merged)",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        required=True,
        help="Ground-truth CSV for ORIGINAL patches only (HNPs not in CSV)",
    )
    
    # Validation
    parser.add_argument(
        "--val-csv",
        type=Path,
        required=True,
        help="CSV file for validation full images",
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        required=True,
        help="Directory containing validation full images",
    )
    
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("output/stage2"),
        help="Directory for logs and checkpoints",
    )
    parser.add_argument(
        "--epoch-count",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--valid-freq",
        type=int,
        default=1,
        help="Validate every N epochs (default: 1 = every epoch)",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate for stage 2")
    parser.add_argument(
        "--class-map",
        type=str,
        default=None,
        help="JSON file or comma-separated list of class names (background excluded)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Sliding-window patch size used for validation stitcher",
    )
    parser.add_argument(
        "--stitch-overlap",
        type=int,
        default=160,
        help="Sliding-window overlap (pixels) for validation stitcher",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project (optional)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (optional)",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="disabled",
        help="Weights & Biases mode",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="stage2",
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--skip-checkpoint-meta",
        action="store_true",
        help="Skip adding metadata to saved checkpoints",
    )
    return parser.parse_args()


def init_wandb(
    project: Optional[str],
    entity: Optional[str],
    mode: str,
    run_name: str,
    config: dict,
) -> bool:
    if not project or mode == "disabled":
        return False

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise RuntimeError("wandb is not installed but logging was requested") from exc

    wandb.init(project=project, entity=entity, mode=mode, config=config, name=run_name)
    return True


def enrich_checkpoint(
    checkpoint_path: Path,
    class_map: dict[int, str],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    stage: str,
) -> None:
    if not checkpoint_path.exists():
        print(f"[WARN] Checkpoint not found, skipping enrichment: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["classes"] = class_map
    checkpoint["mean"] = list(mean)
    checkpoint["std"] = list(std)
    checkpoint["stage"] = stage
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Enriched checkpoint metadata: {checkpoint_path.name}")


def main() -> None:
    args = parse_args()
    set_seed(9292)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.work_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load checkpoint and metadata
    # ------------------------------------------------------------------ #
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if args.class_map is None and "classes" in checkpoint:
        class_map = {int(k): str(v) for k, v in checkpoint["classes"].items()}
    else:
        class_map = parse_class_map(args.class_map)

    mean = tuple(checkpoint.get("mean", DEFAULT_MEAN))
    std = tuple(checkpoint.get("std", DEFAULT_STD))
    num_classes = len(class_map) + 1  # background + foreground classes
    patch_size = args.patch_size

    model = HerdNet(
        num_classes=num_classes,
        down_ratio=2,
        num_layers=34,
        head_conv=64,
        pretrained=False,
    )

    class_weights = [
        DEFAULT_CLASS_WEIGHTS.get(class_map[idx + 1].lower(), 1.0)
        for idx in range(len(class_map))
    ]
    weight_vector = [0.1, *class_weights]
    weight_tensor = torch.tensor(weight_vector, dtype=torch.float32, device=device)

    losses = [
        {
            "loss": FocalLoss(reduction="mean"),
            "idx": 0,
            "idy": 0,
            "lambda": 1.0,
            "name": "focal_loss",
        },
        {
            "loss": torch.nn.CrossEntropyLoss(reduction="mean", weight=weight_tensor),
            "idx": 1,
            "idy": 1,
            "lambda": 1.0,
            "name": "ce_loss",
        },
    ]

    model = LossWrapper(model, losses=losses).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # ------------------------------------------------------------------ #
    # Datasets & dataloaders
    # ------------------------------------------------------------------ #
    train_transforms = [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.Blur(blur_limit=15, p=0.2),
        A.Normalize(mean=mean, std=std, p=1.0),
    ]
    train_end_transforms = [
        MultiTransformsWrapper(
            [
                FIDT(num_classes=num_classes, add_bg=False, down_ratio=2),
                PointsToMask(
                    radius=2,
                    num_classes=num_classes,
                    squeeze=True,
                    down_ratio=int(patch_size // 16),
                ),
            ]
        )
    ]

    # Training dataset: FolderDataset automatically handles background patches
    # Patches in train_root but NOT in train_csv are treated as background
    train_dataset = FolderDataset(
        csv_file=str(args.train_csv),
        root_dir=str(args.train_root),
        albu_transforms=train_transforms,
        end_transforms=train_end_transforms,
    )
    
    print(f"[INFO] Total training samples: {len(train_dataset)}")
    print(f"[INFO] (Includes original patches with GT + HNP patches as background)")

    val_dataset = CSVDataset(
        csv_file=str(args.val_csv),
        root_dir=str(args.val_root),
        albu_transforms=[A.Normalize(mean=mean, std=std, p=1.0)],
        end_transforms=[DownSample(down_ratio=2, anno_type="point")],
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,  # Must be 1 for full-resolution validation images
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # ------------------------------------------------------------------ #
    # Evaluation utilities
    # ------------------------------------------------------------------ #
    metrics = PointsMetrics(radius=5, num_classes=num_classes)
    stitcher = HerdNetStitcher(
        model=model,
        size=(patch_size, patch_size),
        overlap=args.stitch_overlap,
        down_ratio=2,
        reduction="mean",
        up=False,
        device_name=device.type,
    )
    evaluator = HerdNetEvaluator(
        model=model,
        dataloader=val_loader,
        metrics=metrics,
        stitcher=stitcher,
        work_dir=str(args.work_dir),
        header="validation",
        device_name=device.type,
        lmds_kwargs={"kernel_size": (3, 3), "adapt_ts": 0.3},
        print_freq=10,
    )

    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=5e-4)

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        num_epochs=args.epoch_count,
        evaluator=evaluator,
        work_dir=str(args.work_dir),
        print_freq=100,
        valid_freq=args.valid_freq,
        device_name=device.type,
        auto_lr={
            "mode": "max",
            "patience": 10,
            "threshold": 1e-4,
            "threshold_mode": "rel",
            "cooldown": 10,
            "min_lr": 1e-6,
        },
    )

    wandb_flag = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        run_name=args.wandb_run_name,
        config={
            "stage": "stage2",
            "batch_size": args.batch_size,
            "val_batch_size": 1,
            "epochs": args.epoch_count,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "down_ratio": 2,
            "total_patches": len(train_dataset),
        },
    )

    trainer.start(
        warmup_iters=1,
        checkpoints="best",
        select="max",
        validate_on="f1_score",
        wandb_flag=wandb_flag,
    )

    if not args.skip_checkpoint_meta:
        best_path = args.work_dir / "best_model.pth"
        latest_path = args.work_dir / "latest_model.pth"
        enrich_checkpoint(best_path, class_map, mean, std, stage="stage2")
        enrich_checkpoint(latest_path, class_map, mean, std, stage="stage2")


if __name__ == "__main__":
    main()
