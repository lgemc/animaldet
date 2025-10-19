#!/usr/bin/env python3
"""Stage 1 training script for HerdNet on patch datasets.

This follows the first training phase described in Delplanque et al. (2023):
 - Training on 512x512 patches generated from the full-resolution images
 - Batch size 4, learning rate 1e-4, weight decay 5e-4
 - F1-score (threshold 5 px) used for checkpoint selection
 - Hard Negative Patches are *not* included at this stage

Usage
-----

    python scripts/train_stage1.py \
        --root asg_herdnet/data-delplanque \
        --work-dir output/stage1 \
        --wandb-mode disabled
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import json

import albumentations as A
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from animaloc.data.transforms import (
    DownSample,
    FIDT,
    MultiTransformsWrapper,
    PointsToMask,
)
from animaloc.datasets import CSVDataset
from animaloc.eval import HerdNetEvaluator, HerdNetStitcher, PointsMetrics
from animaloc.models import HerdNet, LossWrapper
from animaloc.train import Trainer
from animaloc.train.losses import FocalLoss
from animaloc.utils.seed import set_seed

DEFAULT_CLASS_WEIGHTS = {
    "hartebeest": 1.0,
    "buffalo": 2.0,
    "kob": 1.0,
    "warthog": 6.0,
    "waterbuck": 12.0,
    "elephant": 1.0,
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 HerdNet training (patches only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("asg_herdnet/data-delplanque"),
        help="Root directory containing train/val patch folders",
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=None,
        help="Override train patch directory (defaults to <root>/train_patches)",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Override train CSV (defaults to <train-root>/gt.csv)",
    )
    parser.add_argument(
        "--val-root",
        type=Path,
        default=None,
        help="Override validation patch directory (defaults to <root>/val_patches)",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=None,
        help="Override validation CSV (defaults to <val-root>/gt.csv)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("output/stage1"),
        help="Directory where checkpoints and logs will be written",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--valid-freq",
        type=int,
        default=1,
        help="Validate every N epochs (default: 1 = every epoch)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataloader worker count",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name (leave empty to disable logging)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/user (optional)",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="disabled",
        help="Weights & Biases init mode",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="stage1",
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--pretrained-backbone",
        type=str,
        default="dla34.in1k",
        help="timm model id to load DLA backbone weights (set to 'none' to disable)",
    )
    parser.add_argument(
        "--class-map",
        type=str,
        default=None,
        help="JSON file or comma-separated list of class names (background excluded)",
    )
    parser.add_argument(
        "--skip-checkpoint-meta",
        action="store_true",
        help="Do not inject metadata (classes/mean/std) into saved checkpoints",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    train_root = args.train_root or (args.root / "train_patches")
    val_root = args.val_root or (args.root / "val_patches")

    def resolve_csv(explicit: Optional[Path], candidates: list[Path], label: str) -> Path:
        if explicit:
            if not explicit.exists():
                raise FileNotFoundError(f"{label} not found: {explicit}")
            return explicit
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"{label} not found. Tried: {', '.join(str(p) for p in candidates)}"
        )

    train_csv = resolve_csv(
        args.train_csv,
        [train_root / "gt.csv", args.root / "train_patches.csv"],
        "train csv",
    )
    val_csv = resolve_csv(
        args.val_csv,
        [val_root / "gt.csv", args.root / "val_patches.csv"],
        "validation csv",
    )

    for path, label in [
        (train_root, "train root"),
        (train_csv, "train csv"),
        (val_root, "validation root"),
        (val_csv, "validation csv"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    return train_root, train_csv, val_root, val_csv


def init_wandb(
    project: Optional[str],
    entity: Optional[str],
    mode: str,
    config: dict,
    run_name: str,
) -> bool:
    if not project or mode == "disabled":
        return False

    try:
        import wandb
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "wandb is not installed but logging was requested"
        ) from exc

    wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        config=config,
        name=run_name,
    )
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def load_backbone_from_timm(base_module: torch.nn.Module, model_id: str) -> None:
    if model_id.lower() == "none":
        print("[INFO] Skipping timm backbone loading (pretrained-backbone=none)")
        return

    try:
        import timm
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "timm is required to load pretrained backbone weights"
        ) from exc

    print(f"[INFO] Loading backbone weights from timm model '{model_id}'")
    try:
        timm_model = timm.create_model(model_id, pretrained=True)
    except Exception as exc:  # pragma: no cover - network/device failures
        raise RuntimeError(
            f"Unable to instantiate timm model '{model_id}'. "
            "Check your network connectivity or provide a cached checkpoint."
        ) from exc

    state_dict = timm_model.state_dict()
    missing, unexpected = base_module.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[WARN] Missing keys while loading backbone: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading backbone: {unexpected}")

    del timm_model


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


def parse_class_map(arg: Optional[str]) -> dict[int, str]:
    default = {
        1: "Hartebeest",
        2: "Buffalo",
        3: "Kob",
        4: "Warthog",
        5: "Waterbuck",
        6: "Elephant",
    }
    if arg is None:
        return default

    candidate = Path(arg)
    if candidate.exists():
        data = json.loads(candidate.read_text())
        return {int(k): str(v) for k, v in data.items()}

    names = [token.strip() for token in arg.split(",") if token.strip()]
    if not names:
        raise ValueError("Provided class-map string is empty after parsing")

    return {idx + 1: name for idx, name in enumerate(names)}


def main() -> None:
    args = parse_args()
    set_seed(9292)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = 512
    num_classes = len(class_map) + 1  # + background
    down_ratio = 2
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    class_map = parse_class_map(args.class_map)

    train_root, train_csv, val_root, val_csv = resolve_paths(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Datasets & dataloaders
    # ------------------------------------------------------------------ #

    train_transforms = [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.2
        ),
        A.Blur(blur_limit=15, p=0.2),
        A.Normalize(mean=mean, std=std, p=1.0),
    ]
    train_end_transforms = [
        MultiTransformsWrapper(
            [
                FIDT(num_classes=2, add_bg=False, down_ratio=down_ratio),
                PointsToMask(
                    radius=2,
                    num_classes=num_classes,
                    squeeze=True,
                    down_ratio=32,
                ),
            ]
        )
    ]

    val_transforms = [A.Normalize(mean=mean, std=std, p=1.0)]
    val_end_transforms = [DownSample(down_ratio=down_ratio, anno_type="point")]

    train_dataset = CSVDataset(
        csv_file=str(train_csv),
        root_dir=str(train_root),
        albu_transforms=train_transforms,
        end_transforms=train_end_transforms,
    )
    val_dataset = CSVDataset(
        csv_file=str(val_csv),
        root_dir=str(val_root),
        albu_transforms=val_transforms,
        end_transforms=val_end_transforms,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Must be 1 for full-resolution validation images
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # ------------------------------------------------------------------ #
    # Model, losses, optimiser
    # ------------------------------------------------------------------ #

    model = HerdNet(
        num_classes=num_classes,
        down_ratio=down_ratio,
        num_layers=34,
        head_conv=64,
        pretrained=False,
    ).to(device)

    load_backbone_from_timm(model.base_0, args.pretrained_backbone)

    per_class_weights = [
        DEFAULT_CLASS_WEIGHTS.get(class_map[idx + 1].lower(), 1.0)
        for idx in range(len(class_map))
    ]
    weight_vector = [0.1, *per_class_weights]
    class_weights = torch.tensor(weight_vector, dtype=torch.float32, device=device)
    losses = [
        {
            "loss": FocalLoss(reduction="mean", normalize=False),
            "idx": 0,
            "idy": 0,
            "lambda": 1.0,
            "name": "focal_loss",
        },
        {
            "loss": torch.nn.CrossEntropyLoss(
                reduction="mean", weight=class_weights
            ),
            "idx": 1,
            "idy": 1,
            "lambda": 1.0,
            "name": "ce_loss",
        },
    ]
    model = LossWrapper(model, losses=losses).to(device)

    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

    # ------------------------------------------------------------------ #
    # Evaluation utilities
    # ------------------------------------------------------------------ #

    metrics = PointsMetrics(radius=5, num_classes=num_classes)
    stitcher = HerdNetStitcher(
        model=model,
        size=(patch_size, patch_size),
        overlap=160,
        down_ratio=down_ratio,
        reduction="mean",
        up=False,
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

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        num_epochs=args.epochs,
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

    # Optional WandB logging ------------------------------------------------ #
    wandb_flag = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config={
            "stage": "stage1",
            "batch_size": args.batch_size,
            "val_batch_size": 1,
            "epochs": args.epochs,
            "lr": 1e-4,
            "weight_decay": 5e-4,
            "num_workers": args.num_workers,
            "down_ratio": down_ratio,
        },
        run_name=args.wandb_run_name,
    )

    trainer.start(
        warmup_iters=100,
        checkpoints="best",
        select="max",
        validate_on="f1_score",
        wandb_flag=wandb_flag,
    )

    best_path = args.work_dir / "best_model.pth"
    latest_path = args.work_dir / "latest_model.pth"
    enrich_checkpoint(best_path, class_map, mean, std, stage="stage1")
    enrich_checkpoint(latest_path, class_map, mean, std, stage="stage1")


if __name__ == "__main__":
    main()
