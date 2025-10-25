#!/usr/bin/env python3
"""Generate Hard Negative Patches (HNPs) after stage-1 HerdNet training.

This script generates ONLY hard negative patches without copying original patches.
The HNP directory can then be concatenated with the original patches in stage 2.
"""

from __future__ import annotations

import argparse
import ast
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import cv2
from PIL import Image
from torch.utils.data import DataLoader

from animaloc.data import PatchesBuffer
from animaloc.data.transforms import DownSample
from animaloc.datasets import CSVDataset
from animaloc.eval import HerdNetEvaluator, HerdNetStitcher
from animaloc.eval.metrics import PointsMetrics
from albumentations import PadIfNeeded

from predict_evaluate_full_image import (
    DEFAULT_CLASSES,
    DEFAULT_MEAN,
    DEFAULT_STD,
    build_model,
    load_checkpoint,
    split_stitcher_output,
    normalize_image_name,
)


def _normalize_image_column(value: str) -> str:
    if isinstance(value, str) and (value.startswith("[") or value.startswith("(")):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)) and parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Hard Negative Patches (HNPs) for HerdNet stage 2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage-1 checkpoint (.pth)")
    parser.add_argument("--train-csv", type=Path, required=True, help="CSV for full training images")
    parser.add_argument("--train-root", type=Path, required=True, help="Directory of full training images")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination directory for HNP patches (without original patches)",
    )
    parser.add_argument(
        "--detections-csv",
        type=Path,
        default=None,
        help="Optional path for saving raw detections CSV (defaults inside output-root)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu (auto detected)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--patch-size", type=int, default=512, help="Patcher height/width")
    parser.add_argument("--patch-overlap", type=int, default=160, help="Patch overlap for patcher")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum detection confidence to keep when creating patches",
    )
    return parser.parse_args()


def to_numpy_points(target_entry: torch.Tensor) -> List[Tuple[float, float]]:
    tensor = target_entry
    if isinstance(tensor, list):
        tensor = tensor[0]
    if isinstance(tensor, torch.Tensor):
        if tensor.ndim == 3:
            tensor = tensor.squeeze(0)
        return [tuple(map(float, coords)) for coords in tensor.tolist()]
    raise TypeError("Unexpected type for target points")


def to_numpy_labels(target_entry: torch.Tensor) -> List[int]:
    tensor = target_entry
    if isinstance(tensor, list):
        tensor = tensor[0]
    if isinstance(tensor, torch.Tensor):
        if tensor.ndim == 2:
            tensor = tensor.squeeze(0)
        return [int(x) for x in tensor.tolist()]
    raise TypeError("Unexpected type for target labels")


def export_detections(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["images", "x", "y", "labels", "scores", "det_score"]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def collect_detections(
    checkpoint: Dict,
    dataloader: DataLoader,
    device: torch.device,
    down_ratio: int,
    patch_size: int,
    overlap: int,
    kernel_size: Tuple[int, int],
    adapt_ts: float,
    neg_ts: float,
    upsample: bool,
) -> List[Dict[str, object]]:
    """Collect detections using HerdNetEvaluator (as in the original author's code).
    
    This approach aligns with the author's inference script which always uses up=True
    in the stitcher and handles LMDS efficiently internally.
    """
    classes = {int(k): str(v) for k, v in checkpoint.get("classes", DEFAULT_CLASSES).items()}
    num_classes = len(classes) + 1

    model_wrapper = build_model(num_classes, device)
    model_wrapper.load_state_dict(checkpoint["model_state_dict"])
    model_wrapper.eval()

    # Build stitcher (with up=True as the author always does)
    stitcher = HerdNetStitcher(
        model=model_wrapper,
        size=(patch_size, patch_size),
        overlap=overlap,
        down_ratio=down_ratio,
        reduction="mean",
        up=upsample,  # Always True in author's code for inference
        device_name=device.type,
    )
    
    # Build metrics (required by evaluator, but we don't use the metrics output)
    metrics = PointsMetrics(
        5,  # 5-pixel threshold as per paper (positional argument)
        num_classes=num_classes,
    )
    
    # Build evaluator (handles LMDS internally, more efficient than manual approach)
    evaluator = HerdNetEvaluator(
        model=model_wrapper,
        dataloader=dataloader,
        metrics=metrics,
        stitcher=stitcher,
        lmds_kwargs={
            "kernel_size": kernel_size,
            "adapt_ts": adapt_ts,
            "neg_ts": neg_ts,
        },
        device_name=device.type,
        print_freq=10,
        work_dir=None,  # We don't need to save outputs
        header="[HNP Generation]",
    )
    
    # Run evaluation to collect detections
    print("[INFO] Running inference with HerdNetEvaluator...")
    evaluator.evaluate(wandb_flag=False, viz=False, log_meters=False)
    
    # Extract detections from evaluator
    detections_df = evaluator.detections
    detections_df.dropna(inplace=True)
    
    # Convert to list of dicts
    detections = []
    for _, row in detections_df.iterrows():
        detections.append({
            "images": str(row["images"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "labels": int(row["labels"]),
            "scores": float(row["scores"]) if "scores" in row else 1.0,
            "det_score": float(row["det_score"]) if "det_score" in row else float(row["scores"]) if "scores" in row else 1.0,
        })
    
    print(f"[INFO] Collected {len(detections)} detections")
    return detections


def run_patcher(
    train_root: Path,
    patch_size: int,
    overlap: int,
    detections_csv: Path,
    dest_dir: Path,
    min_score: float,
) -> int:
    """Generate HNP patches from model detections.
    
    Extracts patches centered on ALL model detections (TPs + FPs).
    The gt.csv generated by PatchesBuffer should be DISCARDED - use the original
    train_patches.csv instead when training Stage 2.
    """
    detections = pd.read_csv(detections_csv)
    detections["images"] = detections["images"].apply(_normalize_image_column)
    if "scores" in detections.columns:
        detections = detections[detections["scores"] >= min_score]
    if detections.empty:
        print("[WARN] No detections found; no HNP patches will be generated.")
        return 0

    detections_path = detections_csv
    if min_score > 0:
        detections_path = detections_csv.parent / f"{detections_csv.stem}_filtered.csv"
        detections.to_csv(detections_path, index=False)

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Use PatchesBuffer to calculate patch boundaries and extract patches
    buffer = PatchesBuffer(
        str(detections_path),
        str(train_root),
        (patch_size, patch_size),
        overlap=overlap,
        min_visibility=0.0,
    ).buffer
    
    # Generate gt.csv (will be discarded - just for reference)
    buffer.drop(columns="limits").to_csv(dest_dir / "gt.csv", index=False)
    print(f"[INFO] Generated gt.csv with {len(buffer)} entries (for reference only, will be discarded)")

    source_images = detections["images"].unique()
    image_paths = [train_root / img for img in source_images]

    padder = PadIfNeeded(
        patch_size,
        patch_size,
        position=PadIfNeeded.PositionType.TOP_LEFT,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
    )

    patch_count = 0
    for img_path in tqdm(image_paths, desc="Saving HNP patches"):
        pil_img = Image.open(img_path)
        img_name = img_path.name

        img_buffer = buffer[buffer["base_images"] == img_name]
        for row in img_buffer[["images", "limits"]].to_numpy().tolist():
            patch_name, limits = row
            cropped = np.array(pil_img.crop(limits.get_tuple))
            padded = Image.fromarray(padder(image=cropped)["image"])
            padded.save(dest_dir / patch_name)
            patch_count += 1

    return patch_count


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = load_checkpoint(args.checkpoint, device)
    mean = tuple(checkpoint.get("mean", DEFAULT_MEAN))
    std = tuple(checkpoint.get("std", DEFAULT_STD))

    dataset = CSVDataset(
        csv_file=str(args.train_csv),
        root_dir=str(args.train_root),
        albu_transforms=[A.Normalize(mean=mean, std=std, p=1.0)],
        end_transforms=[DownSample(down_ratio=2, anno_type="point")],
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    detections = collect_detections(
        checkpoint=checkpoint,
        dataloader=dataloader,
        device=device,
        down_ratio=2,
        patch_size=args.patch_size,
        overlap=args.patch_overlap,
        kernel_size=(3, 3),
        adapt_ts=0.3,
        neg_ts=0.1,
        upsample=True,  # Always True: HNP mining is done on full 24MP images (paper sec 3.3.2.1)
    )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Export ALL detections (TPs + FPs mixed - this is intentional)
    detections_csv = args.detections_csv or (output_root / "detections.csv")
    export_detections(detections_csv, detections)
    print(f"[INFO] Stored detections CSV at {detections_csv} ({len(detections)} total)")
    print(f"[INFO] This includes both True Positives and False Positives (not filtered)")

    hnp_count = run_patcher(
        train_root=args.train_root,
        patch_size=args.patch_size,
        overlap=args.patch_overlap,
        detections_csv=detections_csv,
        dest_dir=output_root,
        min_score=args.min_score,
    )

    print(f"\n{'='*80}")
    print(f"[SUCCESS] Generated {hnp_count} HNP patches in {output_root}")
    print(f"[INFO] Detections CSV: {detections_csv}")
    print(f"[INFO] HNP patches: {output_root}/*.JPG")
    print(f"[INFO] gt.csv: {output_root / 'gt.csv'} (DISCARD THIS - use original train_patches.csv)")
    print(f"{'='*80}")
    print(f"\n[NEXT STEPS]:")
    print(f"1. Merge HNP patches with original training patches:")
    print(f"   cp {output_root}/*.JPG <original_train_patches>/")
    print(f"")
    print(f"2. Train Stage 2 with the ORIGINAL gt.csv (not the one from HNPs):")
    print(f"   python train_stage2.py \\")
    print(f"     --checkpoint <stage1_checkpoint> \\")
    print(f"     --train-root <original_train_patches_with_hnps> \\")
    print(f"     --train-csv <original_train_patches.csv> \\")
    print(f"     --val-csv <val.csv> \\")
    print(f"     --val-root <val> \\")
    print(f"     ...")
    print(f"")
    print(f"FolderDataset will automatically treat HNPs as background (patches not in CSV).")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
