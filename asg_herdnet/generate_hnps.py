#!/usr/bin/env python3
"""Generate Hard Negative Patches (HNPs) after stage-1 HerdNet training."""

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

from animaloc.data import ImageToPatches, PatchesBuffer, save_batch_images
from animaloc.data.transforms import DownSample
from animaloc.datasets import CSVDataset
from animaloc.eval.lmds import HerdNetLMDS
from animaloc.eval.stitchers import HerdNetStitcher
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
        "--original-patches",
        type=Path,
        required=True,
        help="Directory containing the original training patches",
    )
    parser.add_argument(
        "--original-csv",
        type=Path,
        default=None,
        help="Ground-truth CSV for the original training patches (defaults to --train-csv)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination directory that will contain merged patches (GT + HNPs)",
    )
    parser.add_argument(
        "--detections-csv",
        type=Path,
        default=None,
        help="Optional path for saving raw detections CSV (defaults inside output-root)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto detected)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--patch-size", type=int, default=512, help="Patcher height/width")
    parser.add_argument("--patch-overlap", type=int, default=0, help="Patch overlap for patcher")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum detection confidence to keep when creating patches",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete the temporary folder containing HNP-only patches",
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
) -> List[Dict[str, object]]:
    classes = {int(k): str(v) for k, v in checkpoint.get("classes", DEFAULT_CLASSES).items()}
    num_classes = len(classes) + 1

    model_wrapper = build_model(num_classes, device)
    model_wrapper.load_state_dict(checkpoint["model_state_dict"])
    model_wrapper.eval()

    stitcher = HerdNetStitcher(
        model=model_wrapper,
        size=(patch_size, patch_size),
        overlap=overlap,
        down_ratio=down_ratio,
        reduction="mean",
        up=False,
    )
    stitcher.device = device

    lmds = HerdNetLMDS(
        up=False,
        kernel_size=kernel_size,
        adapt_ts=adapt_ts,
        neg_ts=neg_ts,
    )

    detections: List[Dict[str, object]] = []

    with torch.no_grad():
        for images, target in tqdm(dataloader, desc="Collecting detections"):
            image_name = normalize_image_name(target["image_name"][0])
            images = images.to(device)
            heatmap, clsmap = split_stitcher_output(stitcher(images[0]))

            counts, locs, labels, scores, dscores = lmds((heatmap, clsmap))
            locs = locs[0]
            labels = labels[0]
            scores = scores[0]
            dscores = dscores[0]

            preds_xy = [(float(col), float(row)) for row, col in locs]
            pred_labels = [int(lbl) for lbl in labels]
            pred_scores = [float(s) for s in scores]
            pred_det_scores = [float(s) for s in dscores]

            for (x, y), lbl, score, det_score in zip(preds_xy, pred_labels, pred_scores, pred_det_scores):
                detections.append(
                    {
                        "images": image_name,
                        "x": x,
                        "y": y,
                        "labels": lbl,
                        "scores": score,
                        "det_score": det_score,
                    }
                )

    return detections


def run_patcher(
    train_root: Path,
    patch_size: int,
    overlap: int,
    detections_csv: Path,
    dest_dir: Path,
    min_score: float,
) -> None:
    detections = pd.read_csv(detections_csv)
    detections["images"] = detections["images"].apply(_normalize_image_column)
    if "scores" in detections.columns:
        detections = detections[detections["scores"] >= min_score]
    if detections.empty:
        print("[WARN] No detections found; no HNP patches will be generated.")
        return

    detections_path = detections_csv
    if min_score > 0:
        detections_path = detections_csv.parent / f"{detections_csv.stem}_filtered.csv"
        detections.to_csv(detections_path, index=False)

    dest_dir.mkdir(parents=True, exist_ok=True)

    buffer = PatchesBuffer(
        str(detections_path),
        str(train_root),
        (patch_size, patch_size),
        overlap=overlap,
        min_visibility=0.0,
    ).buffer
    buffer.drop(columns="limits").to_csv(dest_dir / "gt.csv", index=False)

    images = sorted(train_root.glob("*"))
    keep_all = False
    source_images = detections["images"].unique()
    if keep_all:
        image_paths = images
    else:
        image_paths = [train_root / img for img in source_images]

    padder = PadIfNeeded(
        patch_size,
        patch_size,
        position=PadIfNeeded.PositionType.TOP_LEFT,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
    )
    to_tensor = T.ToTensor()

    for img_path in tqdm(image_paths, desc="Saving HNP patches"):
        pil_img = Image.open(img_path)
        img_tensor = to_tensor(pil_img)
        img_name = img_path.name

        if keep_all:
                patches = ImageToPatches(img_tensor, (patch_size, patch_size), overlap=overlap).make_patches()
                save_batch_images(patches, img_name, str(dest_dir))
        else:
            img_buffer = buffer[buffer["base_images"] == img_name]
            for row in img_buffer[["images", "limits"]].to_numpy().tolist():
                patch_name, limits = row
                cropped = np.array(pil_img.crop(limits.get_tuple))
                padded = Image.fromarray(padder(image=cropped)["image"])
                padded.save(dest_dir / patch_name)


def copy_images(src_dir: Path, dst_dir: Path, ignore_suffixes: Sequence[str] = (".csv",)) -> int:
    count = 0
    for item in src_dir.iterdir():
        if item.is_dir():
            dst_sub = dst_dir / item.name
            dst_sub.mkdir(parents=True, exist_ok=True)
            count += copy_images(item, dst_sub, ignore_suffixes)
        else:
            if any(item.name.endswith(suf) for suf in ignore_suffixes):
                continue
            shutil.copy2(item, dst_dir / item.name)
            count += 1
    return count


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
    )

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    detections_csv = args.detections_csv or (output_root / "hnp_detections.csv")
    export_detections(detections_csv, detections)
    print(f"[INFO] Stored detections CSV at {detections_csv}")

    temp_dir = output_root / "hnp_temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    run_patcher(
        train_root=args.train_root,
        patch_size=args.patch_size,
        overlap=args.patch_overlap,
        detections_csv=detections_csv,
        dest_dir=temp_dir,
        min_score=args.min_score,
    )

    print(f"[INFO] Copying original patches from {args.original_patches} to {output_root}")
    shutil.copytree(args.original_patches, output_root, dirs_exist_ok=True)

    copied = copy_images(temp_dir, output_root)
    print(f"[INFO] Added {copied} hard-negative patch images")

    candidate_csvs = []
    if args.original_csv:
        candidate_csvs.append(Path(args.original_csv))
    candidate_csvs.append(args.original_patches / "gt.csv")
    candidate_csvs.append(args.original_patches.parent / f"{args.original_patches.name}.csv")
    candidate_csvs.append(Path(args.train_csv))

    original_csv = next((path for path in candidate_csvs if path.exists()), None)
    if original_csv is None:
        raise FileNotFoundError(
            "Could not locate a patch CSV. Checked: "
            + ", ".join(str(p) for p in candidate_csvs)
        )

    gt_dest = output_root / original_csv.name
    shutil.copy2(original_csv, gt_dest)
    print(f"[INFO] Copied ground-truth CSV to {gt_dest}")

    if args.keep_temp:
        print(f"[INFO] Temporary HNP patches retained in {temp_dir}")
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
