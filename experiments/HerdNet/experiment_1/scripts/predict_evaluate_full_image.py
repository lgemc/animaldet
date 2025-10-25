#!/usr/bin/env python3
"""Evaluate HerdNet on full-resolution images and export detections.

This script loads a trained HerdNet checkpoint, runs sliding-window inference
over a CSV-described dataset (validation or test), computes the key metrics used
in the paper (precision, recall, F1, MAE, RMSE, etc.), and saves the detections
to disk for subsequent visualization (e.g. with FiftyOne).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

from tqdm import tqdm

import albumentations as A
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from animaloc.data.transforms import DownSample
from animaloc.datasets import CSVDataset
from animaloc.eval.lmds import HerdNetLMDS
from animaloc.eval.metrics import PointsMetrics
from animaloc.eval.stitchers import HerdNetStitcher
from animaloc.models import HerdNet, LossWrapper
from animaloc.train.losses import FocalLoss


DEFAULT_CLASSES = {
    1: "Hartebeest",
    2: "Buffalo",
    3: "Kob",
    4: "Warthog",
    5: "Waterbuck",
    6: "Elephant",
}
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HerdNet on full images and export detections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--csv", type=Path, required=True, help="Annotation CSV file")
    parser.add_argument("--root", type=Path, required=True, help="Directory with corresponding images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder to store detections/metrics (defaults to checkpoint parent)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if omitted)")
    parser.add_argument("--batch-size", type=int, default=1, help="Dataloader batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--down-ratio", type=int, default=2, help="Model downsampling ratio")
    parser.add_argument("--patch-size", type=int, default=512, help="Sliding window size")
    parser.add_argument("--overlap", type=int, default=160, help="Sliding window overlap in pixels")
    parser.add_argument(
        "--upsample",
        action="store_true",
        help="Upsample stitcher outputs back to original resolution (set dataset down_ratio to 1)",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        nargs=2,
        default=(3, 3),
        help="LMDS kernel size used to identify peaks",
    )
    parser.add_argument(
        "--adapt-ts",
        type=float,
        default=0.3,
        help="Adaptive threshold for HerdNetLMDS (0.3 in the paper)",
    )
    parser.add_argument(
        "--neg-ts",
        type=float,
        default=0.1,
        help="Negative sample threshold for HerdNetLMDS",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional path to store the aggregated metrics as JSON",
    )
    parser.add_argument(
        "--detections-csv",
        type=Path,
        default=None,
        help="Optional override for detections CSV path",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def build_model(num_classes: int, device: torch.device) -> LossWrapper:
    base_model = HerdNet(
        num_classes=num_classes,
        down_ratio=2,
        num_layers=34,
        head_conv=64,
        pretrained=False,
    )

    class_weights = torch.tensor([0.1, 1.0, 2.0, 1.0, 6.0, 12.0, 1.0], dtype=torch.float32, device=device)

    losses = [
        {
            "loss": FocalLoss(reduction="mean"),
            "idx": 0,
            "idy": 0,
            "lambda": 1.0,
            "name": "focal_loss",
        },
        {
            "loss": CrossEntropyLoss(reduction="mean", weight=class_weights),
            "idx": 1,
            "idy": 1,
            "lambda": 1.0,
            "name": "ce_loss",
        },
    ]

    wrapper = LossWrapper(base_model, losses=losses)
    wrapper = wrapper.to(device)
    return wrapper


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
    fieldnames = ["images", "x", "y", "labels", "scores", "det_score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def split_stitcher_output(
    output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, (tuple, list)):
        if len(output) != 2:
            raise ValueError("Expected tuple/list of length 2 from stitcher")
        heatmap, clsmap = output
        if isinstance(heatmap, torch.Tensor) and heatmap.ndim == 3:
            heatmap = heatmap.unsqueeze(0)
        if isinstance(clsmap, torch.Tensor) and clsmap.ndim == 3:
            clsmap = clsmap.unsqueeze(0)
        return heatmap, clsmap

    if isinstance(output, torch.Tensor):
        if output.ndim == 3:
            output = output.unsqueeze(0)
        if output.shape[1] < 2:
            raise ValueError("Stitcher output tensor must have at least 2 channels")
        heatmap = output[:, :1, ...]
        clsmap = output[:, 1:, ...]
        return heatmap, clsmap

    raise TypeError(f"Unsupported stitcher output type: {type(output)}")


def normalize_image_name(value: Union[str, List[str], Tuple[str, ...]]) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return ""
        value = value[0]
    return str(value)


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint = load_checkpoint(args.checkpoint, device)
    classes = {int(k): str(v) for k, v in checkpoint.get("classes", DEFAULT_CLASSES).items()}
    mean = tuple(checkpoint.get("mean", DEFAULT_MEAN))
    std = tuple(checkpoint.get("std", DEFAULT_STD))

    num_classes = len(classes) + 1  # + background

    dataset_down_ratio = 1 if args.upsample else args.down_ratio

    dataset = CSVDataset(
        csv_file=str(args.csv),
        root_dir=str(args.root),
        albu_transforms=[A.Normalize(mean=mean, std=std, p=1.0)],
        end_transforms=[DownSample(down_ratio=dataset_down_ratio, anno_type="point")],
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_wrapper = build_model(num_classes, device)
    model_wrapper.load_state_dict(checkpoint["model_state_dict"])
    model_wrapper.eval()

    stitcher = HerdNetStitcher(
        model=model_wrapper,
        size=(args.patch_size, args.patch_size),
        overlap=args.overlap,
        down_ratio=args.down_ratio,
        reduction="mean",
        up=args.upsample,
        device_name=device.type,
    )

    lmds = HerdNetLMDS(
        up=False,
        kernel_size=tuple(args.kernel_size),
        adapt_ts=args.adapt_ts,
        neg_ts=args.neg_ts,
    )

    metrics = PointsMetrics(radius=5, num_classes=num_classes)
    detections: List[Dict[str, object]] = []

    model_wrapper.eval()
    for images, target in tqdm(dataloader, desc="Collecting detections"):
        image_name = normalize_image_name(target["image_name"][0])

        images = images.to(device)
        heatmap, clsmap = split_stitcher_output(stitcher(images[0]))

        counts, locs, labels, scores, dscores = lmds((heatmap, clsmap))

        locs = locs[0]
        labels = labels[0]
        scores = scores[0]
        dscores = dscores[0]
        counts = counts[0]

        preds_xy = [(float(col), float(row)) for row, col in locs]
        pred_labels = [int(lbl) for lbl in labels]
        pred_scores = [float(s) for s in scores]

        for (x, y), lbl, score, dscore in zip(preds_xy, pred_labels, pred_scores, dscores):
            detections.append(
                {
                    "images": image_name,
                    "x": x,
                    "y": y,
                    "labels": lbl,
                    "scores": score,
                    "det_score": float(dscore),
                }
            )

        gt_points = to_numpy_points(target["points"])
        gt_labels = to_numpy_labels(target["labels"])
        gt_coords = [(float(x), float(y)) for x, y in gt_points]

        metrics.feed(
            gt={"loc": gt_coords, "labels": gt_labels},
            preds={"loc": preds_xy, "labels": pred_labels, "scores": pred_scores},
            est_count=counts,
        )

    metrics_per_class = metrics.copy()
    metrics.aggregate()

    overall = {
        "precision": metrics.precision(),
        "recall": metrics.recall(),
        "f1_score": metrics.fbeta_score(),
        "mae": metrics.mae(),
        "rmse": metrics.rmse(),
        "mse": metrics.mse(),
        "accuracy": metrics.accuracy(),
    }

    per_class = {}
    for class_id, class_name in classes.items():
        per_class[class_name] = {
            "precision": metrics_per_class.precision(class_id),
            "recall": metrics_per_class.recall(class_id),
            "f1_score": metrics_per_class.fbeta_score(class_id),
            "mae": metrics_per_class.mae(class_id),
            "rmse": metrics_per_class.rmse(class_id),
        }

    output_dir = args.output_dir or args.checkpoint.parent / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_path = args.detections_csv or (output_dir / "detections.csv")
    export_detections(detections_path, detections)

    metrics_summary = {
        "overall": overall,
        "per_class": per_class,
        "classes": classes,
        "checkpoint": str(args.checkpoint),
        "csv": str(args.csv),
    }

    if args.metrics_json:
        metrics_path = args.metrics_json
    else:
        metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2))

    print("=== HerdNet evaluation summary ===")
    print(json.dumps(metrics_summary["overall"], indent=2))
    print("Per-class F1:")
    for name, scores in metrics_summary["per_class"].items():
        print(f"  {name:10s} -> F1: {scores['f1_score']:.3f}, Recall: {scores['recall']:.3f}, Precision: {scores['precision']:.3f}")
    print(f"\nDetections saved to: {detections_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
