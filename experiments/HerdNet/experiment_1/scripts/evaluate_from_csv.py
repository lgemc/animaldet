#!/usr/bin/env python3
"""Compute HerdNet metrics directly from ground truth and detections CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from animaloc.eval.metrics import PointsMetrics


DEFAULT_CLASSES = {
    1: "Hartebeest",
    2: "Buffalo",
    3: "Kob",
    4: "Warthog",
    5: "Waterbuck",
    6: "Elephant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detection CSVs against ground truth CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gt-csv", type=Path, required=True, help="Ground truth CSV (images,x,y,labels)")
    parser.add_argument("--detections-csv", type=Path, required=True, help="Detections CSV (images,x,y,labels,scores)")
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="Optional JSON mapping of class ids to labels",
    )
    parser.add_argument("--radius", type=float, default=5.0, help="Matching radius in pixels")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to store metrics as JSON",
    )
    return parser.parse_args()


def load_class_map(path: Optional[Path]) -> Dict[int, str]:
    if path is None:
        return DEFAULT_CLASSES
    data = json.loads(path.read_text())
    return {int(k): str(v) for k, v in data.items()}


def extract_points(rows: pd.DataFrame) -> Tuple[List[Tuple[float, float]], List[int], List[float]]:
    coords = [(float(row["x"]), float(row["y"])) for _, row in rows.iterrows()]
    labels = [int(row["labels"]) for _, row in rows.iterrows()]
    scores = [float(row["scores"]) for _, row in rows.iterrows()] if "scores" in rows.columns else []
    return coords, labels, scores


def main() -> None:
    args = parse_args()
    class_map = load_class_map(args.class_map)
    num_classes = len(class_map) + 1

    gt_df = pd.read_csv(args.gt_csv)
    det_df = pd.read_csv(args.detections_csv)

    metrics = PointsMetrics(radius=args.radius, num_classes=num_classes)
    per_class_metrics = metrics.copy()

    all_images = sorted(set(gt_df["images"]) | set(det_df["images"]))

    for image_name in all_images:
        gt_rows = gt_df[gt_df["images"] == image_name]
        det_rows = det_df[det_df["images"] == image_name]

        gt_coords, gt_labels, _ = extract_points(gt_rows)
        pred_coords, pred_labels, pred_scores = extract_points(det_rows)

        counts = [pred_labels.count(cls_id) for cls_id in range(1, num_classes)]

        metrics.feed(
            gt={"loc": gt_coords, "labels": gt_labels},
            preds={"loc": pred_coords, "labels": pred_labels, "scores": pred_scores},
            est_count=counts,
        )

    per_class_metrics = metrics.copy()
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
    for class_id, class_name in class_map.items():
        per_class[class_name] = {
            "precision": per_class_metrics.precision(class_id),
            "recall": per_class_metrics.recall(class_id),
            "f1_score": per_class_metrics.fbeta_score(class_id),
            "mae": per_class_metrics.mae(class_id),
            "rmse": per_class_metrics.rmse(class_id),
        }

    summary = {
        "overall": overall,
        "per_class": per_class,
        "classes": class_map,
        "gt_csv": str(args.gt_csv),
        "detections_csv": str(args.detections_csv),
        "radius": args.radius,
    }

    print("=== Metrics from CSVs ===")
    print(json.dumps(summary["overall"], indent=2))
    print("Per-class F1:")
    for name, scores in summary["per_class"].items():
        print(
            f"  {name:10s} -> F1: {scores['f1_score']:.3f}, "
            f"Recall: {scores['recall']:.3f}, Precision: {scores['precision']:.3f}"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
