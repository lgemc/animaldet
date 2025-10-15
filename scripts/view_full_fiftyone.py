#!/usr/bin/env python3
"""Visualize full-size HerdNet datasets (ground truth + detections) in FiftyOne."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from PIL import Image
import fiftyone as fo


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
        description="Launch FiftyOne to inspect full-resolution HerdNet datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, required=True, help="Directory with full images")
    parser.add_argument("--gt-csv", type=Path, required=True, help="Ground truth CSV file")
    parser.add_argument(
        "--detections-csv",
        type=Path,
        default=None,
        help="Optional CSV file with model detections (from evaluate_full.py)",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="Optional JSON mapping of class ids to labels",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="herdnet_full",
        help="Name for the temporary FiftyOne dataset",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Create the dataset but do not open the FiftyOne App",
    )
    return parser.parse_args()


def load_class_map(path: Optional[Path]) -> Dict[int, str]:
    if path is None:
        return DEFAULT_CLASSES
    data = json.loads(path.read_text())
    return {int(k): str(v) for k, v in data.items()}


def to_keypoints(
    rows: Iterable[pd.Series],
    width: int,
    height: int,
    class_map: Dict[int, str],
) -> fo.Keypoints:
    keypoints = []
    for _, row in rows.iterrows():
        x = float(row["x"]) / width
        y = float(row["y"]) / height
        label = class_map.get(int(row["labels"]), str(row["labels"]))
        confidence = float(row["scores"]) if "scores" in row else None
        keypoints.append(fo.Keypoint(label=label, points=[[x, y]], confidence=[confidence] if confidence is not None else None))
    return fo.Keypoints(keypoints=keypoints)


def build_dataset(
    name: str,
    root: Path,
    gt: pd.DataFrame,
    detections: Optional[pd.DataFrame],
    class_map: Dict[int, str],
) -> fo.Dataset:
    if name in fo.list_datasets():
        fo.delete_dataset(name)

    dataset = fo.Dataset(name)
    grouped_gt = gt.groupby("images")
    grouped_det = detections.groupby("images") if detections is not None else None

    for image_name, gt_rows in grouped_gt:
        image_path = root / image_name
        if not image_path.exists():
            print(f"[WARN] Missing image {image_name}, skipping")
            continue

        with Image.open(image_path) as im:
            width, height = im.size

        sample = fo.Sample(filepath=str(image_path))
        sample.metadata = fo.ImageMetadata(width=width, height=height)
        sample["ground_truth"] = to_keypoints(gt_rows, width, height, class_map)

        if grouped_det is not None and image_name in grouped_det.groups:
            det_rows = grouped_det.get_group(image_name)
            sample["predictions"] = to_keypoints(det_rows, width, height, class_map)

        dataset.add_sample(sample)

    dataset.save()
    return dataset


def main() -> None:
    args = parse_args()

    class_map = load_class_map(args.class_map)
    gt_df = pd.read_csv(args.gt_csv)

    det_df = None
    if args.detections_csv is not None:
        det_df = pd.read_csv(args.detections_csv)

    dataset = build_dataset(
        name=args.dataset_name,
        root=args.root,
        gt=gt_df,
        detections=det_df,
        class_map=class_map,
    )

    print(f"Created FiftyOne dataset '{dataset.name}' with {len(dataset)} samples")
    if args.no_launch:
        return

    session = fo.launch_app(dataset)
    print("Press Ctrl+C to close the FiftyOne session.")
    session.wait()


if __name__ == "__main__":
    main()
