#!/usr/bin/env python3
"""Visualize HerdNet patch datasets with FiftyOne.

This helper script builds lightweight FiftyOne datasets from the CSV/patch
pairs generated with ``tools/patcher.py`` (e.g. ``train_patches.csv`` and the
``train_patches`` folder). Each annotation row is converted into a single-point
`Keypoint` (or, optionally, a tiny `Detection`) to inspect labels and locations
in the FiftyOne app.

Example
-------

    python scripts/view_patches_fiftyone.py \\
        --root asg_herdnet/data-delplanque \\
        --splits train_patches val_patches \\
        --representation detections \\
        --point-radius-px 3

Use ``Ctrl+C`` in the terminal to terminate the script once you are done
exploring the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import fiftyone as fo
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and launch FiftyOne sessions for HerdNet patch datasets",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Base directory containing <split>.csv and <split> patch folders",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_patches", "val_patches"],
        help="Dataset splits to visualise (expects <split>.csv and <split>/)",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="Optional JSON file mapping label ids (int) to display names",
    )
    parser.add_argument(
        "--representation",
        choices=["keypoints", "detections"],
        default="keypoints",
        help="Visualisation mode: normalized keypoints or tiny detection boxes",
    )
    parser.add_argument(
        "--point-radius-px",
        type=float,
        default=4.0,
        help="Radius in pixels to build square detections when using 'detections' mode",
    )
    parser.add_argument(
        "--name-prefix",
        default="herdnet",
        help="Prefix to use when naming the temporary FiftyOne datasets",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Create datasets without opening the FiftyOne App",
    )
    return parser.parse_args()


def load_class_map(path: Optional[Path]) -> Dict[int, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    return {int(k): str(v) for k, v in data.items()}


def normalise_points(
    rows: Iterable[pd.Series], width: float, height: float, class_map: Dict[int, str]
) -> fo.Keypoints:
    keypoints = []
    for _, row in rows:
        label_id = int(row["labels"])
        label = class_map.get(label_id, str(label_id))
        x_norm = float(row["x"]) / width
        y_norm = float(row["y"]) / height
        keypoints.append(fo.Keypoint(label=label, points=[[x_norm, y_norm]]))
    return fo.Keypoints(keypoints=keypoints)


def to_detections(
    rows: Iterable[pd.Series],
    width: float,
    height: float,
    class_map: Dict[int, str],
    radius_px: float,
) -> fo.Detections:
    detections = []
    for _, row in rows:
        label_id = int(row["labels"])
        label = class_map.get(label_id, str(label_id))

        x = float(row["x"])
        y = float(row["y"])

        left = max(x - radius_px, 0.0)
        top = max(y - radius_px, 0.0)
        right = min(x + radius_px, width)
        bottom = min(y + radius_px, height)

        w = max(right - left, 1.0)
        h = max(bottom - top, 1.0)

        detections.append(
            fo.Detection(
                label=label,
                bounding_box=[left / width, top / height, w / width, h / height],
            )
        )
    return fo.Detections(detections=detections)


def build_dataset(
    split: str,
    root: Path,
    class_map: Dict[int, str],
    prefix: str,
    representation: str,
    radius_px: float,
) -> fo.Dataset:
    img_dir = root / split
    csv_candidates = [
        root / f"{split}.csv",
        img_dir / "gt.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)

    if csv_path is None:
        raise FileNotFoundError(
            f"Could not locate annotations for split '{split}'. "
            f"Tried: {', '.join(str(p) for p in csv_candidates)}"
        )
    if not img_dir.exists():
        raise FileNotFoundError(f"Patch directory not found: {img_dir}")

    dataset_name = f"{prefix}_{split}"
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(dataset_name)
    df = pd.read_csv(csv_path)

    grouped = df.groupby("images")
    for image_name, group in grouped:
        image_path = img_dir / image_name
        if not image_path.exists():
            print(f"[WARN] Missing image for {image_name}, skipping")
            continue

        with Image.open(image_path) as im:
            width, height = im.size

        sample = fo.Sample(filepath=str(image_path))
        sample.metadata = fo.ImageMetadata(width=width, height=height)

        annotations = None
        rows = group[["x", "y", "labels"]].iterrows()
        if representation == "keypoints":
            annotations = normalise_points(rows, width, height, class_map)
            field_name = "points"
        else:
            annotations = to_detections(
                rows, width, height, class_map, radius_px=radius_px
            )
            field_name = "detections"

        sample[field_name] = annotations

        if "base_images" in group.columns:
            sample["base_image"] = group["base_images"].iloc[0]

        dataset.add_sample(sample)

    dataset.save()
    return dataset


def main() -> None:
    args = parse_args()
    class_map = load_class_map(args.class_map)

    datasets = []
    for split in args.splits:
        dataset = build_dataset(
            split,
            args.root,
            class_map,
            args.name_prefix,
            args.representation,
            args.point_radius_px,
        )
        datasets.append(dataset)
        print(f"Created FiftyOne dataset: {dataset.name} ({len(dataset)} samples)")

    if args.no_launch:
        return

    # Launch a session with the first dataset; others are available via the UI
    session = fo.launch_app(datasets[0])
    print("FiftyOne session running. Press Ctrl+C to exit.")
    session.wait()


if __name__ == "__main__":
    main()
