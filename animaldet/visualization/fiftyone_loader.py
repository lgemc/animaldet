"""FiftyOne dataset loaders for animaldet datasets."""

import os
from pathlib import Path
from typing import Optional

import fiftyone as fo
import pandas as pd


def load_ungulate_dataset(
    csv_path: str | Path,
    images_dir: str | Path,
    name: str = "ungulate",
    persistent: bool = False,
    database_uri: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> fo.Dataset:
    """Load ungulate dataset with bounding box annotations into FiftyOne.

    Args:
        csv_path: Path to gt.csv file with bbox annotations
        images_dir: Directory containing the patch images
        name: Dataset name in FiftyOne
        persistent: Whether to persist dataset to database
        database_uri: MongoDB connection URI (e.g., mongodb://localhost:27017)
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        FiftyOne dataset with bbox detections
    """
    # Configure database URI if provided
    if database_uri:
        fo.config.database_uri = database_uri

    csv_path = Path(csv_path)
    images_dir = Path(images_dir)

    # Read CSV with bbox annotations
    df = pd.read_csv(csv_path)

    # Check if dataset exists
    if fo.dataset_exists(name):
        print(f"Loading existing dataset '{name}'")
        return fo.load_dataset(name)

    # Create new dataset
    dataset = fo.Dataset(name=name, persistent=persistent)
    dataset.persistent = persistent

    # Group by image
    samples = []
    for idx, (image_name, group) in enumerate(df.groupby("images")):
        if max_samples is not None and idx >= max_samples:
            break
        image_path = images_dir / image_name

        if not image_path.exists():
            continue

        sample = fo.Sample(filepath=str(image_path))

        # Add detections
        detections = []
        for _, row in group.iterrows():
            # Convert to relative coordinates [0, 1]
            # FiftyOne expects [x_top_left, y_top_left, width, height] in relative coords
            x_min, y_min, x_max, y_max = row["x_min"], row["y_min"], row["x_max"], row["y_max"]

            # Get image dimensions from first sample
            if not hasattr(sample, "_image_dims"):
                import PIL.Image

                with PIL.Image.open(image_path) as img:
                    img_w, img_h = img.size
                sample._image_dims = (img_w, img_h)
            else:
                img_w, img_h = sample._image_dims

            # Convert to relative coordinates
            rel_x = x_min / img_w
            rel_y = y_min / img_h
            rel_w = (x_max - x_min) / img_w
            rel_h = (y_max - y_min) / img_h

            detection = fo.Detection(
                label=str(int(row["labels"])),
                bounding_box=[rel_x, rel_y, rel_w, rel_h],
            )
            detections.append(detection)

        sample["ground_truth"] = fo.Detections(detections=detections)
        samples.append(sample)

    dataset.add_samples(samples)

    print(f"Loaded {len(dataset)} images with {sum(len(s['ground_truth'].detections) for s in dataset)} detections")
    return dataset


def load_herdnet_dataset(
    csv_path: str | Path,
    images_dir: str | Path,
    name: str = "herdnet",
    persistent: bool = False,
    point_radius: int = 10,
    database_uri: Optional[str] = None,
) -> fo.Dataset:
    """Load HerdNet dataset with point annotations into FiftyOne.

    Args:
        csv_path: Path to CSV file with point annotations
        images_dir: Directory containing the patch images
        name: Dataset name in FiftyOne
        persistent: Whether to persist dataset to database
        point_radius: Radius for converting points to bboxes for visualization
        database_uri: MongoDB connection URI (e.g., mongodb://localhost:27017)

    Returns:
        FiftyOne dataset with keypoint detections
    """
    # Configure database URI if provided
    if database_uri:
        fo.config.database_uri = database_uri

    csv_path = Path(csv_path)
    images_dir = Path(images_dir)

    # Read CSV with point annotations
    df = pd.read_csv(csv_path)

    # Check if dataset exists
    if fo.dataset_exists(name):
        print(f"Loading existing dataset '{name}'")
        return fo.load_dataset(name)

    # Create new dataset
    dataset = fo.Dataset(name=name, persistent=persistent)
    dataset.persistent = persistent

    # Group by image
    samples = []
    for image_name, group in df.groupby("images"):
        image_path = images_dir / image_name

        if not image_path.exists():
            continue

        sample = fo.Sample(filepath=str(image_path))

        # Get image dimensions
        import PIL.Image

        with PIL.Image.open(image_path) as img:
            img_w, img_h = img.size

        # Add keypoints
        keypoints = []
        for _, row in group.iterrows():
            x, y = row["x"], row["y"]

            # Convert to relative coordinates
            rel_x = x / img_w
            rel_y = y / img_h

            keypoint = fo.Keypoint(
                label=str(int(row["labels"])),
                points=[(rel_x, rel_y)],
            )
            keypoints.append(keypoint)

        sample["ground_truth"] = fo.Keypoints(keypoints=keypoints)
        samples.append(sample)

    dataset.add_samples(samples)

    print(f"Loaded {len(dataset)} images with {sum(len(s['ground_truth'].keypoints) for s in dataset)} keypoints")
    return dataset
