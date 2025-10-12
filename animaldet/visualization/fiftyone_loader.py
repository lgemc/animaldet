"""FiftyOne dataset loaders for animaldet datasets."""

import os
from pathlib import Path
from typing import Optional

import fiftyone as fo
import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to a standard format.

    Supports multiple formats:
    - Ungulate format: images, x_min, y_min, x_max, y_max, labels
    - HerdNet format: Image, x1, y1, x2, y2, Label
    - Point format: images/Image, x, y, labels/Label

    Args:
        df: Input dataframe with any supported column format

    Returns:
        DataFrame with normalized column names: images, labels, and either:
        - x_min, y_min, x_max, y_max (for bboxes)
        - x, y (for points)
    """
    df = df.copy()
    columns = df.columns.tolist()

    # Normalize image column name
    if 'Image' in columns:
        df.rename(columns={'Image': 'images'}, inplace=True)

    # Normalize label column name
    if 'Label' in columns:
        df.rename(columns={'Label': 'labels'}, inplace=True)

    # Normalize coordinate column names
    # HerdNet bbox format: x1, y1, x2, y2 -> x_min, y_min, x_max, y_max
    if 'x1' in columns and 'y1' in columns and 'x2' in columns and 'y2' in columns:
        df.rename(columns={
            'x1': 'x_min',
            'y1': 'y_min',
            'x2': 'x_max',
            'y2': 'y_max'
        }, inplace=True)

    # RF-DETR bbox format: x, y, x_max, y_max -> x_min, y_min, x_max, y_max
    # (x, y are top-left corner, not center points)
    elif 'x' in columns and 'y' in columns and 'x_max' in columns and 'y_max' in columns:
        df.rename(columns={
            'x': 'x_min',
            'y': 'y_min'
        }, inplace=True)

    return df


def load_ungulate_dataset(
    csv_path: str | Path,
    images_dir: str | Path,
    name: str = "ungulate",
    persistent: bool = False,
    database_uri: Optional[str] = None,
    max_samples: Optional[int] = None,
    bbox_padding_large_images: float = 0.002,
) -> fo.Dataset:
    """Load ungulate dataset with bounding box annotations into FiftyOne.

    Args:
        csv_path: Path to gt.csv file with bbox annotations
        images_dir: Directory containing the patch images
        name: Dataset name in FiftyOne
        persistent: Whether to persist dataset to database
        database_uri: MongoDB connection URI (e.g., mongodb://localhost:27017)
        max_samples: Maximum number of samples to load (None for all)
        bbox_padding_large_images: Extra padding (relative coords) for bboxes on images > 2000px wide

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
    df = normalize_column_names(df)

    # Delete existing dataset if it exists (force fresh load)
    if fo.dataset_exists(name):
        print(f"Deleting existing dataset '{name}' to force reload")
        fo.delete_dataset(name)

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

            # Add padding for better visibility on large images
            if img_w > 2000:
                padding = bbox_padding_large_images
                rel_x = max(0, rel_x - padding)
                rel_y = max(0, rel_y - padding)
                rel_w = min(1 - rel_x, rel_w + 2 * padding)
                rel_h = min(1 - rel_y, rel_h + 2 * padding)

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
    bbox_padding_large_images: float = 0.002,
    gt_csv_path: Optional[str | Path] = None,
) -> fo.Dataset:
    """Load HerdNet dataset with point annotations into FiftyOne.

    Args:
        csv_path: Path to CSV file with point annotations (predictions)
        images_dir: Directory containing the patch images
        name: Dataset name in FiftyOne
        persistent: Whether to persist dataset to database
        point_radius: Radius for converting points to bboxes for visualization (scaled for large images)
        database_uri: MongoDB connection URI (e.g., mongodb://localhost:27017)
        bbox_padding_large_images: Extra padding (relative coords) for bboxes on images > 2000px wide
        gt_csv_path: Optional path to CSV file with ground truth annotations

    Returns:
        FiftyOne dataset with keypoint detections
    """
    # Configure database URI if provided
    if database_uri:
        fo.config.database_uri = database_uri

    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    if gt_csv_path:
        gt_csv_path = Path(gt_csv_path)

    # Read CSV with point or bbox annotations (predictions)
    df = pd.read_csv(csv_path)
    df = normalize_column_names(df)

    # Read ground truth CSV if provided
    gt_df = None
    if gt_csv_path:
        gt_df = pd.read_csv(gt_csv_path)
        gt_df = normalize_column_names(gt_df)

    # Delete existing dataset if it exists (force fresh load)
    if fo.dataset_exists(name):
        print(f"Deleting existing dataset '{name}' to force reload")
        fo.delete_dataset(name)

    # Create new dataset
    dataset = fo.Dataset(name=name, persistent=persistent)
    dataset.persistent = persistent

    # Detect if we have bboxes or points in predictions
    has_bboxes = all(col in df.columns for col in ['x_min', 'y_min', 'x_max', 'y_max'])
    has_points = 'x' in df.columns and 'y' in df.columns

    # Detect if ground truth has bboxes or points
    gt_has_bboxes = False
    gt_has_points = False
    if gt_df is not None:
        gt_has_bboxes = all(col in gt_df.columns for col in ['x_min', 'y_min', 'x_max', 'y_max'])
        gt_has_points = 'x' in gt_df.columns and 'y' in gt_df.columns

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

        # Add predictions
        if has_bboxes:
            # Add detections
            detections = []
            for _, row in group.iterrows():
                x_min, y_min, x_max, y_max = row["x_min"], row["y_min"], row["x_max"], row["y_max"]

                # Convert to relative coordinates
                rel_x = x_min / img_w
                rel_y = y_min / img_h
                rel_w = (x_max - x_min) / img_w
                rel_h = (y_max - y_min) / img_h

                # Add padding for better visibility on large images
                if img_w > 2000:
                    padding = bbox_padding_large_images
                    rel_x = max(0, rel_x - padding)
                    rel_y = max(0, rel_y - padding)
                    rel_w = min(1 - rel_x, rel_w + 2 * padding)
                    rel_h = min(1 - rel_y, rel_h + 2 * padding)

                detection = fo.Detection(
                    label=str(int(row["labels"])),
                    bounding_box=[rel_x, rel_y, rel_w, rel_h],
                )
                detections.append(detection)

            # Use "predictions" field if ground truth is provided, otherwise "ground_truth"
            field_name = "predictions" if gt_df is not None else "ground_truth"
            sample[field_name] = fo.Detections(detections=detections)
        elif has_points:
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

            # Use "predictions" field if ground truth is provided, otherwise "ground_truth"
            field_name = "predictions" if gt_df is not None else "ground_truth"
            sample[field_name] = fo.Keypoints(keypoints=keypoints)

        # Add ground truth if provided
        if gt_df is not None:
            gt_group = gt_df[gt_df["images"] == image_name]

            if gt_has_bboxes:
                gt_detections = []
                for _, row in gt_group.iterrows():
                    x_min, y_min, x_max, y_max = row["x_min"], row["y_min"], row["x_max"], row["y_max"]

                    # Convert to relative coordinates
                    rel_x = x_min / img_w
                    rel_y = y_min / img_h
                    rel_w = (x_max - x_min) / img_w
                    rel_h = (y_max - y_min) / img_h

                    # Add padding for better visibility on large images
                    if img_w > 2000:
                        padding = bbox_padding_large_images
                        rel_x = max(0, rel_x - padding)
                        rel_y = max(0, rel_y - padding)
                        rel_w = min(1 - rel_x, rel_w + 2 * padding)
                        rel_h = min(1 - rel_y, rel_h + 2 * padding)

                    gt_detection = fo.Detection(
                        label=str(int(row["labels"])),
                        bounding_box=[rel_x, rel_y, rel_w, rel_h],
                    )
                    gt_detections.append(gt_detection)

                sample["ground_truth"] = fo.Detections(detections=gt_detections)
            elif gt_has_points:
                gt_keypoints = []
                for _, row in gt_group.iterrows():
                    x, y = row["x"], row["y"]

                    # Convert to relative coordinates
                    rel_x = x / img_w
                    rel_y = y / img_h

                    gt_keypoint = fo.Keypoint(
                        label=str(int(row["labels"])),
                        points=[(rel_x, rel_y)],
                    )
                    gt_keypoints.append(gt_keypoint)

                sample["ground_truth"] = fo.Keypoints(keypoints=gt_keypoints)

        samples.append(sample)

    dataset.add_samples(samples)

    # Print summary
    if gt_df is not None:
        if has_bboxes:
            print(f"Loaded {len(dataset)} images with {sum(len(s['predictions'].detections) for s in dataset)} predicted detections")
        else:
            print(f"Loaded {len(dataset)} images with {sum(len(s['predictions'].keypoints) for s in dataset)} predicted keypoints")

        if gt_has_bboxes:
            print(f"Loaded {sum(len(s['ground_truth'].detections) for s in dataset)} ground truth detections")
        else:
            print(f"Loaded {sum(len(s['ground_truth'].keypoints) for s in dataset)} ground truth keypoints")
    else:
        if has_bboxes:
            print(f"Loaded {len(dataset)} images with {sum(len(s['ground_truth'].detections) for s in dataset)} detections")
        else:
            print(f"Loaded {len(dataset)} images with {sum(len(s['ground_truth'].keypoints) for s in dataset)} keypoints")

    return dataset
