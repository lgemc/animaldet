#!/usr/bin/env python3
"""
Create COCO annotation file from CSV that includes ALL images (with and without annotations).

This script is similar to HerdNet's FolderDataset approach - it finds all images in a directory
and creates entries for them, even if they don't have annotations (background images).
"""

import json
import os
import argparse
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm


def create_coco_from_csv_with_background(
    csv_path: str,
    image_dir: str,
    output_path: str,
    category_names: list[str] = None
):
    """
    Create COCO annotation file from CSV, including all images in directory.

    Args:
        csv_path: Path to CSV file with annotations (images, x_min, y_min, x_max, y_max, labels)
        image_dir: Directory containing all images (including background images)
        output_path: Output path for COCO JSON file
        category_names: List of category names (optional)
    """

    # Read CSV annotations
    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Get all image files in directory
    print(f"Scanning directory {image_dir}...")
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'))
    ])

    print(f"Found {len(image_files)} total images")
    print(f"Found {df['images'].nunique()} images with annotations")
    print(f"Found {len(image_files) - df['images'].nunique()} background images")

    # Initialize COCO structure
    coco = {
        "info": {
            "description": "Animal detection dataset with background images",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create categories
    if category_names:
        categories = category_names
    else:
        # Auto-detect from CSV labels
        unique_labels = sorted(df['labels'].unique())
        categories = [f"category_{i}" for i in unique_labels]

    for cat_id, cat_name in enumerate(categories, start=1):
        coco["categories"].append({
            "id": cat_id,
            "name": cat_name,
            "supercategory": "animal"
        })

    print(f"Categories: {categories}")

    # Create image entries for ALL images
    image_id_map = {}
    print("Creating image entries...")
    for img_id, img_name in enumerate(tqdm(image_files), start=1):
        img_path = os.path.join(image_dir, img_name)

        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Warning: Could not read {img_name}: {e}")
            continue

        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        image_id_map[img_name] = img_id

    # Create annotations (only for images with annotations)
    print("Creating annotations...")
    ann_id = 1
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row['images']

        if img_name not in image_id_map:
            print(f"Warning: Image {img_name} in CSV but not in directory")
            continue

        img_id = image_id_map[img_name]

        # Convert to COCO format (x, y, width, height)
        x_min = row['x_min']
        y_min = row['y_min']
        x_max = row['x_max']
        y_max = row['y_max']

        width = x_max - x_min
        height = y_max - y_min

        # Skip invalid boxes
        if width <= 0 or height <= 0:
            continue

        category_id = int(row['labels'])

        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": category_id,
            "bbox": [float(x_min), float(y_min), float(width), float(height)],
            "area": float(width * height),
            "iscrowd": 0
        })

        ann_id += 1

    # Save COCO JSON
    print(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    # Print statistics
    print(f"\nCOCO annotation file created successfully!")
    print(f"  Total images: {len(coco['images'])}")
    print(f"  Images with annotations: {len(set(ann['image_id'] for ann in coco['annotations']))}")
    print(f"  Background images: {len(coco['images']) - len(set(ann['image_id'] for ann in coco['annotations']))}")
    print(f"  Total annotations: {len(coco['annotations'])}")
    print(f"  Categories: {len(coco['categories'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Create COCO annotation file including background images"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file with annotations"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing all images"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for COCO JSON file"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Category names (optional)"
    )

    args = parser.parse_args()

    create_coco_from_csv_with_background(
        csv_path=args.csv,
        image_dir=args.image_dir,
        output_path=args.output,
        category_names=args.categories
    )


if __name__ == "__main__":
    main()
