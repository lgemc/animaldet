#!/usr/bin/env python3
"""Convert ungulate dataset format to COCO format for RF-DETR.

This script converts the ungulate dataset (CSV format with patches) to COCO JSON format
that can be used with RF-DETR training.

Input format (CSV):
    images,base_images,x,y,x_min,y_min,x_max,y_max,labels

Output format (COCO JSON):
    {
        "images": [...],
        "annotations": [...],
        "categories": [...]
    }
"""

import json
import csv
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import argparse


def convert_ungulate_to_coco(
    patches_dir: Path,
    gt_csv: Path,
    output_json: Path,
    preserve_species: bool = True
) -> None:
    """Convert ungulate dataset to COCO format.

    Args:
        patches_dir: Directory containing patch images
        gt_csv: Path to ground truth CSV file
        output_json: Output path for COCO JSON file
        preserve_species: If True, preserve individual species as categories.
                         If False, collapse all to single "ungulate" class.
    """
    # First pass: collect all unique labels from the data
    unique_labels = set()
    with open(gt_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            unique_labels.add(int(row['labels']))

    # Create species mapping: original label -> 0-indexed category
    # Sort to ensure consistent ordering
    sorted_labels = sorted(unique_labels)
    label_to_category = {label: idx for idx, label in enumerate(sorted_labels)}

    # Initialize COCO structure with categories
    if preserve_species:
        # Create separate category for each species
        # COCO category IDs are 0-indexed
        categories = [
            {
                "id": category_id,
                "name": f"species_{original_label}",
                "supercategory": "ungulate"
            }
            for original_label, category_id in sorted(label_to_category.items(), key=lambda x: x[1])
        ]
    else:
        # Single "ungulate" category
        categories = [
            {
                "id": 0,
                "name": "ungulate",
                "supercategory": "animal"
            }
        ]

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    # Track image IDs
    image_id_map: Dict[str, int] = {}
    next_image_id = 1
    next_annotation_id = 1

    # Group annotations by image
    annotations_by_image = defaultdict(list)

    # Read CSV file
    with open(gt_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row['images']
            x_min = float(row['x_min'])
            y_min = float(row['y_min'])
            x_max = float(row['x_max'])
            y_max = float(row['y_max'])
            label = int(row['labels'])

            # Calculate bbox in COCO format [x, y, width, height]
            width = x_max - x_min
            height = y_max - y_min

            # Map label to category_id
            if preserve_species:
                # Use species-specific category (0-indexed)
                cat_id = label_to_category[label]
            else:
                # Use single "ungulate" category
                cat_id = 0

            # Create annotation
            annotation = {
                "bbox": [x_min, y_min, width, height],
                "category_id": cat_id
            }

            annotations_by_image[image_name].append(annotation)

    # Create images and annotations
    for image_name, annotations in sorted(annotations_by_image.items()):
        # Check if image exists
        image_path = patches_dir / image_name
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        # Assign image ID
        image_id = next_image_id
        image_id_map[image_name] = image_id
        next_image_id += 1

        # Add image entry
        # Assume all images are 560x560 based on the ungulate dataset
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "height": 560,
            "width": 560
        })

        # Add annotations for this image
        for ann in annotations:
            coco_data["annotations"].append({
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["bbox"][2] * ann["bbox"][3],
                "iscrowd": 0
            })
            next_annotation_id += 1

    # Write output JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)

    # Print statistics
    print(f"\nConversion complete!")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")

    # Category distribution
    if preserve_species:
        from collections import Counter
        cat_counts = Counter(ann['category_id'] for ann in coco_data['annotations'])
        print(f"\nCategory distribution:")
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        for cat_id, count in sorted(cat_counts.items()):
            print(f"  {cat_id_to_name[cat_id]:>10s} (id={cat_id}): {count:>6d} annotations")

    print(f"\nOutput saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ungulate dataset to COCO format for RF-DETR"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Dataset split to convert"
    )
    parser.add_argument(
        "--ungulate-root",
        type=str,
        default="data/ungulate",
        help="Root directory of ungulate dataset"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/rfdetr/ungulate",
        help="Output root directory for COCO format"
    )
    parser.add_argument(
        "--preserve-species",
        action="store_true",
        default=True,
        help="Preserve individual species as separate categories (default: True)"
    )
    parser.add_argument(
        "--single-class",
        action="store_true",
        help="Collapse all species into single 'ungulate' class"
    )

    args = parser.parse_args()

    # Set up paths
    ungulate_root = Path(args.ungulate_root)
    output_root = Path(args.output_root)

    # Determine split-specific paths
    split_suffix = "val" if args.split == "val" else args.split
    patches_dir = ungulate_root / f"{split_suffix}_patches"
    gt_csv = patches_dir / "gt.csv"

    # Output paths following COCO standard naming convention
    # RF-DETR expects: train2017, val2017, test2017
    output_images_dir = output_root / f"{args.split}2017"
    # RF-DETR expects: instances_train2017.json, instances_val2017.json, image_info_test-dev2017.json
    if args.split == "test":
        output_json = output_root / f"annotations/image_info_test-dev2017.json"
    else:
        output_json = output_root / f"annotations/instances_{args.split}2017.json"

    # Verify input exists
    if not patches_dir.exists():
        print(f"Error: Patches directory not found: {patches_dir}")
        return

    if not gt_csv.exists():
        print(f"Error: Ground truth CSV not found: {gt_csv}")
        return

    # Determine whether to preserve species
    preserve_species = not args.single_class

    print(f"Converting {args.split} split...")
    print(f"  Patches dir: {patches_dir}")
    print(f"  GT CSV: {gt_csv}")
    print(f"  Output JSON: {output_json}")
    print(f"  Preserve species: {preserve_species}")

    # Convert to COCO format
    convert_ungulate_to_coco(
        patches_dir=patches_dir,
        gt_csv=gt_csv,
        output_json=output_json,
        preserve_species=preserve_species
    )

    # Copy or symlink images to output directory
    output_images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nNext steps:")
    print(f"1. Copy or symlink images:")
    print(f"   ln -s {patches_dir.absolute()} {output_images_dir.absolute()}")
    print(f"2. Or copy images:")
    print(f"   cp {patches_dir}/*.jpg {output_images_dir}/")
    print(f"\nAlternatively, you can point the dataset config directly to: {patches_dir}")


if __name__ == "__main__":
    main()
