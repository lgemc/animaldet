#!/usr/bin/env python3
"""Convert patch-based dataset format to COCO format for RF-DETR.

This script converts datasets (ungulate, HerdNet, etc.) from CSV format with patches
to COCO JSON format that can be used with RF-DETR training.

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
import os


def convert_to_coco(
    patches_dir: Path,
    gt_csv: Path,
    output_json: Path,
    preserve_species: bool = True,
    image_width: int = 560,
    image_height: int = 560
) -> None:
    """Convert patch-based dataset to COCO format.

    Args:
        patches_dir: Directory containing patch images
        gt_csv: Path to ground truth CSV file
        output_json: Output path for COCO JSON file
        preserve_species: If True, preserve individual species as categories.
                         If False, collapse all to single "animal" class.
        image_width: Width of patch images (default: 560)
        image_height: Height of patch images (default: 560)
    """
    # First pass: collect all unique labels from the data
    unique_labels = set()
    with open(gt_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['labels']:  # Skip empty rows
                unique_labels.add(int(float(row['labels'])))

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
                "supercategory": "animal"
            }
            for original_label, category_id in sorted(label_to_category.items(), key=lambda x: x[1])
        ]
    else:
        # Single "animal" category
        categories = [
            {
                "id": 0,
                "name": "animal",
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
            # Skip rows without labels (patches without animals)
            if not row['labels'] or not row['x_min']:
                continue

            image_name = row['images']
            x_min = float(row['x_min'])
            y_min = float(row['y_min'])
            x_max = float(row['x_max'])
            y_max = float(row['y_max'])
            label = int(float(row['labels']))

            # Calculate bbox in COCO format [x, y, width, height]
            width = x_max - x_min
            height = y_max - y_min

            # Map label to category_id
            if preserve_species:
                # Use species-specific category (0-indexed)
                cat_id = label_to_category[label]
            else:
                # Use single "animal" category
                cat_id = 0

            # Create annotation
            annotation = {
                "bbox": [x_min, y_min, width, height],
                "category_id": cat_id
            }

            annotations_by_image[image_name].append(annotation)

    # Get all images (including those without annotations)
    all_images = set()
    for img_file in patches_dir.glob("*.[Jj][Pp][Gg]"):
        all_images.add(img_file.name)
    for img_file in patches_dir.glob("*.[Jj][Pp][Ee][Gg]"):
        all_images.add(img_file.name)
    for img_file in patches_dir.glob("*.[Pp][Nn][Gg]"):
        all_images.add(img_file.name)

    # Create images and annotations
    for image_name in sorted(all_images):
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
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "height": image_height,
            "width": image_width
        })

        # Add annotations for this image (if any)
        if image_name in annotations_by_image:
            for ann in annotations_by_image[image_name]:
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
    if preserve_species and len(coco_data['categories']) > 1:
        from collections import Counter
        cat_counts = Counter(ann['category_id'] for ann in coco_data['annotations'])
        print(f"\nCategory distribution:")
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        for cat_id, count in sorted(cat_counts.items()):
            print(f"  {cat_id_to_name[cat_id]:>15s} (id={cat_id}): {count:>6d} annotations")

    print(f"\nOutput saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert patch-based dataset to COCO format for RF-DETR"
    )

    # Single file mode arguments
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to single CSV file to convert (single-file mode)"
    )
    parser.add_argument(
        "--patches-dir",
        type=str,
        help="Directory containing patch images (required with --csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (required with --csv)"
    )

    # Batch mode arguments
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Root directory of input dataset (e.g., data/herdnet/processed/560_all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output root directory for COCO format (e.g., data/rfdetr/herdnet/560_all)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to convert (default: train val test)"
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
        help="Collapse all species into single 'animal' class"
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=560,
        help="Width of patch images (default: 560)"
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=560,
        help="Height of patch images (default: 560)"
    )
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Create symlinks to original images instead of copying"
    )

    args = parser.parse_args()

    # Determine whether to preserve species
    preserve_species = not args.single_class

    # Single file mode
    if args.csv:
        if not args.patches_dir or not args.output:
            parser.error("--csv requires both --patches-dir and --output")

        print("Single file conversion mode")
        print(f"  CSV: {args.csv}")
        print(f"  Patches dir: {args.patches_dir}")
        print(f"  Output: {args.output}")
        print(f"  Preserve species: {preserve_species}")
        print()

        convert_to_coco(
            patches_dir=Path(args.patches_dir),
            gt_csv=Path(args.csv),
            output_json=Path(args.output),
            preserve_species=preserve_species,
            image_width=args.image_width,
            image_height=args.image_height
        )
        return

    # Batch mode - process multiple splits
    if not args.input_dir or not args.output_dir:
        parser.error("Either provide --csv with --patches-dir and --output, or provide --input-dir and --output-dir")

    # Set up paths
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    print(f"Converting dataset from: {input_root}")
    print(f"Output directory: {output_root}")
    print(f"Preserve species: {preserve_species}")
    print(f"Splits: {', '.join(args.splits)}")
    print()

    # Process each split
    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")

        # Input paths
        patches_dir = input_root / split
        gt_csv = patches_dir / "gt.csv"

        # Output paths following COCO standard naming convention
        # RF-DETR expects: train2017, val2017, test2017
        output_images_dir = output_root / f"{split}2017"
        # RF-DETR expects: instances_train2017.json, instances_val2017.json, image_info_test-dev2017.json
        if split == "test":
            output_json = output_root / f"annotations/image_info_test-dev2017.json"
        else:
            output_json = output_root / f"annotations/instances_{split}2017.json"

        # Verify input exists
        if not patches_dir.exists():
            print(f"Warning: Patches directory not found: {patches_dir} - skipping")
            continue

        if not gt_csv.exists():
            print(f"Warning: Ground truth CSV not found: {gt_csv} - skipping")
            continue

        print(f"  Input patches dir: {patches_dir}")
        print(f"  Input GT CSV: {gt_csv}")
        print(f"  Output JSON: {output_json}")
        print(f"  Output images dir: {output_images_dir}")

        # Convert to COCO format
        convert_to_coco(
            patches_dir=patches_dir,
            gt_csv=gt_csv,
            output_json=output_json,
            preserve_species=preserve_species,
            image_width=args.image_width,
            image_height=args.image_height
        )

        # Create symlink or copy images
        output_images_dir.mkdir(parents=True, exist_ok=True)

        if args.symlink_images:
            # Create symlink to patches directory
            if output_images_dir.exists() and not output_images_dir.is_symlink():
                print(f"\nWarning: {output_images_dir} already exists and is not a symlink")
                print(f"  Skipping symlink creation. Remove directory manually if needed.")
            elif output_images_dir.is_symlink():
                print(f"\nSymlink already exists: {output_images_dir} -> {output_images_dir.resolve()}")
            else:
                output_images_dir.rmdir()  # Remove empty dir
                output_images_dir.symlink_to(patches_dir.absolute())
                print(f"\nCreated symlink: {output_images_dir} -> {patches_dir.absolute()}")
        else:
            print(f"\nNote: Images not copied/symlinked automatically.")
            print(f"  Option 1 - Create symlink:")
            print(f"    ln -s {patches_dir.absolute()} {output_images_dir.absolute()}")
            print(f"  Option 2 - Or update config to point directly to: {patches_dir}")

    print(f"\n{'='*60}")
    print("All splits processed!")
    print(f"{'='*60}")
    print(f"\nOutput structure:")
    print(f"  {output_root}/")
    print(f"    annotations/")
    for split in args.splits:
        if split == "test":
            print(f"      image_info_test-dev2017.json")
        else:
            print(f"      instances_{split}2017.json")
    for split in args.splits:
        print(f"    {split}2017/")


if __name__ == "__main__":
    main()
