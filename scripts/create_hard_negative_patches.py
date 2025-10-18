#!/usr/bin/env python3
"""
Create hard negative patches from detected false positives.

Extracts image patches around hard negative detections and creates
COCO-format annotations (empty annotations for background patches).

Usage:
    uv run scripts/create_hard_negative_patches.py \
        --hard_negatives outputs/hard_negatives.csv \
        --source_images data/herdnet/raw/ \
        --output_patches data/rfdetr/herdnet/560_all/hard_negatives/ \
        --output_annotations data/rfdetr/herdnet/560_all/annotations/hard_negatives_train2017.json \
        --patch_size 560
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_image_path(image_name: str, source_dir: Path) -> Path:
    """
    Find the full path to an image by searching in source directory.

    Args:
        image_name: Name of the image file
        source_dir: Root directory to search

    Returns:
        Full path to the image file
    """
    # Search for image in all subdirectories
    for img_path in source_dir.rglob(image_name):
        return img_path

    raise FileNotFoundError(f"Image {image_name} not found in {source_dir}")


def extract_patch(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
    patch_size: int
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Extract a patch centered on the bounding box.

    Args:
        image: Source PIL Image
        bbox: Bounding box (x_min, y_min, x_max, y_max)
        patch_size: Size of the patch (width and height)

    Returns:
        Tuple of (patch image, patch coordinates (x, y, width, height))
    """
    img_width, img_height = image.size
    x_min, y_min, x_max, y_max = bbox

    # Calculate center of bounding box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Calculate patch boundaries
    patch_x_min = int(center_x - patch_size / 2)
    patch_y_min = int(center_y - patch_size / 2)
    patch_x_max = patch_x_min + patch_size
    patch_y_max = patch_y_min + patch_size

    # Ensure patch stays within image bounds
    if patch_x_min < 0:
        patch_x_min = 0
        patch_x_max = patch_size
    elif patch_x_max > img_width:
        patch_x_max = img_width
        patch_x_min = img_width - patch_size

    if patch_y_min < 0:
        patch_y_min = 0
        patch_y_max = patch_size
    elif patch_y_max > img_height:
        patch_y_max = img_height
        patch_y_min = img_height - patch_size

    # Clamp to image bounds (for images smaller than patch_size)
    patch_x_min = max(0, patch_x_min)
    patch_y_min = max(0, patch_y_min)
    patch_x_max = min(img_width, patch_x_max)
    patch_y_max = min(img_height, patch_y_max)

    # Extract patch
    patch = image.crop((patch_x_min, patch_y_min, patch_x_max, patch_y_max))

    # If patch is smaller than desired size (edge cases), pad it
    if patch.size[0] < patch_size or patch.size[1] < patch_size:
        padded = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
        padded.paste(patch, (0, 0))
        patch = padded

    return patch, (patch_x_min, patch_y_min, patch_x_max - patch_x_min, patch_y_max - patch_y_min)


def create_coco_annotations(
    images_info: List[Dict],
    categories: List[Dict]
) -> Dict:
    """
    Create COCO format annotation structure.

    Args:
        images_info: List of image info dicts
        categories: List of category dicts

    Returns:
        COCO format dictionary
    """
    return {
        "images": images_info,
        "annotations": [],  # Empty - these are background patches
        "categories": categories
    }


def create_hard_negative_patches(
    hard_negatives_path: str,
    source_images_dir: str,
    output_patches_dir: str,
    output_annotations_path: str,
    patch_size: int = 560,
    max_patches: int = None
):
    """
    Create hard negative patches and COCO annotations.

    Args:
        hard_negatives_path: Path to hard negatives CSV
        source_images_dir: Root directory containing source images
        output_patches_dir: Directory to save extracted patches
        output_annotations_path: Path to save COCO annotations JSON
        patch_size: Size of extracted patches
        max_patches: Maximum number of patches to create (None for all)
    """
    logger.info(f"Loading hard negatives from {hard_negatives_path}")
    hard_negatives = pd.read_csv(hard_negatives_path)

    if max_patches:
        hard_negatives = hard_negatives.head(max_patches)
        logger.info(f"Limited to {max_patches} patches")

    logger.info(f"Processing {len(hard_negatives)} hard negatives")

    # Create output directories
    output_patches_path = Path(output_patches_dir)
    output_patches_path.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_images_dir)

    # Define categories (matching RF-DETR/HerdNet categories)
    categories = [
        {"id": 1, "name": "animal"},
        {"id": 2, "name": "cattle"},
        {"id": 3, "name": "sheep"},
        {"id": 4, "name": "horse"}
    ]

    images_info = []
    skipped = 0

    logger.info("Extracting patches...")
    for idx, row in tqdm(hard_negatives.iterrows(), total=len(hard_negatives)):
        image_name = row['images']
        bbox = (row['x'], row['y'], row['x_max'], row['y_max'])

        try:
            # Find and load source image
            image_path = find_image_path(image_name, source_path)
            image = Image.open(image_path).convert('RGB')

            # Extract patch
            patch, patch_coords = extract_patch(image, bbox, patch_size)

            # Generate unique patch filename
            patch_filename = f"hard_negative_{idx:06d}.jpg"
            patch_path = output_patches_path / patch_filename

            # Save patch
            patch.save(patch_path, quality=95)

            # Add to COCO images list
            images_info.append({
                "id": idx + 1,  # 1-indexed
                "file_name": patch_filename,
                "height": patch.size[1],
                "width": patch.size[0],
                "source_image": image_name,
                "source_bbox": list(bbox),
                "confidence": float(row['scores']),
                "max_iou": float(row['max_iou'])
            })

        except FileNotFoundError as e:
            logger.warning(f"Skipping: {e}")
            skipped += 1
            continue
        except Exception as e:
            logger.error(f"Error processing {image_name}: {e}")
            skipped += 1
            continue

    logger.info(f"Created {len(images_info)} patches")
    if skipped > 0:
        logger.warning(f"Skipped {skipped} patches due to errors")

    # Create COCO annotations
    coco_data = create_coco_annotations(images_info, categories)

    # Save annotations
    annotations_path = Path(output_annotations_path)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)

    with open(annotations_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    logger.info(f"Annotations saved to {annotations_path}")
    logger.info(f"Patches saved to {output_patches_path}")

    return coco_data


def main():
    parser = argparse.ArgumentParser(description="Create hard negative patches")
    parser.add_argument(
        "--hard_negatives",
        type=str,
        required=True,
        help="Path to hard negatives CSV"
    )
    parser.add_argument(
        "--source_images",
        type=str,
        required=True,
        help="Root directory containing source images"
    )
    parser.add_argument(
        "--output_patches",
        type=str,
        required=True,
        help="Directory to save extracted patches"
    )
    parser.add_argument(
        "--output_annotations",
        type=str,
        required=True,
        help="Path to save COCO annotations JSON"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=560,
        help="Size of extracted patches (default: 560)"
    )
    parser.add_argument(
        "--max_patches",
        type=int,
        default=None,
        help="Maximum number of patches to create (default: all)"
    )

    args = parser.parse_args()

    create_hard_negative_patches(
        hard_negatives_path=args.hard_negatives,
        source_images_dir=args.source_images,
        output_patches_dir=args.output_patches,
        output_annotations_path=args.output_annotations,
        patch_size=args.patch_size,
        max_patches=args.max_patches
    )


if __name__ == "__main__":
    main()
