#!/usr/bin/env python3
"""
Merge COCO annotation files without modifying original files.

This script combines a base COCO annotation file with additional annotations,
renumbering IDs to avoid conflicts and creating a new merged file.

Usage:
    uv run scripts/merge_annotations.py \
        --base data/rfdetr/herdnet/560_all/annotations/instances_train2017.json \
        --additional data/rfdetr/herdnet/560_all/annotations/hard_negatives_train2017.json \
        --output data/rfdetr/herdnet/560_all/annotations/instances_train_with_hard_negatives2017.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_coco_annotations(path: str) -> Dict:
    """Load COCO format annotations from JSON file."""
    logger.info(f"Loading annotations from {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def merge_coco_annotations(
    base_path: str,
    additional_paths: List[str],
    output_path: str
):
    """
    Merge multiple COCO annotation files.

    Args:
        base_path: Path to base COCO annotations
        additional_paths: List of paths to additional COCO annotations
        output_path: Path to save merged annotations
    """
    # Load base annotations
    merged = load_coco_annotations(base_path)

    logger.info(f"Base annotations: {len(merged['images'])} images, {len(merged['annotations'])} annotations")

    # Track highest IDs from base
    max_image_id = max([img['id'] for img in merged['images']])
    max_annotation_id = max([ann['id'] for ann in merged['annotations']]) if merged['annotations'] else 0

    logger.info(f"Base max image ID: {max_image_id}")
    logger.info(f"Base max annotation ID: {max_annotation_id}")

    # Process each additional annotation file
    for additional_path in additional_paths:
        additional = load_coco_annotations(additional_path)

        logger.info(f"\nMerging {additional_path}")
        logger.info(f"  Additional: {len(additional['images'])} images, {len(additional['annotations'])} annotations")

        # Create mapping for image IDs
        image_id_mapping = {}

        # Add images with renumbered IDs
        for img in additional['images']:
            old_id = img['id']
            new_id = max_image_id + 1

            # Create new image entry
            new_img = img.copy()
            new_img['id'] = new_id

            merged['images'].append(new_img)
            image_id_mapping[old_id] = new_id

            max_image_id = new_id

        # Add annotations with renumbered IDs
        for ann in additional['annotations']:
            old_ann_id = ann['id']
            old_image_id = ann['image_id']

            new_ann = ann.copy()
            new_ann['id'] = max_annotation_id + 1
            new_ann['image_id'] = image_id_mapping[old_image_id]

            merged['annotations'].append(new_ann)
            max_annotation_id = new_ann['id']

        logger.info(f"  After merge: {len(merged['images'])} images, {len(merged['annotations'])} annotations")

    # Verify categories match
    if 'categories' in merged:
        base_categories = {cat['id']: cat['name'] for cat in merged['categories']}
        for additional_path in additional_paths:
            additional = load_coco_annotations(additional_path)
            if 'categories' in additional:
                add_categories = {cat['id']: cat['name'] for cat in additional['categories']}
                if base_categories != add_categories:
                    logger.warning(f"Categories differ between base and {additional_path}")
                    logger.warning(f"  Base: {base_categories}")
                    logger.warning(f"  Additional: {add_categories}")

    # Save merged annotations
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    logger.info(f"\nMerged annotations saved to {output_path}")
    logger.info(f"Final counts: {len(merged['images'])} images, {len(merged['annotations'])} annotations")

    # Print summary statistics
    logger.info("\nSummary:")
    logger.info(f"  Total images: {len(merged['images'])}")
    logger.info(f"  Total annotations: {len(merged['annotations'])}")
    logger.info(f"  Categories: {len(merged.get('categories', []))}")

    if merged['annotations']:
        # Count annotations per category
        category_counts = {}
        for ann in merged['annotations']:
            cat_id = ann['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

        logger.info("\n  Annotations per category:")
        for cat in merged.get('categories', []):
            count = category_counts.get(cat['id'], 0)
            logger.info(f"    {cat['name']}: {count}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge COCO annotation files")
    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Path to base COCO annotations JSON"
    )
    parser.add_argument(
        "--additional",
        type=str,
        nargs='+',
        required=True,
        help="Path(s) to additional COCO annotations JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save merged annotations JSON"
    )

    args = parser.parse_args()

    merge_coco_annotations(
        base_path=args.base,
        additional_paths=args.additional if isinstance(args.additional, list) else [args.additional],
        output_path=args.output
    )


if __name__ == "__main__":
    main()
