#!/usr/bin/env python3
"""
Extract hard negative detections by comparing predictions with ground truth.

A hard negative is a false positive detection with:
- Low IoU with all ground truth boxes (< iou_threshold)
- High confidence score (> confidence_threshold)

Usage:
    uv run scripts/extract_hard_negatives.py \
        --predictions outputs/inference/rfdetr_detections_on_background_train.csv \
        --ground_truth data/herdnet/raw/groundtruth/csv/test_big_size_A_B_E_K_WH_WB.csv \
        --output outputs/hard_negatives.csv \
        --iou_threshold 0.3 \
        --confidence_threshold 0.5
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: [x_min, y_min, x_max, y_max]
        box2: [x_min, y_min, x_max, y_max]

    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def extract_hard_negatives(
    predictions_path: str,
    ground_truth_path: str,
    output_path: str,
    iou_threshold: float = 0.3,
    confidence_threshold: float = 0.5
):
    """
    Extract hard negative detections from predictions.

    Args:
        predictions_path: Path to predictions CSV
        ground_truth_path: Path to ground truth CSV
        output_path: Path to save hard negatives CSV
        iou_threshold: Maximum IoU to consider as hard negative
        confidence_threshold: Minimum confidence score for hard negatives
    """
    logger.info(f"Loading predictions from {predictions_path}")
    predictions = pd.read_csv(predictions_path)

    logger.info(f"Loading ground truth from {ground_truth_path}")
    ground_truth = pd.read_csv(ground_truth_path)

    # Normalize column names
    if 'Image' in ground_truth.columns:
        ground_truth.rename(columns={'Image': 'images'}, inplace=True)
    if 'Label' in ground_truth.columns:
        ground_truth.rename(columns={'Label': 'labels'}, inplace=True)

    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Ground truth shape: {ground_truth.shape}")
    logger.info(f"Parameters: IoU threshold={iou_threshold}, confidence threshold={confidence_threshold}")

    # Group ground truth by image for efficient lookup
    gt_by_image = ground_truth.groupby('images')

    hard_negatives = []
    total_predictions = len(predictions)

    logger.info("Processing predictions...")
    for idx, pred_row in tqdm(predictions.iterrows(), total=total_predictions):
        # Skip low confidence predictions
        if pred_row['scores'] < confidence_threshold:
            continue

        image_name = pred_row['images']
        pred_box = [pred_row['x'], pred_row['y'], pred_row['x_max'], pred_row['y_max']]

        # Get all ground truth boxes for this image
        if image_name not in gt_by_image.groups:
            # No ground truth for this image - it's a hard negative
            max_iou = 0.0
        else:
            gt_boxes = gt_by_image.get_group(image_name)

            # Calculate IoU with all ground truth boxes
            ious = []
            for _, gt_row in gt_boxes.iterrows():
                gt_box = [gt_row['x1'], gt_row['y1'], gt_row['x2'], gt_row['y2']]
                iou = calculate_iou(pred_box, gt_box)
                ious.append(iou)

            max_iou = max(ious) if ious else 0.0

        # Mark as hard negative if IoU is below threshold
        if max_iou < iou_threshold:
            hard_negatives.append({
                'images': image_name,
                'x': pred_row['x'],
                'y': pred_row['y'],
                'x_max': pred_row['x_max'],
                'y_max': pred_row['y_max'],
                'labels': pred_row['labels'],
                'scores': pred_row['scores'],
                'max_iou': max_iou
            })

    # Create DataFrame and save
    hard_negatives_df = pd.DataFrame(hard_negatives)

    logger.info(f"Found {len(hard_negatives_df)} hard negatives out of {total_predictions} predictions")
    logger.info(f"Hard negative rate: {len(hard_negatives_df) / total_predictions * 100:.2f}%")

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hard_negatives_df.to_csv(output_path, index=False)
    logger.info(f"Hard negatives saved to {output_path}")

    # Print statistics
    logger.info("\nStatistics:")
    logger.info(f"  Mean confidence: {hard_negatives_df['scores'].mean():.3f}")
    logger.info(f"  Mean max IoU: {hard_negatives_df['max_iou'].mean():.3f}")
    logger.info(f"  Unique images: {hard_negatives_df['images'].nunique()}")

    return hard_negatives_df


def main():
    parser = argparse.ArgumentParser(description="Extract hard negative detections")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions CSV"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save hard negatives CSV"
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.3,
        help="Maximum IoU to consider as hard negative (default: 0.3)"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score for hard negatives (default: 0.5)"
    )

    args = parser.parse_args()

    extract_hard_negatives(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        output_path=args.output,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold
    )


if __name__ == "__main__":
    main()
