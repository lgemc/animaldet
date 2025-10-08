"""Detection metrics including F1 score with center-based matching.

This module provides evaluation metrics for object detection tasks,
adapted from RF-DETR-LGEMC with improvements for modularity.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np


def calculate_box_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Calculate center of a bounding box in COCO format [x, y, width, height].

    Args:
        bbox: Bounding box as [x, y, width, height]

    Returns:
        Tuple of (center_x, center_y)
    """
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def center_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Euclidean distance between centers of two boxes.

    Args:
        bbox1: First bounding box [x, y, width, height]
        bbox2: Second bounding box [x, y, width, height]

    Returns:
        Euclidean distance between centers
    """
    cx1, cy1 = calculate_box_center(bbox1)
    cx2, cy2 = calculate_box_center(bbox2)
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def match_detections(
    predictions: List[Dict],
    ground_truths: List[Dict],
    center_threshold: float = 50.0,
    score_threshold: float = 0.5,
    verbose: bool = False
) -> Tuple[int, int, int, Dict]:
    """
    Match predictions to ground truths based on class and center proximity.

    Args:
        predictions: List of predictions with keys: image_id, category_id, bbox, score
        ground_truths: List of ground truths with keys: image_id, category_id, bbox
        center_threshold: Maximum center distance (in pixels) to consider a match
        score_threshold: Minimum confidence score for predictions
        verbose: If True, print detailed debug information

    Returns:
        Tuple of (true_positives, false_positives, false_negatives, per_class_stats)
    """
    if verbose:
        print(f"\n[F1 Metric] Matching detections:")
        print(f"  Total predictions before filtering: {len(predictions)}")
        print(f"  Total ground truths: {len(ground_truths)}")
        print(f"  Score threshold: {score_threshold}")
        print(f"  Center threshold: {center_threshold}px")

    # Filter predictions by score threshold
    predictions = [p for p in predictions if p.get('score', 1.0) >= score_threshold]

    if verbose:
        print(f"  Predictions after score filtering: {len(predictions)}")

    # Group by image_id for efficient matching
    preds_by_image = defaultdict(list)
    gts_by_image = defaultdict(list)

    for pred in predictions:
        preds_by_image[pred['image_id']].append(pred)

    for gt in ground_truths:
        gts_by_image[gt['image_id']].append(gt)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    per_class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    # Get all unique image_ids
    all_image_ids = set(list(preds_by_image.keys()) + list(gts_by_image.keys()))

    for image_id in all_image_ids:
        img_preds = preds_by_image.get(image_id, [])
        img_gts = gts_by_image.get(image_id, [])

        matched_gts = set()

        # Sort predictions by score (highest first) for greedy matching
        img_preds = sorted(img_preds, key=lambda x: x.get('score', 1.0), reverse=True)

        for pred in img_preds:
            pred_class = pred['category_id']
            pred_bbox = pred['bbox']

            best_match_idx = None
            best_match_dist = center_threshold

            # Find best matching ground truth
            for idx, gt in enumerate(img_gts):
                if idx in matched_gts:
                    continue

                if gt['category_id'] != pred_class:
                    continue

                dist = center_distance(pred_bbox, gt['bbox'])

                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match_idx = idx

            if best_match_idx is not None:
                # True positive
                true_positives += 1
                per_class_stats[pred_class]['tp'] += 1
                matched_gts.add(best_match_idx)
            else:
                # False positive
                false_positives += 1
                per_class_stats[pred_class]['fp'] += 1

        # Count unmatched ground truths as false negatives
        for idx, gt in enumerate(img_gts):
            if idx not in matched_gts:
                false_negatives += 1
                per_class_stats[gt['category_id']]['fn'] += 1

    if verbose:
        print(f"\n[F1 Metric] Matching results:")
        print(f"  True Positives (TP): {true_positives}")
        print(f"  False Positives (FP): {false_positives}")
        print(f"  False Negatives (FN): {false_negatives}")

    return true_positives, false_positives, false_negatives, dict(per_class_stats)


def calculate_f1_score(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def evaluate_detection_f1(
    predictions: List[Dict],
    ground_truths: List[Dict],
    center_threshold: float = 50.0,
    score_threshold: float = 0.5,
    class_names: Dict[int, str] = None,
    verbose: bool = False
) -> Dict:
    """
    Evaluate object detection predictions using F1 score with center-based matching.

    Args:
        predictions: List of predictions in COCO format
                    Each dict should have: image_id, category_id, bbox [x,y,w,h], score
        ground_truths: List of ground truths in COCO format
                      Each dict should have: image_id, category_id, bbox [x,y,w,h]
        center_threshold: Maximum center distance in pixels to consider a match
        score_threshold: Minimum confidence score for predictions
        class_names: Optional mapping of category_id to class names
        verbose: If True, print detailed debug information

    Returns:
        Dictionary with overall and per-class metrics
    """
    tp, fp, fn, per_class_stats = match_detections(
        predictions, ground_truths, center_threshold, score_threshold, verbose
    )

    overall_precision, overall_recall, overall_f1 = calculate_f1_score(tp, fp, fn)

    if verbose:
        print(f"\n[F1 Metric] Final F1 Score:")
        print(f"  Precision: {overall_precision:.4f}")
        print(f"  Recall: {overall_recall:.4f}")
        print(f"  F1: {overall_f1:.4f}")

    per_class_f1 = {}
    for class_id, stats in per_class_stats.items():
        precision, recall, f1 = calculate_f1_score(
            stats['tp'], stats['fp'], stats['fn']
        )
        class_label = class_names.get(class_id, f"class_{class_id}") if class_names else f"class_{class_id}"
        per_class_f1[class_label] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': stats['tp'],
            'fp': stats['fp'],
            'fn': stats['fn']
        }

    return {
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        },
        'per_class': per_class_f1,
        'config': {
            'center_threshold': center_threshold,
            'score_threshold': score_threshold
        }
    }
