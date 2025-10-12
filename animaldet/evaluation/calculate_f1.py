"""
F1 Score Calculation for Animal Detection.

This module calculates F1 scores comparing ground truth annotations with model predictions.
True Positive: Detection center within 20px of ground truth center AND correct class match.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class F1ScoreCalculator:
    """Calculate F1 scores for detection results."""

    def __init__(self, distance_threshold: float = 20.0):
        """
        Initialize F1 score calculator.

        Args:
            distance_threshold: Maximum distance in pixels for a true positive match.
        """
        self.distance_threshold = distance_threshold

    def load_ground_truth(self, csv_path: str) -> pd.DataFrame:
        """
        Load ground truth annotations.

        Expected format: Image, x1, y1, x2, y2, Label

        Args:
            csv_path: Path to ground truth CSV file.

        Returns:
            DataFrame with ground truth annotations.
        """
        df = pd.read_csv(csv_path)

        # Calculate center coordinates
        df['x_center'] = (df['x1'] + df['x2']) / 2
        df['y_center'] = (df['y1'] + df['y2']) / 2

        return df

    def load_predictions(self, csv_path: str, format_type: str = 'auto') -> pd.DataFrame:
        """
        Load prediction results.

        Supports two formats:
        - RF-DETR: images, x, y, x_max, y_max, labels, scores
        - HerdNet: images, labels, scores, dscores, x, y, count_*

        Args:
            csv_path: Path to predictions CSV file.
            format_type: 'rfdetr', 'herdnet', or 'auto' to detect automatically.

        Returns:
            DataFrame with predictions in standardized format.
        """
        df = pd.read_csv(csv_path)

        # Auto-detect format
        if format_type == 'auto':
            if 'x_max' in df.columns and 'y_max' in df.columns:
                format_type = 'rfdetr'
            elif 'dscores' in df.columns:
                format_type = 'herdnet'
            else:
                raise ValueError("Could not auto-detect prediction format")

        # Standardize column names
        if 'images' in df.columns:
            df = df.rename(columns={'images': 'Image'})

        if format_type == 'rfdetr':
            # RF-DETR format: has bounding box coordinates (x, y, x_max, y_max)
            df['x_center'] = (df['x'] + df['x_max']) / 2
            df['y_center'] = (df['y'] + df['y_max']) / 2
        elif format_type == 'herdnet':
            # HerdNet format: has center point (x, y)
            df['x_center'] = df['x']
            df['y_center'] = df['y']

        # Standardize label column name
        if 'labels' in df.columns:
            df = df.rename(columns={'labels': 'Label'})

        return df

    def calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def match_predictions_to_ground_truth(
        self,
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame,
        image_name: str,
        class_agnostic: bool = False
    ) -> Tuple[List[bool], List[bool]]:
        """
        Match predictions to ground truth for a single image.

        Args:
            gt_df: Ground truth dataframe for the image.
            pred_df: Predictions dataframe for the image.
            image_name: Name of the image being processed.
            class_agnostic: If True, ignore class labels (match by distance only).

        Returns:
            Tuple of (matched_predictions, matched_ground_truths) as boolean lists.
        """
        gt_matched = [False] * len(gt_df)
        pred_matched = [False] * len(pred_df)

        # For each prediction, find the closest ground truth
        for pred_idx, pred_row in pred_df.iterrows():
            pred_x = pred_row['x_center']
            pred_y = pred_row['y_center']
            pred_label = pred_row['Label']

            best_match_idx = -1
            best_distance = float('inf')

            for gt_idx, gt_row in gt_df.iterrows():
                # Skip if already matched
                if gt_matched[gt_idx]:
                    continue

                gt_x = gt_row['x_center']
                gt_y = gt_row['y_center']
                gt_label = gt_row['Label']

                # Check class match (unless class_agnostic)
                if not class_agnostic and pred_label != gt_label:
                    continue

                # Calculate distance
                distance = self.calculate_distance(pred_x, pred_y, gt_x, gt_y)

                # Check if within threshold and better than previous matches
                if distance <= self.distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_match_idx = gt_idx

            # Mark as matched if valid match found
            if best_match_idx != -1:
                gt_matched[best_match_idx] = True
                pred_matched[pred_idx] = True

        return pred_matched, gt_matched

    def calculate_metrics(
        self,
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate precision, recall, and F1 score.
        Calculates both strict (class-matched) and binary (class-agnostic) metrics.

        Args:
            gt_df: Ground truth dataframe.
            pred_df: Predictions dataframe.

        Returns:
            Dictionary with global, binary, and per-class metrics.
        """
        # Get unique images
        all_images = set(gt_df['Image'].unique()) | set(pred_df['Image'].unique())

        # Strict (class-matched) counts
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Binary (class-agnostic) counts
        binary_tp = 0
        binary_fp = 0
        binary_fn = 0

        # Per-class counts
        all_classes = set(gt_df['Label'].unique()) | set(pred_df['Label'].unique())
        class_metrics = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in all_classes}

        # Process each image
        for image in all_images:
            gt_image = gt_df[gt_df['Image'] == image].reset_index(drop=True)
            pred_image = pred_df[pred_df['Image'] == image].reset_index(drop=True)

            if len(pred_image) == 0:
                # All ground truths are false negatives
                total_fn += len(gt_image)
                binary_fn += len(gt_image)
                for _, gt_row in gt_image.iterrows():
                    class_metrics[gt_row['Label']]['fn'] += 1
                continue

            if len(gt_image) == 0:
                # All predictions are false positives
                total_fp += len(pred_image)
                binary_fp += len(pred_image)
                for _, pred_row in pred_image.iterrows():
                    class_metrics[pred_row['Label']]['fp'] += 1
                continue

            # Strict matching (class-matched)
            pred_matched, gt_matched = self.match_predictions_to_ground_truth(
                gt_image, pred_image, image, class_agnostic=False
            )

            # Count true positives (strict)
            tp_count = sum(pred_matched)
            total_tp += tp_count

            # Count false positives (strict)
            fp_count = sum(1 for m in pred_matched if not m)
            total_fp += fp_count

            # Count false negatives (strict)
            fn_count = sum(1 for m in gt_matched if not m)
            total_fn += fn_count

            # Update per-class counts
            for idx, matched in enumerate(pred_matched):
                label = pred_image.iloc[idx]['Label']
                if matched:
                    class_metrics[label]['tp'] += 1
                else:
                    class_metrics[label]['fp'] += 1

            for idx, matched in enumerate(gt_matched):
                if not matched:
                    label = gt_image.iloc[idx]['Label']
                    class_metrics[label]['fn'] += 1

            # Binary matching (class-agnostic)
            pred_matched_binary, gt_matched_binary = self.match_predictions_to_ground_truth(
                gt_image, pred_image, image, class_agnostic=True
            )

            binary_tp += sum(pred_matched_binary)
            binary_fp += sum(1 for m in pred_matched_binary if not m)
            binary_fn += sum(1 for m in gt_matched_binary if not m)

        # Calculate global metrics (strict)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate binary metrics (class-agnostic)
        binary_precision = binary_tp / (binary_tp + binary_fp) if (binary_tp + binary_fp) > 0 else 0.0
        binary_recall = binary_tp / (binary_tp + binary_fn) if (binary_tp + binary_fn) > 0 else 0.0
        binary_f1 = 2 * binary_precision * binary_recall / (binary_precision + binary_recall) if (binary_precision + binary_recall) > 0 else 0.0

        results = {
            'global': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn
            },
            'binary': {
                'precision': binary_precision,
                'recall': binary_recall,
                'f1_score': binary_f1,
                'tp': binary_tp,
                'fp': binary_fp,
                'fn': binary_fn
            }
        }

        # Calculate per-class metrics
        for cls, counts in class_metrics.items():
            tp = counts['tp']
            fp = counts['fp']
            fn = counts['fn']

            cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0.0

            results[f'class_{cls}'] = {
                'precision': cls_precision,
                'recall': cls_recall,
                'f1_score': cls_f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }

        return results

    def save_results(self, results: Dict[str, Dict[str, float]], output_path: str):
        """
        Save F1 score results to CSV.

        Args:
            results: Dictionary with metrics results.
            output_path: Path to save CSV file.
        """
        rows = []
        for category, metrics in results.items():
            row = {'category': category}
            row.update(metrics)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

        return df


def calculate_f1_scores(
    gt_csv_path: str,
    pred_csv_path: str,
    output_csv_path: str,
    prediction_format: str = 'auto',
    distance_threshold: float = 20.0
) -> pd.DataFrame:
    """
    Calculate F1 scores and save results.

    Args:
        gt_csv_path: Path to ground truth CSV.
        pred_csv_path: Path to predictions CSV.
        output_csv_path: Path to save results CSV.
        prediction_format: 'rfdetr', 'herdnet', or 'auto'.
        distance_threshold: Distance threshold for true positive match (default: 20px).

    Returns:
        DataFrame with calculated metrics.
    """
    calculator = F1ScoreCalculator(distance_threshold=distance_threshold)

    # Load data
    logger.info(f"Loading ground truth from {gt_csv_path}")
    gt_df = calculator.load_ground_truth(gt_csv_path)

    logger.info(f"Loading predictions from {pred_csv_path}")
    pred_df = calculator.load_predictions(pred_csv_path, format_type=prediction_format)

    # Calculate metrics
    logger.info("Calculating F1 scores...")
    results = calculator.calculate_metrics(gt_df, pred_df)

    # Save results
    results_df = calculator.save_results(results, output_csv_path)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("F1 SCORE RESULTS")
    logger.info("="*60)
    logger.info(f"\nGlobal Metrics (strict, class-matched):")
    logger.info(f"  Precision: {results['global']['precision']:.4f}")
    logger.info(f"  Recall:    {results['global']['recall']:.4f}")
    logger.info(f"  F1 Score:  {results['global']['f1_score']:.4f}")
    logger.info(f"  TP: {results['global']['tp']}, FP: {results['global']['fp']}, FN: {results['global']['fn']}")

    logger.info(f"\nBinary Metrics (class-agnostic, location-only):")
    logger.info(f"  Precision: {results['binary']['precision']:.4f}")
    logger.info(f"  Recall:    {results['binary']['recall']:.4f}")
    logger.info(f"  F1 Score:  {results['binary']['f1_score']:.4f}")
    logger.info(f"  TP: {results['binary']['tp']}, FP: {results['binary']['fp']}, FN: {results['binary']['fn']}")

    logger.info(f"\nPer-Class Metrics:")
    for key in sorted([k for k in results.keys() if k.startswith('class_')]):
        cls = key.replace('class_', '')
        logger.info(f"  Class {cls}:")
        logger.info(f"    Precision: {results[key]['precision']:.4f}")
        logger.info(f"    Recall:    {results[key]['recall']:.4f}")
        logger.info(f"    F1 Score:  {results[key]['f1_score']:.4f}")
        logger.info(f"    TP: {results[key]['tp']}, FP: {results[key]['fp']}, FN: {results[key]['fn']}")

    logger.info("="*60 + "\n")

    return results_df