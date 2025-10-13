#!/usr/bin/env python3
"""
Calculate Confidence vs F1 Score from saved predictions CSV.

This tool analyzes the relationship between confidence thresholds and F1 scores
using predictions and ground truths from a saved CSV file. Based on animaldet's
F1 evaluation metrics and inspired by RF-DETR confidence analysis.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from animaldet.evaluation.metrics.detection import (
    calculate_f1_score,
    match_detections,
)

logger = logging.getLogger(__name__)


def load_ground_truths_from_csv(csv_path: str) -> List[Dict]:
    """
    Load ground truths from HerdNet CSV format.

    Supports two CSV formats:
    1. Format 1: columns are images, x, y, x_max, y_max, labels
    2. Format 2: columns are Image, x1, y1, x2, y2, Label

    Args:
        csv_path: Path to ground truth CSV file

    Returns:
        List of ground truth dictionaries
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded ground truth CSV with {len(df)} rows")

    all_ground_truths = []

    # Detect format
    if "images" in df.columns:
        # Format 1
        for _, row in df.iterrows():
            image_id = row["images"]
            x1, y1 = row["x"], row["y"]
            x2, y2 = row["x_max"], row["y_max"]
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

            all_ground_truths.append({
                "image_id": image_id,
                "category_id": int(row["labels"]),
                "bbox": bbox_xywh,
            })
    elif "Image" in df.columns:
        # Format 2
        for _, row in df.iterrows():
            image_id = row["Image"]
            x1, y1 = row["x1"], row["y1"]
            x2, y2 = row["x2"], row["y2"]
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

            all_ground_truths.append({
                "image_id": image_id,
                "category_id": int(row["Label"]),
                "bbox": bbox_xywh,
            })
    else:
        raise ValueError(f"Unknown ground truth CSV format. Expected either 'images' or 'Image' column")

    return all_ground_truths


def load_predictions_from_csv(csv_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load predictions and ground truths from CSV file.

    Supports two CSV formats:
    1. Grouped format: columns are image_id, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels
       - Boxes are stored as string representations of lists
    2. Flat format: columns are images, x, y, x_max, y_max, labels, scores
       - One detection per row

    Args:
        csv_path: Path to CSV file with predictions

    Returns:
        Tuple of (predictions, ground_truths)
    """
    import ast

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} rows")

    all_predictions = []
    all_ground_truths = []

    # Detect CSV format
    if "image_id" in df.columns:
        # Grouped format
        for _, row in df.iterrows():
            image_id = row["image_id"]

            # Parse predictions
            if pd.notna(row["pred_boxes"]) and row["pred_boxes"] != "[]":
                pred_boxes = ast.literal_eval(row["pred_boxes"])
                pred_labels = ast.literal_eval(row["pred_labels"])
                pred_scores = ast.literal_eval(row["pred_scores"])

                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                    x1, y1, x2, y2 = box
                    bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

                    all_predictions.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": bbox_xywh,
                        "score": float(score),
                    })

            # Parse ground truths
            if pd.notna(row["gt_boxes"]) and row["gt_boxes"] != "[]":
                gt_boxes = ast.literal_eval(row["gt_boxes"])
                gt_labels = ast.literal_eval(row["gt_labels"])

                for box, label in zip(gt_boxes, gt_labels):
                    # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                    x1, y1, x2, y2 = box
                    bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

                    all_ground_truths.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": bbox_xywh,
                    })

    elif "images" in df.columns:
        # Flat format: one detection per row
        for _, row in df.iterrows():
            image_id = row["images"]
            x1, y1 = row["x"], row["y"]
            x2, y2 = row["x_max"], row["y_max"]
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

            all_predictions.append({
                "image_id": image_id,
                "category_id": int(row["labels"]),
                "bbox": bbox_xywh,
                "score": float(row["scores"]),
            })
    else:
        raise ValueError(f"Unknown CSV format. Expected either 'image_id' or 'images' column")

    return all_predictions, all_ground_truths


def calculate_f1_at_thresholds(
    predictions: List[Dict],
    ground_truths: List[Dict],
    confidence_thresholds: np.ndarray,
    center_threshold: float = 50.0,
) -> Dict:
    """
    Calculate F1 score at different confidence thresholds.

    Args:
        predictions: List of all predictions with scores
        ground_truths: List of all ground truths
        confidence_thresholds: Array of confidence thresholds to evaluate
        center_threshold: Center distance threshold in pixels

    Returns:
        Dictionary with threshold results
    """
    results = {
        "thresholds": [],
        "f1_scores": [],
        "precisions": [],
        "recalls": [],
        "num_predictions": [],
        "true_positives": [],
        "false_positives": [],
        "false_negatives": [],
    }

    logger.info(f"\nEvaluating F1 at {len(confidence_thresholds)} confidence thresholds...")
    logger.info(f"Total predictions: {len(predictions)}")
    logger.info(f"Total ground truths: {len(ground_truths)}")

    for threshold in tqdm(confidence_thresholds, desc="Calculating F1"):
        # Filter predictions by confidence threshold
        filtered_preds = [p for p in predictions if p["score"] >= threshold]

        # Match detections using animaldet's F1 evaluation
        tp, fp, fn, _ = match_detections(
            filtered_preds,
            ground_truths,
            center_threshold=center_threshold,
            score_threshold=threshold,
        )

        # Calculate F1
        precision, recall, f1 = calculate_f1_score(tp, fp, fn)

        results["thresholds"].append(float(threshold))
        results["f1_scores"].append(f1)
        results["precisions"].append(precision)
        results["recalls"].append(recall)
        results["num_predictions"].append(len(filtered_preds))
        results["true_positives"].append(tp)
        results["false_positives"].append(fp)
        results["false_negatives"].append(fn)

    return results


def plot_confidence_vs_f1(
    results: Dict, output_path: str = "confidence_vs_f1.png"
) -> Tuple[float, float]:
    """
    Plot confidence threshold vs F1 score and related metrics.

    Args:
        results: Dictionary with threshold results
        output_path: Path to save the plot

    Returns:
        Tuple of (best_threshold, best_f1)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: F1 vs Confidence
    axes[0, 0].plot(
        results["thresholds"],
        results["f1_scores"],
        marker="o",
        linewidth=2,
        markersize=4,
    )
    axes[0, 0].set_xlabel("Confidence Threshold")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].set_title("F1 Score vs Confidence Threshold")
    axes[0, 0].grid(True, alpha=0.3)

    # Find and mark best F1
    best_idx = np.argmax(results["f1_scores"])
    best_threshold = results["thresholds"][best_idx]
    best_f1 = results["f1_scores"][best_idx]
    axes[0, 0].axvline(
        best_threshold,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Best: {best_threshold:.2f} (F1={best_f1:.4f})",
    )
    axes[0, 0].legend()

    # Plot 2: Precision and Recall vs Confidence
    axes[0, 1].plot(
        results["thresholds"],
        results["precisions"],
        marker="o",
        label="Precision",
        linewidth=2,
        markersize=4,
    )
    axes[0, 1].plot(
        results["thresholds"],
        results["recalls"],
        marker="s",
        label="Recall",
        linewidth=2,
        markersize=4,
    )
    axes[0, 1].set_xlabel("Confidence Threshold")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Precision & Recall vs Confidence Threshold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(best_threshold, color="r", linestyle="--", alpha=0.5)

    # Plot 3: Number of predictions vs Confidence
    axes[1, 0].plot(
        results["thresholds"],
        results["num_predictions"],
        marker="o",
        linewidth=2,
        markersize=4,
        color="green",
    )
    axes[1, 0].set_xlabel("Confidence Threshold")
    axes[1, 0].set_ylabel("Number of Predictions")
    axes[1, 0].set_title("Number of Predictions vs Confidence Threshold")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(best_threshold, color="r", linestyle="--", alpha=0.5)

    # Plot 4: TP, FP, FN vs Confidence
    axes[1, 1].plot(
        results["thresholds"],
        results["true_positives"],
        marker="o",
        label="True Positives",
        linewidth=2,
        markersize=4,
    )
    axes[1, 1].plot(
        results["thresholds"],
        results["false_positives"],
        marker="s",
        label="False Positives",
        linewidth=2,
        markersize=4,
    )
    axes[1, 1].plot(
        results["thresholds"],
        results["false_negatives"],
        marker="^",
        label="False Negatives",
        linewidth=2,
        markersize=4,
    )
    axes[1, 1].set_xlabel("Confidence Threshold")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("TP/FP/FN vs Confidence Threshold")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(best_threshold, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {output_path}")

    return best_threshold, best_f1


def print_summary(results: Dict):
    """Print summary of results."""
    best_idx = np.argmax(results["f1_scores"])
    best_threshold = results["thresholds"][best_idx]
    best_f1 = results["f1_scores"][best_idx]

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best confidence threshold: {best_threshold:.2f}")
    logger.info(f"Best F1 score: {best_f1:.4f}")
    logger.info(f"Precision at best threshold: {results['precisions'][best_idx]:.4f}")
    logger.info(f"Recall at best threshold: {results['recalls'][best_idx]:.4f}")
    logger.info(f"Predictions at best threshold: {results['num_predictions'][best_idx]}")
    logger.info(f"True Positives: {results['true_positives'][best_idx]}")
    logger.info(f"False Positives: {results['false_positives'][best_idx]}")
    logger.info(f"False Negatives: {results['false_negatives'][best_idx]}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Confidence vs F1 Score from predictions CSV"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file with predictions",
    )
    parser.add_argument(
        "--gt-csv",
        type=str,
        default=None,
        help="Path to ground truth CSV file (optional, only needed if predictions CSV doesn't contain GT)",
    )
    parser.add_argument(
        "--center-threshold",
        type=float,
        default=50.0,
        help="Center distance threshold in pixels (default: 50.0)",
    )
    parser.add_argument(
        "--conf-min",
        type=float,
        default=0.05,
        help="Minimum confidence threshold to evaluate (default: 0.05)",
    )
    parser.add_argument(
        "--conf-max",
        type=float,
        default=0.95,
        help="Maximum confidence threshold to evaluate (default: 0.95)",
    )
    parser.add_argument(
        "--conf-step",
        type=float,
        default=0.05,
        help="Step size for confidence thresholds (default: 0.05)",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="confidence_vs_f1.png",
        help="Output path for plot (default: confidence_vs_f1.png)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="confidence_vs_f1_results.json",
        help="Output path for JSON results (default: confidence_vs_f1_results.json)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load predictions from CSV
    logger.info(f"Loading predictions from {args.csv}")
    predictions, ground_truths = load_predictions_from_csv(args.csv)

    # Load ground truths from separate CSV if provided
    if args.gt_csv:
        logger.info(f"Loading ground truths from {args.gt_csv}")
        ground_truths = load_ground_truths_from_csv(args.gt_csv)

    logger.info(f"\nLoaded:")
    logger.info(f"  {len(predictions)} predictions")
    logger.info(f"  {len(ground_truths)} ground truths")

    if len(ground_truths) == 0:
        logger.error("No ground truths found! Please provide a ground truth CSV with --gt-csv")
        return

    # Generate confidence thresholds
    confidence_thresholds = np.arange(args.conf_min, args.conf_max + args.conf_step, args.conf_step)

    # Calculate F1 at different thresholds
    results = calculate_f1_at_thresholds(
        predictions,
        ground_truths,
        confidence_thresholds,
        center_threshold=args.center_threshold,
    )

    # Plot results
    best_threshold, best_f1 = plot_confidence_vs_f1(results, args.output_plot)

    # Print summary
    print_summary(results)

    # Save results to JSON
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
