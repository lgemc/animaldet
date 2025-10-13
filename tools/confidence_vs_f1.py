#!/usr/bin/env python3
"""
Calculate Confidence vs F1 Score for RF-DETR model using unified inference pipeline.

This script runs inference at multiple confidence thresholds and plots F1 scores,
precision, recall, and detection counts.
"""
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

logger = logging.getLogger(__name__)


def run_inference_at_threshold(
    yaml_cfg: OmegaConf,
    checkpoint_path: Path,
    threshold: float,
    device: str = "cuda",
) -> pd.DataFrame:
    """Run RF-DETR inference at a specific confidence threshold.

    Args:
        yaml_cfg: Configuration object
        checkpoint_path: Path to model checkpoint
        threshold: Confidence threshold for detections
        device: Device to run inference on

    Returns:
        DataFrame with detections
    """
    from animaldet.experiments.rfdetr.adapters.model import build_model
    from animaldet.experiments.rfdetr.stitcher import RFDETRStitcher

    # Build and load model
    model = build_model(yaml_cfg.model, device=device)

    # Load checkpoint
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'model' in checkpoint_data:
        state_dict = checkpoint_data['model']
    elif 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
    else:
        state_dict = checkpoint_data
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Load test CSV
    df = pd.read_csv(yaml_cfg.data.test_csv)

    # Normalize column names
    column_mapping = {}
    if 'Image' in df.columns:
        column_mapping['Image'] = 'images'
    if 'Label' in df.columns:
        column_mapping['Label'] = 'labels'
    if 'x1' in df.columns:
        column_mapping['x1'] = 'x_min'
    if 'y1' in df.columns:
        column_mapping['y1'] = 'y_min'
    if 'x2' in df.columns:
        column_mapping['x2'] = 'x_max'
    if 'y2' in df.columns:
        column_mapping['y2'] = 'y_max'

    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Get unique images
    image_files = df['images'].unique()

    # Create stitcher with current threshold
    stitcher = RFDETRStitcher(
        model=model,
        size=(yaml_cfg.model.resolution, yaml_cfg.model.resolution),
        overlap=0,
        batch_size=yaml_cfg.inference.batch_size,
        confidence_threshold=threshold,
        nms_threshold=yaml_cfg.evaluator.nms_threshold,
        device_name=device,
    )

    # Create transforms
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Run inference
    all_detections = []
    for img_file in image_files:
        img_path = Path(yaml_cfg.data.test_root) / img_file

        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue

        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        transformed = transform(image=image_np)
        image_tensor = transformed['image']

        # Run inference
        detections = stitcher(image_tensor)

        # Convert to dataframe format
        n_dets = len(detections['scores'])
        if n_dets > 0:
            for i in range(n_dets):
                all_detections.append({
                    'images': img_file,
                    'x': float(detections['boxes'][i, 0]),
                    'y': float(detections['boxes'][i, 1]),
                    'x_max': float(detections['boxes'][i, 2]),
                    'y_max': float(detections['boxes'][i, 3]),
                    'labels': int(detections['labels'][i]),
                    'scores': float(detections['scores'][i]),
                })

    return pd.DataFrame(all_detections)


def calculate_metrics(
    detections_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    radius: float = 20.0
) -> Dict:
    """Calculate detection metrics.

    Args:
        detections_df: DataFrame with detections
        ground_truth_df: DataFrame with ground truth annotations
        radius: Matching radius in pixels

    Returns:
        Dictionary with precision, recall, f1_score, TP, FP, FN
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Group by image
    for img_name in ground_truth_df['images'].unique():
        # Get detections and GT for this image
        img_dets = detections_df[detections_df['images'] == img_name]
        img_gt = ground_truth_df[ground_truth_df['images'] == img_name]

        # Convert boxes to center points for GT
        gt_centers = np.column_stack([
            (img_gt['x_min'].values + img_gt['x_max'].values) / 2,
            (img_gt['y_min'].values + img_gt['y_max'].values) / 2,
        ])

        det_centers = np.column_stack([
            img_dets['x'].values,
            img_dets['y'].values,
        ]) if len(img_dets) > 0 else np.empty((0, 2))

        # Match detections to ground truth
        matched_gt = set()
        for det_idx, det_center in enumerate(det_centers):
            if len(gt_centers) > 0:
                distances = np.linalg.norm(gt_centers - det_center, axis=1)
                min_dist_idx = np.argmin(distances)
                min_dist = distances[min_dist_idx]

                if min_dist <= radius and min_dist_idx not in matched_gt:
                    true_positives += 1
                    matched_gt.add(min_dist_idx)
                else:
                    false_positives += 1
            else:
                false_positives += 1

        # Unmatched GT are false negatives
        false_negatives += len(gt_centers) - len(matched_gt)

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
    }


def calculate_f1_at_thresholds(
    yaml_cfg: OmegaConf,
    checkpoint_path: Path,
    confidence_thresholds: np.ndarray,
    device: str = "cuda",
) -> Dict:
    """Calculate F1 score at different confidence thresholds.

    Args:
        yaml_cfg: Configuration object
        checkpoint_path: Path to model checkpoint
        confidence_thresholds: Array of confidence thresholds to evaluate
        device: Device to run inference on

    Returns:
        Dictionary with thresholds and corresponding metrics
    """
    results = {
        'thresholds': [],
        'f1_scores': [],
        'precisions': [],
        'recalls': [],
        'num_predictions': [],
        'true_positives': [],
        'false_positives': [],
        'false_negatives': []
    }

    # Load ground truth once
    gt_df = pd.read_csv(yaml_cfg.data.test_csv)

    # Normalize column names
    column_mapping = {}
    if 'Image' in gt_df.columns:
        column_mapping['Image'] = 'images'
    if 'x1' in gt_df.columns:
        column_mapping['x1'] = 'x_min'
    if 'y1' in gt_df.columns:
        column_mapping['y1'] = 'y_min'
    if 'x2' in gt_df.columns:
        column_mapping['x2'] = 'x_max'
    if 'y2' in gt_df.columns:
        column_mapping['y2'] = 'y_max'

    if column_mapping:
        gt_df = gt_df.rename(columns=column_mapping)

    logger.info(f"Total ground truths: {len(gt_df)}")

    # Run inference at each threshold
    for threshold in tqdm(confidence_thresholds, desc="Evaluating thresholds"):
        logger.info(f"\nRunning inference at threshold {threshold:.2f}")

        # Run inference
        detections_df = run_inference_at_threshold(
            yaml_cfg, checkpoint_path, threshold, device
        )

        # Calculate metrics
        metrics = calculate_metrics(
            detections_df, gt_df, yaml_cfg.evaluator.metrics_radius
        )

        # Store results
        results['thresholds'].append(float(threshold))
        results['f1_scores'].append(metrics['f1_score'])
        results['precisions'].append(metrics['precision'])
        results['recalls'].append(metrics['recall'])
        results['num_predictions'].append(len(detections_df))
        results['true_positives'].append(metrics['true_positives'])
        results['false_positives'].append(metrics['false_positives'])
        results['false_negatives'].append(metrics['false_negatives'])

        logger.info(
            f"Threshold {threshold:.2f}: "
            f"F1={metrics['f1_score']:.4f}, "
            f"P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}, "
            f"Predictions={len(detections_df)}"
        )

    return results


def plot_confidence_vs_f1(results: Dict, output_path: Path):
    """Plot confidence threshold vs F1 score and related metrics.

    Args:
        results: Dictionary with evaluation results
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: F1 vs Confidence
    axes[0, 0].plot(results['thresholds'], results['f1_scores'],
                     marker='o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Confidence Threshold')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].set_title('F1 Score vs Confidence Threshold')
    axes[0, 0].grid(True, alpha=0.3)

    # Find and mark best F1
    best_idx = np.argmax(results['f1_scores'])
    best_threshold = results['thresholds'][best_idx]
    best_f1 = results['f1_scores'][best_idx]
    axes[0, 0].axvline(best_threshold, color='r', linestyle='--', alpha=0.5,
                        label=f'Best: {best_threshold:.2f} (F1={best_f1:.4f})')
    axes[0, 0].legend()

    # Plot 2: Precision and Recall vs Confidence
    axes[0, 1].plot(results['thresholds'], results['precisions'],
                     marker='o', label='Precision', linewidth=2, markersize=4)
    axes[0, 1].plot(results['thresholds'], results['recalls'],
                     marker='s', label='Recall', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Confidence Threshold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision & Recall vs Confidence Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(best_threshold, color='r', linestyle='--', alpha=0.5)

    # Plot 3: Number of predictions vs Confidence
    axes[1, 0].plot(results['thresholds'], results['num_predictions'],
                     marker='o', linewidth=2, markersize=4, color='green')
    axes[1, 0].set_xlabel('Confidence Threshold')
    axes[1, 0].set_ylabel('Number of Predictions')
    axes[1, 0].set_title('Number of Predictions vs Confidence Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(best_threshold, color='r', linestyle='--', alpha=0.5)

    # Plot 4: TP, FP, FN vs Confidence
    axes[1, 1].plot(results['thresholds'], results['true_positives'],
                     marker='o', label='True Positives', linewidth=2, markersize=4)
    axes[1, 1].plot(results['thresholds'], results['false_positives'],
                     marker='s', label='False Positives', linewidth=2, markersize=4)
    axes[1, 1].plot(results['thresholds'], results['false_negatives'],
                     marker='^', label='False Negatives', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('Confidence Threshold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('TP/FP/FN vs Confidence Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(best_threshold, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")

    return best_threshold, best_f1


def main(
    config: str,
    checkpoint: str = None,
    output_dir: str = "./outputs/confidence_analysis",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.05,
    device: str = "cuda",
):
    """Main function to calculate and plot confidence vs F1 scores.

    Args:
        config: Path to config file
        checkpoint: Path to checkpoint (overrides config)
        output_dir: Directory to save outputs
        min_threshold: Minimum confidence threshold
        max_threshold: Maximum confidence threshold
        step: Step size for thresholds
        device: Device to use
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load config
    yaml_cfg = OmegaConf.load(config)

    # Remove model_type if present
    if 'model_type' in yaml_cfg:
        from animaldet.experiments.rfdetr.adapters.config import RFDETRExperimentConfig
        yaml_cfg_clean = OmegaConf.create(OmegaConf.to_container(yaml_cfg))
        del yaml_cfg_clean['model_type']
        yaml_cfg = OmegaConf.merge(
            OmegaConf.structured(RFDETRExperimentConfig), yaml_cfg_clean
        )

    # Override checkpoint if provided
    if checkpoint:
        yaml_cfg.inference.checkpoint_path = checkpoint

    # Validate paths
    checkpoint_path = Path(yaml_cfg.inference.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_csv_path = Path(yaml_cfg.data.test_csv)
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Test CSV: {test_csv_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Device: {device}")

    # Generate confidence thresholds
    confidence_thresholds = np.arange(min_threshold, max_threshold + step, step)
    logger.info(f"Evaluating {len(confidence_thresholds)} confidence thresholds: "
                f"{min_threshold} to {max_threshold} (step={step})")

    # Calculate F1 at different thresholds
    results = calculate_f1_at_thresholds(
        yaml_cfg, checkpoint_path, confidence_thresholds, device
    )

    # Plot results
    plot_path = output_path / "confidence_vs_f1.png"
    best_threshold, best_f1 = plot_confidence_vs_f1(results, plot_path)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Best confidence threshold: {best_threshold:.2f}")
    logger.info(f"Best F1 score: {best_f1:.4f}")

    best_idx = np.argmax(results['f1_scores'])
    logger.info(f"Precision at best threshold: {results['precisions'][best_idx]:.4f}")
    logger.info(f"Recall at best threshold: {results['recalls'][best_idx]:.4f}")
    logger.info(f"Predictions at best threshold: {results['num_predictions'][best_idx]}")
    logger.info(f"True Positives: {results['true_positives'][best_idx]}")
    logger.info(f"False Positives: {results['false_positives'][best_idx]}")
    logger.info(f"False Negatives: {results['false_negatives'][best_idx]}")

    # Save results to JSON
    import json
    results_file = output_path / "confidence_vs_f1_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Save summary to CSV
    summary_file = output_path / "confidence_vs_f1_summary.csv"
    pd.DataFrame(results).to_csv(summary_file, index=False)
    logger.info(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python confidence_vs_f1.py <config> [--checkpoint path] [--output-dir path]")
        print("       [--min-threshold 0.05] [--max-threshold 0.95] [--step 0.05]")
        sys.exit(1)

    config_arg = sys.argv[1]
    checkpoint_arg = None
    output_dir_arg = "./outputs/confidence_analysis"
    min_threshold_arg = 0.05
    max_threshold_arg = 0.95
    step_arg = 0.05
    device_arg = "cuda"

    # Simple argument parsing
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--checkpoint' and i + 1 < len(sys.argv):
            checkpoint_arg = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--output-dir' and i + 1 < len(sys.argv):
            output_dir_arg = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--min-threshold' and i + 1 < len(sys.argv):
            min_threshold_arg = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--max-threshold' and i + 1 < len(sys.argv):
            max_threshold_arg = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--step' and i + 1 < len(sys.argv):
            step_arg = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--device' and i + 1 < len(sys.argv):
            device_arg = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    main(
        config=config_arg,
        checkpoint=checkpoint_arg,
        output_dir=output_dir_arg,
        min_threshold=min_threshold_arg,
        max_threshold=max_threshold_arg,
        step=step_arg,
        device=device_arg,
    )
