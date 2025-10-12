#!/usr/bin/env python3
"""RF-DETR inference using stitcher for large images.

This script handles inference on large images by automatically dividing them
into patches at the model's expected resolution (560x560), running inference,
and rescaling predictions back to the original image coordinates.
"""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from animaldet.experiments.rfdetr.adapters.model import build_model
from animaldet.experiments.rfdetr.adapters.config import RFDETRExperimentConfig
from animaldet.experiments.rfdetr.stitcher import RFDETRStitcher

logger = logging.getLogger(__name__)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """Load model checkpoint.

    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    return model


def inference_main(
    config: Optional[str] = None,
    checkpoint: Optional[str] = None,
    images_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    threshold: Optional[float] = None,
    device: str = "cuda",
    batch_size: Optional[int] = None,
):
    """RF-DETR inference on large images using stitcher.

    Args:
        config: Path to config file
        checkpoint: Path to checkpoint file
        images_dir: Directory containing images for inference
        output_csv: Path to output CSV file
        threshold: Detection confidence threshold
        device: Device to use ('cuda' or 'cpu')
        batch_size: Batch size for patch inference
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load config
    config = config or "configs/inference/rfdetr_test.yaml"
    yaml_cfg = OmegaConf.load(config)
    cfg = OmegaConf.merge(OmegaConf.structured(RFDETRExperimentConfig), yaml_cfg)

    # Override with args
    if checkpoint:
        cfg.inference.checkpoint_path = checkpoint
    if images_dir:
        cfg.data.test_root = images_dir
    if output_csv:
        cfg.inference.output_path = Path(output_csv).parent
    if threshold is not None:
        cfg.inference.threshold = threshold
    if batch_size is not None:
        cfg.inference.batch_size = batch_size

    # Validate paths
    checkpoint_path = Path(cfg.inference.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load test CSV
    test_csv_path = Path(cfg.data.test_csv)
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    output_path = Path(cfg.inference.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Test CSV: {test_csv_path}")
    logger.info(f"Output: {output_path}")

    # Build and load model
    logger.info("Building RF-DETR model...")
    model = build_model(cfg.model, device=device)
    model = load_checkpoint(model, str(checkpoint_path))
    model.eval()

    # Load and normalize CSV
    df = pd.read_csv(test_csv_path)

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
    logger.info(f"Processing {len(image_files)} images...")

    # Create stitcher
    stitcher = RFDETRStitcher(
        model=model,
        size=(cfg.model.resolution, cfg.model.resolution),
        overlap=0,  # Non-overlapping patches
        batch_size=cfg.inference.batch_size,
        confidence_threshold=cfg.inference.threshold,
        nms_threshold=cfg.evaluator.nms_threshold,
        device_name=device,
    )

    # Create transforms for preprocessing
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Run inference on all images
    all_detections = []
    for img_file in image_files:
        img_path = Path(cfg.data.test_root) / img_file

        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue

        # Load image
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)

        # Apply transforms
        transformed = transform(image=image_np)
        image_tensor = transformed['image']  # [C, H, W]

        logger.info(f"Processing {img_file} (size: {image_tensor.shape[1]}x{image_tensor.shape[2]})")

        # Run inference with stitcher
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

        logger.info(f"  Found {n_dets} detections")

    # Save detections
    detections_df = pd.DataFrame(all_detections)
    detections_path = output_path / cfg.inference.detections_csv

    if len(detections_df) > 0:
        detections_df.to_csv(detections_path, index=False)
        logger.info(f"Saved {len(detections_df)} detections to {detections_path}")
    else:
        logger.warning("No detections found!")
        # Save empty CSV with headers
        pd.DataFrame(columns=['images', 'x', 'y', 'x_max', 'y_max', 'labels', 'scores']).to_csv(
            detections_path, index=False
        )

    # Calculate and save metrics if ground truth is available
    if 'x_min' in df.columns and 'y_min' in df.columns:
        logger.info("\\nCalculating metrics...")
        metrics = calculate_metrics(detections_df, df, cfg.evaluator.metrics_radius)

        # Save metrics
        results_path = output_path / cfg.inference.results_csv
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(results_path, index=False)

        logger.info(f"\\n{'='*60}")
        logger.info("EVALUATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"{'='*60}")
        logger.info(f"Results saved to: {results_path}")

    return detections_df


def calculate_metrics(
    detections_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    radius: float = 20.0
) -> dict:
    """Calculate detection metrics.

    Args:
        detections_df: DataFrame with detections (images, x, y, labels, scores)
        ground_truth_df: DataFrame with ground truth (images, x_min, y_min, x_max, y_max, labels)
        radius: Matching radius for point-based metrics

    Returns:
        Dictionary with precision, recall, and f1_score
    """
    # Simple point-based matching using box centers
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Group by image
    for img_name in ground_truth_df['images'].unique():
        # Get detections for this image
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
            # Find closest GT
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference_cmd.py <checkpoint> [config] [output_csv]")
        sys.exit(1)

    checkpoint_arg = sys.argv[1]
    config_arg = sys.argv[2] if len(sys.argv) > 2 else None
    output_arg = sys.argv[3] if len(sys.argv) > 3 else None

    inference_main(
        checkpoint=checkpoint_arg,
        config=config_arg,
        output_csv=output_arg
    )
