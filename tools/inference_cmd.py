

#!/usr/bin/env python3
"""Unified inference for HerdNet and RF-DETR models using stitchers."""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd

import albumentations as A
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from animaldet.utils import get_device

logger = logging.getLogger(__name__)


# Custom transform to convert boxes to points (center of bbox)
class BoxesToPoints:
    """Convert bounding boxes to center points for HerdNet evaluation.

    Note: This transform runs BEFORE DataLoader collation.
    The boxes tensor from SampleToTensor is shaped [N, 4].
    We convert to points [N, 2], and DataLoader will add batch dim -> [1, N, 2].
    """

    def __call__(self, image, target):
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']  # Shape: [N, 4]

            # Calculate center points: (x_min + x_max) / 2, (y_min + y_max) / 2
            points = torch.stack([
                (boxes[:, 0] + boxes[:, 2]) / 2,  # x center
                (boxes[:, 1] + boxes[:, 3]) / 2  # y center
            ], dim=1)  # Shape: [N, 2]

            # Store as [N, 2] - DataLoader will add batch dimension
            target['points'] = points
        else:
            # No boxes, create empty points tensor [0, 2]
            target['points'] = torch.empty((0, 2))
        return image, target

def inference_main(
    config: Optional[str] = None,
    checkpoint: Optional[str] = None,
    images_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    threshold: Optional[float] = None,
    device: str = "cuda",
    batch_size: Optional[int] = None,
    model_type: Optional[str] = None,
):
    """Unified inference for HerdNet and RF-DETR models.

    Args:
        config: Path to config file
        checkpoint: Path to checkpoint file
        images_dir: Directory containing images for inference
        output_csv: Path to output CSV file
        threshold: Detection threshold
        device: Device to use
        batch_size: Batch size for inference
        model_type: Model type ('herdnet' or 'rfdetr', overrides config)
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load config
    if config is None:
        raise ValueError("Config file is required. Use --config to specify.")

    yaml_cfg = OmegaConf.load(config)

    # Determine model type from config or argument
    if model_type is None:
        if 'model_type' in yaml_cfg:
            model_type = yaml_cfg.model_type
        else:
            # Try to infer from config structure
            if 'model' in yaml_cfg and 'variant' in yaml_cfg.model:
                model_type = 'rfdetr'
            elif 'model' in yaml_cfg and 'down_ratio' in yaml_cfg.model:
                model_type = 'herdnet'
            else:
                raise ValueError(
                    "Could not determine model type. Add 'model_type: herdnet' or 'model_type: rfdetr' "
                    "to your config, or use --model-type argument."
                )

    model_type = model_type.lower()

    if model_type == 'herdnet':
        return _inference_herdnet(
            yaml_cfg, checkpoint, images_dir, output_csv, threshold, device, batch_size
        )
    elif model_type == 'rfdetr':
        return _inference_rfdetr(
            yaml_cfg, checkpoint, images_dir, output_csv, threshold, device, batch_size
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'herdnet' or 'rfdetr'.")


def _inference_herdnet(
    yaml_cfg: OmegaConf,
    checkpoint: Optional[str] = None,
    images_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    threshold: Optional[float] = None,
    device: str = "cuda",
    batch_size: Optional[int] = None,
):
    """HerdNet inference implementation."""
    from animaldet.experiments.herdnet.adapters.model import build_model
    from animaldet.experiments.herdnet.adapters.config import HerdNetExperimentConfig
    from animaloc.datasets import CSVDataset
    from animaloc.data.transforms import DownSample
    from animaloc.eval import HerdNetStitcher, HerdNetEvaluator, PointsMetrics
    from animaloc.models import load_model

    # Remove model_type from yaml_cfg before merging (it's only for routing)
    yaml_cfg_clean = OmegaConf.create(OmegaConf.to_container(yaml_cfg))
    if 'model_type' in yaml_cfg_clean:
        del yaml_cfg_clean['model_type']

    cfg = OmegaConf.merge(OmegaConf.structured(HerdNetExperimentConfig), yaml_cfg_clean)

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

    output_path = Path(cfg.inference.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output: {output_path}")

    # Get device using centralized utility
    actual_device = get_device(device)

    # Build and load model
    logger.info(f"Building model...")
    model = build_model(cfg.model, device=str(actual_device))
    model = load_model(model, pth_path=str(checkpoint_path))

    # Load and normalize CSV columns (handle both Image/Label and images/labels)
    import pandas as pd
    df = pd.read_csv(cfg.data.test_csv)

    # Rename columns to match expected format
    column_mapping = {}
    if 'Image' in df.columns:
        column_mapping['Image'] = 'images'
    if 'Label' in df.columns:
        column_mapping['Label'] = 'labels'
    # Handle coordinate columns (x1,y1,x2,y2 -> x_min,y_min,x_max,y_max)
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
        # Save normalized CSV to temp file
        from tempfile import NamedTemporaryFile
        temp_csv = NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_csv.name, index=False)
        csv_file = temp_csv.name
    else:
        csv_file = cfg.data.test_csv

    # Create dataset and dataloader
    # Note: For inference on full images, we don't need DownSample transform
    # The stitcher handles patching and the model handles downsampling internally
    dataset = CSVDataset(
        csv_file=csv_file,
        root_dir=cfg.data.test_root,
        albu_transforms=[A.Normalize(p=1.0)],
        end_transforms=[
            BoxesToPoints(),
            DownSample(down_ratio=cfg.model.down_ratio, anno_type='point'),
        ]  # Convert boxes to points for evaluation
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Create stitcher
    stitcher = HerdNetStitcher(
        model=model,
        size=(cfg.data.patch_size, cfg.data.patch_size),
        overlap=cfg.evaluator.stitcher_overlap,
        down_ratio=cfg.model.down_ratio,
        reduction=cfg.evaluator.stitcher_reduction,
    )

    # Create metrics
    metrics = PointsMetrics(
        radius=cfg.evaluator.metrics_radius,
        num_classes=cfg.model.num_classes
    )

    print(f"Using radius: {cfg.evaluator.metrics_radius}, number of classes: {cfg.model.num_classes}")
    # Create evaluator and run (using default LMDS thresholds as per HerdNet notebook)
    evaluator = HerdNetEvaluator(
        model=model,
        dataloader=dataloader,
        metrics=metrics,
        stitcher=stitcher,
        work_dir=str(output_path),
        header='test'
    )

    logger.info("Running inference...")
    f1_score = evaluator.evaluate(returns='f1_score')

    # Log detailed F1 metrics
    results_df = evaluator.results
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"\nOverall F1 Score: {f1_score * 100:.2f}%")

    # Log per-class metrics
    logger.info(f"\n{'Per-Class Metrics:':^60}")
    logger.info(f"{'-'*60}")
    logger.info(f"{'Class':<10} {'N':>8} {'Recall':>10} {'Precision':>10} {'F1-Score':>10}")
    logger.info(f"{'-'*60}")

    for _, row in results_df.iterrows():
        logger.info(
            f"{row['class']:<10} {row['n']:>8.0f} {row['recall']:>10.4f} "
            f"{row['precision']:>10.4f} {row['f1_score']:>10.4f}"
        )

    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {output_path}")

    # Save detections and results using configured filenames
    detections_path = output_path / cfg.inference.detections_csv
    results_path = output_path / cfg.inference.results_csv

    re_scale_detections(evaluator).to_csv(detections_path)
    logger.info(f"Detections CSV: {detections_path}")

    results_df.to_csv(results_path, index=False)
    logger.info(f"Metrics CSV: {results_path}")

    return results_df

def re_scale_detections(evaluator) -> pd.DataFrame:
    ''' Returns detections (image id, location, label and score) in a pandas
    dataframe with coordinates scaled to original image space when using stitcher '''

    assert evaluator._stored_metrics is not None, \
        'No detections have been stored, please use the evaluate method first.'

    img_names = evaluator.dataloader.dataset._img_names
    dets = evaluator._stored_metrics.detections.copy()

    for det in dets:
        det['images'] = img_names[det['images']]

        # Scale coordinates to original image space when using stitcher
        if evaluator.stitcher is not None and 'x' in det and 'y' in det:
            det['x'] = det['x'] * evaluator.stitcher.down_ratio
            det['y'] = det['y'] * evaluator.stitcher.down_ratio

    return pd.DataFrame(data = dets)


def _inference_rfdetr(
    yaml_cfg: OmegaConf,
    checkpoint: Optional[str] = None,
    images_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    threshold: Optional[float] = None,
    device: str = "cuda",
    batch_size: Optional[int] = None,
):
    """RF-DETR inference implementation."""
    from animaldet.experiments.rfdetr.adapters.model import build_model
    from animaldet.experiments.rfdetr.adapters.config import RFDETRExperimentConfig
    from animaldet.experiments.rfdetr.stitcher import RFDETRStitcher
    from PIL import Image
    from albumentations.pytorch import ToTensorV2
    import numpy as np

    # Remove model_type from yaml_cfg before merging (it's only for routing)
    yaml_cfg_clean = OmegaConf.create(OmegaConf.to_container(yaml_cfg))
    if 'model_type' in yaml_cfg_clean:
        del yaml_cfg_clean['model_type']

    cfg = OmegaConf.merge(OmegaConf.structured(RFDETRExperimentConfig), yaml_cfg_clean)

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

    test_csv_path = Path(cfg.data.test_csv)
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    output_path = Path(cfg.inference.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Test CSV: {test_csv_path}")
    logger.info(f"Output: {output_path}")

    # Get device using centralized utility
    actual_device = get_device(device)

    # Load checkpoint first to get the actual architecture
    logger.info(f"Loading checkpoint to inspect architecture...")
    checkpoint = torch.load(checkpoint_path, map_location=actual_device, weights_only=False)

    # Extract architecture parameters from checkpoint args (with fallbacks to config)
    ckpt_args = checkpoint['args']

    # Use checkpoint values if available, otherwise keep config values
    cfg.model.num_classes = getattr(ckpt_args, 'num_classes', cfg.model.num_classes)
    cfg.model.hidden_dim = getattr(ckpt_args, 'hidden_dim', cfg.model.hidden_dim)
    cfg.model.dec_layers = getattr(ckpt_args, 'dec_layers', cfg.model.dec_layers)
    cfg.model.sa_nheads = getattr(ckpt_args, 'sa_nheads', cfg.model.sa_nheads)
    cfg.model.ca_nheads = getattr(ckpt_args, 'ca_nheads', cfg.model.ca_nheads)
    cfg.model.dec_n_points = getattr(ckpt_args, 'dec_n_points', cfg.model.dec_n_points)
    cfg.model.num_queries = getattr(ckpt_args, 'num_queries', cfg.model.num_queries)
    cfg.model.num_select = getattr(ckpt_args, 'num_select', cfg.model.num_select)
    cfg.model.encoder = getattr(ckpt_args, 'encoder', cfg.model.encoder)
    cfg.model.patch_size = getattr(ckpt_args, 'patch_size', cfg.model.patch_size)
    cfg.model.num_windows = getattr(ckpt_args, 'num_windows', cfg.model.num_windows)
    cfg.model.out_feature_indexes = getattr(ckpt_args, 'out_feature_indexes', cfg.model.out_feature_indexes)
    cfg.model.projector_scale = getattr(ckpt_args, 'projector_scale', cfg.model.projector_scale)
    cfg.model.positional_encoding_size = getattr(ckpt_args, 'positional_encoding_size', cfg.model.positional_encoding_size)

    # Check actual num_classes in checkpoint weights (RF-DETR adds +1 internally)
    state_dict = checkpoint.get('model', checkpoint.get('ema_model'))
    actual_num_classes_in_weights = state_dict['class_embed.weight'].shape[0]

    # Adjust num_classes: RF-DETR will add +1, so we pass (actual - 1) to get actual
    cfg.model.num_classes = actual_num_classes_in_weights - 1

    logger.info(f"Model architecture: num_classes={cfg.model.num_classes} (+1 background = {actual_num_classes_in_weights}), "
                f"hidden_dim={cfg.model.hidden_dim}, dec_layers={cfg.model.dec_layers}, "
                f"ca_nheads={cfg.model.ca_nheads}, dec_n_points={cfg.model.dec_n_points}")

    # Build model with checkpoint's architecture (skip head reinitialization)
    logger.info(f"Building RF-DETR model with checkpoint architecture...")
    model = build_model(cfg.model, device=str(actual_device), reinit_head=False)

    # Load checkpoint with strict=True since architecture now matches
    # Use normal model weights
    if 'model' in checkpoint:
        logger.info("Loading normal model weights")
        model.load_state_dict(checkpoint['model'], strict=True)
    else:
        logger.info("Normal model not found in checkpoint, using EMA model")
        model.load_state_dict(checkpoint['ema_model'], strict=True)

    model = model.to(actual_device)
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
        overlap=0,
        batch_size=cfg.inference.batch_size,
        confidence_threshold=cfg.inference.threshold,
        nms_threshold=cfg.evaluator.nms_threshold,
        device_name=str(actual_device),
        voting_threshold=0.5,
    )

    # Create transforms for preprocessing
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Run inference on all images
    all_detections = []
    for img_file in tqdm(image_files, desc="Processing images", unit="img"):
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
        logger.info("\nCalculating metrics...")
        metrics = _calculate_metrics_rfdetr(detections_df, df, cfg.evaluator.metrics_radius)

        # Save metrics
        results_path = output_path / cfg.inference.results_csv
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(results_path, index=False)

        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"{'='*60}")
        logger.info(f"Results saved to: {results_path}")

    return detections_df


def _calculate_metrics_rfdetr(
    detections_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    radius: float = 20.0
) -> dict:
    """Calculate detection metrics for RF-DETR."""
    import numpy as np

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
        print("Usage: python inference_cmd.py <config> [--checkpoint path] [--model-type herdnet|rfdetr]")
        sys.exit(1)

    config_arg = sys.argv[1]
    checkpoint_arg = None
    model_type_arg = None

    # Simple argument parsing
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--checkpoint' and i + 1 < len(sys.argv):
            checkpoint_arg = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--model-type' and i + 1 < len(sys.argv):
            model_type_arg = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    inference_main(
        config=config_arg,
        checkpoint=checkpoint_arg,
        model_type=model_type_arg
    )
