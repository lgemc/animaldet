#!/usr/bin/env python3
"""
Inference command for animaldet CLI.

This module provides inference capabilities for running HerdNet models on full-size images
using the stitcher to handle large images by dividing them into patches.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from animaldet.experiments.herdnet.adapters.model import build_model
from animaldet.experiments.herdnet.adapters.config import HerdNetExperimentConfig
from animaloc.eval import HerdNetStitcher


logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> HerdNetExperimentConfig:
    """Load and parse configuration file."""
    if config_path is None:
        # Use default config
        config_path = "configs/inference/herdnet.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML config
    yaml_cfg = OmegaConf.load(config_path)

    # Merge with structured config
    cfg = OmegaConf.structured(HerdNetExperimentConfig)
    cfg = OmegaConf.merge(cfg, yaml_cfg)

    return cfg


def load_model_and_stitcher(cfg: HerdNetExperimentConfig, checkpoint_path: str, device: str):
    """Load model and create stitcher."""
    # Build model
    model = build_model(cfg.model, device=device)

    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip 'model.' prefix from keys if present (checkpoint contains full LossWrapper state)
    if any(k.startswith('model.') for k in state_dict.keys()):
        # Remove 'model.' prefix to match HerdNet architecture
        state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}

    # Load into the wrapped HerdNet model
    model.model.load_state_dict(state_dict, strict=True)
    model.eval()

    logger.info("Model loaded successfully")

    # Create stitcher
    stitcher = HerdNetStitcher(
        model=model,
        size=(cfg.data.patch_size, cfg.data.patch_size),
        overlap=cfg.evaluator.stitcher_overlap,
        down_ratio=cfg.model.down_ratio,
        reduction=cfg.evaluator.stitcher_reduction
    )

    return model, stitcher


def run_inference_on_image(
    image_path: Path,
    stitcher: HerdNetStitcher,
    threshold: float,
    device: str
) -> pd.DataFrame:
    """Run inference on a single full-size image using the stitcher.

    Args:
        image_path: Path to the image
        stitcher: HerdNetStitcher instance
        threshold: Detection threshold
        device: Device to use

    Returns:
        DataFrame with columns: image, x, y, confidence, class
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)

    # Convert to tensor and apply ImageNet normalization (same as A.Normalize())
    image_tensor = torch.from_numpy(image_np).float() / 255.0
    # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    image_tensor = image_tensor.to(device)  # (C, H, W) - no batch dimension

    # Run stitcher
    with torch.no_grad():
        output = stitcher(image_tensor)  # Returns single concatenated tensor

    # Split output into heatmap and class predictions
    # Output shape: [1, C_combined, H, W] where C_combined = 1 (heatmap) + num_classes (class map)
    output = output[0]  # Remove batch dimension: [C_combined, H, W]
    heatmap = output[0].cpu().numpy()  # First channel is heatmap: (H, W)
    class_map = output[1:].cpu().numpy()  # Remaining channels are class predictions: (num_classes, H, W)

    # Find peaks above threshold
    mask = heatmap > threshold
    coords = np.argwhere(mask)  # (N, 2) in (y, x) format

    if len(coords) == 0:
        # No detections
        return pd.DataFrame(columns=['image', 'x', 'y', 'confidence', 'class'])

    # Get confidence scores
    confidences = heatmap[mask]

    # Scale coordinates by down_ratio to get original image coordinates
    down_ratio = stitcher.down_ratio
    coords = coords.astype(float) * down_ratio

    # Get class predictions at detected points
    # class_map shape: (num_classes, H, W)
    # coords are in original image space, need to scale to heatmap space
    scaled_coords = (coords / down_ratio).astype(int)
    scaled_coords[:, 0] = np.clip(scaled_coords[:, 0], 0, class_map.shape[1] - 1)
    scaled_coords[:, 1] = np.clip(scaled_coords[:, 1], 0, class_map.shape[2] - 1)

    # Extract class predictions at detection locations
    classes = class_map[:, scaled_coords[:, 0], scaled_coords[:, 1]]  # (num_classes, N)
    classes = np.argmax(classes, axis=0)  # (N,)

    # Create DataFrame
    results = pd.DataFrame({
        'image': image_path.name,
        'x': coords[:, 1],  # x = column
        'y': coords[:, 0],  # y = row
        'confidence': confidences,
        'class': classes
    })

    return results


def inference_main(
    config: Optional[str] = None,
    checkpoint: Optional[str] = None,
    images_dir: Optional[str] = None,
    output_csv: Optional[str] = None,
    threshold: Optional[float] = None,
    device: str = "cuda",
    batch_size: Optional[int] = None,
):
    """Main inference function.

    Args:
        config: Path to inference configuration file
        checkpoint: Path to model checkpoint (overrides config)
        images_dir: Directory containing images (overrides config)
        output_csv: Path to output CSV file (default: predictions.csv)
        threshold: Detection threshold (overrides config)
        device: Device to use for inference
        batch_size: Batch size (currently unused, reserved for future)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    logger.info("Loading configuration...")
    cfg = load_config(config)

    # Override config with CLI arguments
    if checkpoint is not None:
        cfg.inference.checkpoint_path = checkpoint
    if threshold is not None:
        cfg.inference.threshold = threshold
    if device is not None:
        cfg.inference.device = device

    # Validate checkpoint path
    if cfg.inference.checkpoint_path is None:
        raise ValueError("Checkpoint path must be specified via config or --checkpoint")

    checkpoint_path = Path(cfg.inference.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Setup images directory and get image list
    if images_dir is not None:
        images_dir = Path(images_dir)
    else:
        # Try to use test_root from config
        if hasattr(cfg.data, 'test_root') and cfg.data.test_root is not None:
            images_dir = Path(cfg.data.test_root)
        else:
            raise ValueError("Images directory must be specified via config or --images-dir")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Setup output CSV path
    if output_csv is None:
        # Use config values if available
        if hasattr(cfg.inference, 'output_path') and hasattr(cfg.inference, 'output_file'):
            output_csv = Path(cfg.inference.output_path) / cfg.inference.output_file
        elif hasattr(cfg.inference, 'output_file'):
            output_csv = Path(cfg.inference.output_file)
        else:
            output_csv = Path("predictions.csv")
    else:
        output_csv = Path(output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Configuration loaded:")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Images directory: {images_dir}")
    logger.info(f"  Output CSV: {output_csv}")
    logger.info(f"  Threshold: {cfg.inference.threshold}")
    logger.info(f"  Device: {cfg.inference.device}")

    # Load model and stitcher
    logger.info("Loading model and stitcher...")
    model, stitcher = load_model_and_stitcher(
        cfg,
        checkpoint_path,
        cfg.inference.device
    )

    # Get list of images to process
    # If CSV is provided in config, extract unique image names from it
    image_files = []
    if hasattr(cfg.data, 'test_csv') and cfg.data.test_csv is not None:
        csv_path = Path(cfg.data.test_csv)
        if csv_path.exists():
            logger.info(f"Reading image list from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            # Get unique image names from the 'Image' column
            unique_images = df['Image'].unique()
            image_files = [images_dir / img for img in unique_images]
            # Filter out non-existent files
            image_files = [f for f in image_files if f.exists()]
            logger.info(f"Found {len(image_files)} unique images from CSV")

    # If no images from CSV, scan directory
    if len(image_files) == 0:
        logger.info("Scanning images directory...")
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])

    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")

    logger.info(f"Processing {len(image_files)} images")

    # Run inference on all images
    all_results = []

    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            results = run_inference_on_image(
                image_path,
                stitcher,
                cfg.inference.threshold,
                cfg.inference.device
            )

            if len(results) > 0:
                all_results.append(results)

        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            continue

    # Combine all results
    if len(all_results) > 0:
        final_results = pd.concat(all_results, ignore_index=True)
    else:
        final_results = pd.DataFrame(columns=['image', 'x', 'y', 'confidence', 'class'])

    # Save to CSV
    final_results.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(final_results)} detections to {output_csv}")

    # Print summary
    logger.info("\nInference Summary:")
    logger.info(f"  Total images processed: {len(image_files)}")
    logger.info(f"  Total detections: {len(final_results)}")
    if len(final_results) > 0:
        logger.info(f"  Detections per image (avg): {len(final_results) / len(image_files):.1f}")
        logger.info(f"  Confidence range: [{final_results['confidence'].min():.3f}, {final_results['confidence'].max():.3f}]")

        # Class distribution
        class_counts = final_results['class'].value_counts().sort_index()
        logger.info("  Class distribution:")
        for class_id, count in class_counts.items():
            logger.info(f"    Class {class_id}: {count} detections")


if __name__ == "__main__":
    import sys

    # Simple command-line interface for testing
    if len(sys.argv) < 4:
        print("Usage: python inference_cmd.py <config> <checkpoint> <images_dir> [output_csv]")
        sys.exit(1)

    config = sys.argv[1]
    checkpoint = sys.argv[2]
    images_dir = sys.argv[3]
    output_csv = sys.argv[4] if len(sys.argv) > 4 else "predictions.csv"

    inference_main(
        config=config,
        checkpoint=checkpoint,
        images_dir=images_dir,
        output_csv=output_csv
    )
