#!/usr/bin/env python3
"""Run inference with a trained model.

This script demonstrates how to use the inference system with config files.

Example usage:
    # Run inference with HerdNet
    python scripts/run_inference.py --config configs/inference/herdnet.yaml

    # Override checkpoint path
    python scripts/run_inference.py --config configs/inference/herdnet.yaml \
        inference.checkpoint_path=./output/model_epoch_50.pth

    # Change detection threshold
    python scripts/run_inference.py --config configs/inference/herdnet.yaml \
        inference.threshold=0.7
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import numpy as np
from PIL import Image

from animaldet.engine import INFERENCE_BUILDERS
from animaldet.experiments.herdnet import HerdNetInference


@hydra.main(version_base=None, config_path="../configs/inference", config_name="herdnet")
def main(cfg: DictConfig) -> None:
    """Run inference from config file.

    Args:
        cfg: Hydra configuration
    """
    # Set seed for reproducibility
    if 'seed' in cfg:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # Print configuration
    print("=" * 80)
    print("Inference Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Build inference instance from config
    inference_name = cfg.inference.name
    print(f"\nBuilding inference: {inference_name}")

    if inference_name in INFERENCE_BUILDERS:
        inference = INFERENCE_BUILDERS[inference_name](cfg)
    else:
        raise ValueError(f"Unknown inference type: {inference_name}")

    print(f"Model loaded from: {cfg.inference.checkpoint_path}")
    print(f"Device: {cfg.inference.device}")
    print(f"Detection threshold: {cfg.inference.threshold}")

    # Example: Run inference on a single image
    if cfg.data.get('test_root'):
        test_root = Path(cfg.data.test_root)
        if test_root.exists():
            # Find first image in test directory
            image_files = list(test_root.glob("*.jpg")) + list(test_root.glob("*.png"))
            if image_files:
                image_path = image_files[0]
                print(f"\nRunning inference on: {image_path}")

                # Load image
                image = Image.open(image_path).convert("RGB")
                image_np = np.array(image)

                # Run inference
                result = inference.predict_image(
                    image_np,
                    threshold=cfg.inference.threshold,
                    return_heatmap=True
                )

                # Print results
                print(f"Detected {len(result['points'])} objects")
                print(f"Detection coordinates (first 5):")
                for i, (point, score) in enumerate(zip(result['points'][:5], result['scores'][:5])):
                    class_info = f", class: {result['classes'][i]}" if 'classes' in result else ""
                    print(f"  {i+1}. Point: ({point[0]:.1f}, {point[1]:.1f}), Score: {score:.3f}{class_info}")

                print(f"\nInference completed successfully!")
            else:
                print(f"\nNo images found in {test_root}")
        else:
            print(f"\nTest directory not found: {test_root}")
    else:
        print("\nNo test directory specified in config")

    print("\n" + "=" * 80)
    print("To use inference programmatically:")
    print("=" * 80)
    print("""
from animaldet.engine import INFERENCE_BUILDERS
from PIL import Image
import numpy as np

# Build inference from config
inference = INFERENCE_BUILDERS['HerdNetInference'](cfg)

# Load and prepare image
image = Image.open('path/to/image.jpg').convert('RGB')
image_np = np.array(image)

# Run inference
result = inference.predict_image(image_np, threshold=0.5)

# Access results
points = result['points']  # (N, 2) array of detection coordinates
scores = result['scores']  # (N,) array of confidence scores
classes = result['classes']  # (N,) array of class predictions (if available)
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
