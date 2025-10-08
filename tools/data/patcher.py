#!/usr/bin/env python3
"""
CLI tool for extracting patches from images using OmegaConf configuration.

Usage:
    python tools/patcher.py --config configs/patcher/default.yaml
    python tools/patcher.py images_root=/path/to/images patch_size=512 overlap=64
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from omegaconf import DictConfig, OmegaConf
import hydra

from animaldet.data.transformers import extract_patches


@hydra.main(version_base=None, config_path="../../configs/patcher", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main patcher function using Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    print("=" * 80)
    print("Patcher Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Extract configuration
    images_root = cfg.images_root
    dest_dir = cfg.dest_dir
    patch_size = (cfg.patch_size, cfg.patch_size)
    overlap = cfg.get("overlap", 0)
    csv_path = cfg.get("csv_path", None)
    min_visibility = cfg.get("min_visibility", 0.1)
    min_area_ratio = cfg.get("min_area_ratio", 0.0)
    save_all = cfg.get("save_all", False)

    # Validate paths
    if not Path(images_root).exists():
        raise ValueError(f"Images root directory does not exist: {images_root}")

    if csv_path is not None and not Path(csv_path).exists():
        raise ValueError(f"CSV file does not exist: {csv_path}")

    # Extract patches
    print(f"\nExtracting patches from: {images_root}")
    print(f"Saving to: {dest_dir}")
    print(f"Patch size: {patch_size[0]}x{patch_size[1]}")
    print(f"Overlap: {overlap}px\n")

    extract_patches(
        images_root=images_root,
        dest_dir=dest_dir,
        patch_size=patch_size,
        overlap=overlap,
        csv_path=csv_path,
        min_visibility=min_visibility,
        min_area_ratio=min_area_ratio,
        save_all=save_all,
    )

    print(f"\n✓ Patches extracted successfully to: {dest_dir}")


if __name__ == "__main__":
    main()
