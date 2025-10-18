"""Patcher command implementation."""

from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig

from animaldet.data.transformers import extract_patches


def patcher_main(
    config: Optional[str],
    images_root: Optional[str],
    dest_dir: Optional[str],
    patch_size: Optional[int],
    overlap: int,
    csv_path: Optional[str],
    save_all: bool,
    min_bbox_size_ratio: float = 0.0,
):
    """Main patcher function.

    Args:
        config: Path to configuration file
        images_root: Root directory containing images
        dest_dir: Destination directory for patches
        patch_size: Size of patches to extract
        overlap: Overlap between patches in pixels
        csv_path: Path to CSV file with annotations
        save_all: Save all patches including those without annotations
        min_bbox_size_ratio: Minimum ratio of bbox area to original bbox area (0.0-1.0)
    """
    # Load config if provided
    if config:
        cfg = OmegaConf.load(config)

        # Override config with CLI arguments
        if images_root is not None:
            cfg.images_root = images_root
        if dest_dir is not None:
            cfg.dest_dir = dest_dir
        if patch_size is not None:
            cfg.patch_size = patch_size
        if csv_path is not None:
            cfg.csv_path = csv_path

        # Extract configuration
        images_root = cfg.images_root
        dest_dir = cfg.dest_dir
        patch_size_tuple = (cfg.patch_size, cfg.patch_size)
        overlap = cfg.get("overlap", overlap)
        csv_path = cfg.get("csv_path", None)
        save_all = cfg.get("save_all", save_all)
        min_bbox_size_ratio = cfg.get("min_bbox_size_ratio", min_bbox_size_ratio)
        column_mapping = cfg.get("column_mapping", None)
    else:
        # Use CLI arguments only
        if not images_root or not dest_dir or not patch_size:
            raise ValueError(
                "--images-root, --dest-dir, and --patch-size are required when --config is not provided"
            )
        patch_size_tuple = (patch_size, patch_size)
        column_mapping = None

    print("=" * 80)
    print("Patcher Configuration:")
    print("=" * 80)
    print(f"images_root: {images_root}")
    print(f"dest_dir: {dest_dir}")
    print(f"patch_size: {patch_size_tuple}")
    print(f"overlap: {overlap}")
    print(f"csv_path: {csv_path}")
    print(f"save_all: {save_all}")
    print(f"min_bbox_size_ratio: {min_bbox_size_ratio}")
    print("=" * 80)

    # Validate paths
    if not Path(images_root).exists():
        raise ValueError(f"Images root directory does not exist: {images_root}")

    if csv_path is not None and not Path(csv_path).exists():
        raise ValueError(f"CSV file does not exist: {csv_path}")

    # Extract patches
    print(f"\nExtracting patches from: {images_root}")
    print(f"Saving to: {dest_dir}")
    print(f"Patch size: {patch_size_tuple[0]}x{patch_size_tuple[1]}")
    print(f"Overlap: {overlap}px\n")

    extract_patches(
        images_root=images_root,
        dest_dir=dest_dir,
        patch_size=patch_size_tuple,
        overlap=overlap,
        csv_path=csv_path,
        save_all=save_all,
        column_mapping=column_mapping,
        min_bbox_size_ratio=min_bbox_size_ratio,
    )

    print(f"\n✓ Patches extracted successfully to: {dest_dir}")