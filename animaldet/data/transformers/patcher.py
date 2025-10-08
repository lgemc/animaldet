"""
Image patcher for extracting patches from large images with annotations.

This module provides utilities to split large images into smaller patches,
optionally filtering patches that contain annotations.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations import PadIfNeeded
from tqdm import tqdm


@dataclass
class BBox:
    """Bounding box representation."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def get_tuple(self) -> Tuple[float, float, float, float]:
        """Return bbox as tuple (x_min, y_min, x_max, y_max)."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def area(self) -> float:
        """Calculate bbox area."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


class ImagePatcher:
    """
    Extract patches from an image with optional overlap.

    Args:
        image: Input image tensor or numpy array (C, H, W) or (H, W, C)
        patch_size: Size of patches as (height, width)
        overlap: Overlap between patches in pixels
    """

    def __init__(
        self,
        image: torch.Tensor | np.ndarray,
        patch_size: Tuple[int, int],
        overlap: int = 0,
    ):
        self.image = image
        self.patch_height, self.patch_width = patch_size
        self.overlap = overlap

        # Handle different input formats
        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] in [1, 3]:  # (C, H, W)
                self.height, self.width = image.shape[1], image.shape[2]
            else:
                self.height, self.width = image.shape[0], image.shape[1]
        else:
            self.height, self.width = image.shape[:2]

    def make_patches(self) -> List[torch.Tensor]:
        """
        Generate patches from the image.

        Returns:
            List of patch tensors
        """
        patches = []
        stride_h = self.patch_height - self.overlap
        stride_w = self.patch_width - self.overlap

        for y in range(0, self.height, stride_h):
            for x in range(0, self.width, stride_w):
                # Calculate patch boundaries
                y_end = min(y + self.patch_height, self.height)
                x_end = min(x + self.patch_width, self.width)

                # Extract patch
                if isinstance(self.image, torch.Tensor):
                    if self.image.ndim == 3 and self.image.shape[0] in [1, 3]:
                        patch = self.image[:, y:y_end, x:x_end]
                    else:
                        patch = self.image[y:y_end, x:x_end]
                else:
                    patch = self.image[y:y_end, x:x_end]
                    patch = torch.from_numpy(patch)

                patches.append(patch)

        return patches

    def get_patch_coords(self) -> List[Tuple[int, int, int, int]]:
        """
        Get coordinates for all patches.

        Returns:
            List of (x_min, y_min, x_max, y_max) tuples
        """
        coords = []
        stride_h = self.patch_height - self.overlap
        stride_w = self.patch_width - self.overlap

        for y in range(0, self.height, stride_h):
            for x in range(0, self.width, stride_w):
                y_end = min(y + self.patch_height, self.height)
                x_end = min(x + self.patch_width, self.width)
                coords.append((x, y, x_end, y_end))

        return coords


class PatchesBuffer:
    """
    Buffer for managing patches with annotations.

    This class processes annotations and determines which patches contain
    annotations based on minimum visibility threshold.

    Args:
        csv_path: Path to CSV file with annotations
        images_root: Root directory containing images
        patch_size: Size of patches as (height, width)
        overlap: Overlap between patches in pixels
        min_visibility: Minimum fraction of annotation area visible in patch
        min_area_ratio: Minimum ratio of cropped area to original bbox area (0.0-1.0)
                       Filters out annotations that are cropped too small
    """

    def __init__(
        self,
        csv_path: str,
        images_root: str,
        patch_size: Tuple[int, int],
        overlap: int = 0,
        min_visibility: float = 0.1,
        min_area_ratio: float = 0.0,
    ):
        self.csv_path = csv_path
        self.images_root = images_root
        self.patch_height, self.patch_width = patch_size
        self.overlap = overlap
        self.min_visibility = min_visibility
        self.min_area_ratio = min_area_ratio

        self._buffer = None

    @property
    def buffer(self) -> pd.DataFrame:
        """
        Get or create the patches buffer.

        Returns:
            DataFrame with columns: images, base_images, x, y, labels, limits
        """
        if self._buffer is None:
            self._buffer = self._create_buffer()
        return self._buffer

    def _create_buffer(self) -> pd.DataFrame:
        """Create the patches buffer from annotations."""
        # Read annotations
        df = pd.read_csv(self.csv_path)

        # Detect annotation format
        has_bbox_format = all(col in df.columns for col in ['x_min', 'y_min', 'x_max', 'y_max'])
        has_point_format = 'x' in df.columns and 'y' in df.columns

        if not has_bbox_format and not has_point_format:
            raise ValueError(
                "CSV must contain either:\n"
                "  - Point format: 'images', 'x', 'y' columns, or\n"
                "  - Bbox format: 'images', 'x_min', 'y_min', 'x_max', 'y_max' columns"
            )

        if 'images' not in df.columns:
            raise ValueError("CSV must contain 'images' column")

        has_labels = 'labels' in df.columns

        patches_data = []

        # Process each image
        for img_name in df['images'].unique():
            img_path = os.path.join(self.images_root, img_name)

            if not os.path.exists(img_path):
                continue

            # Get image dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size

            # Get annotations for this image
            img_df = df[df['images'] == img_name]

            # Generate patch coordinates
            stride_h = self.patch_height - self.overlap
            stride_w = self.patch_width - self.overlap

            patch_idx = 0
            for y in range(0, img_height, stride_h):
                for x in range(0, img_width, stride_w):
                    y_end = min(y + self.patch_height, img_height)
                    x_end = min(x + self.patch_width, img_width)

                    patch_bbox = BBox(x, y, x_end, y_end)

                    # Check which annotations fall in this patch
                    for _, row in img_df.iterrows():
                        if has_bbox_format:
                            # Bounding box format - convert to center point and check intersection
                            ann_x_min, ann_y_min = row['x_min'], row['y_min']
                            ann_x_max, ann_y_max = row['x_max'], row['y_max']
                            ann_x = (ann_x_min + ann_x_max) / 2
                            ann_y = (ann_y_min + ann_y_max) / 2

                            # Calculate intersection area for visibility check
                            ann_bbox = BBox(ann_x_min, ann_y_min, ann_x_max, ann_y_max)
                            intersect_x_min = max(ann_x_min, x)
                            intersect_y_min = max(ann_y_min, y)
                            intersect_x_max = min(ann_x_max, x_end)
                            intersect_y_max = min(ann_y_max, y_end)

                            # Check if there's any intersection
                            if intersect_x_min >= intersect_x_max or intersect_y_min >= intersect_y_max:
                                continue

                            # Calculate visibility ratio
                            intersect_area = (intersect_x_max - intersect_x_min) * (intersect_y_max - intersect_y_min)
                            ann_area = ann_bbox.area()
                            visibility = intersect_area / ann_area if ann_area > 0 else 0

                            if visibility < self.min_visibility:
                                continue

                            # Calculate relative bbox coordinates within patch
                            rel_x_min = max(0, ann_x_min - x)
                            rel_y_min = max(0, ann_y_min - y)
                            rel_x_max = min(self.patch_width, ann_x_max - x)
                            rel_y_max = min(self.patch_height, ann_y_max - y)

                            # Filter by minimum area ratio (cropped area vs original area)
                            cropped_area = (rel_x_max - rel_x_min) * (rel_y_max - rel_y_min)
                            area_ratio = cropped_area / ann_area if ann_area > 0 else 0
                            if area_ratio < self.min_area_ratio:
                                continue
                        else:
                            # Point format
                            ann_x, ann_y = row['x'], row['y']

                            # Check if annotation is within patch
                            if not (x <= ann_x < x_end and y <= ann_y < y_end):
                                continue

                        label = row['labels'] if has_labels else None

                        # Calculate relative coordinates within patch
                        rel_x = ann_x - x
                        rel_y = ann_y - y

                        patch_name = f"{Path(img_name).stem}_patch_{patch_idx:04d}{Path(img_name).suffix}"

                        patch_data = {
                            'images': patch_name,
                            'base_images': img_name,
                            'x': rel_x,
                            'y': rel_y,
                            'limits': patch_bbox,
                        }

                        if has_bbox_format:
                            # Also store bbox coordinates for bbox format
                            patch_data['x_min'] = rel_x_min
                            patch_data['y_min'] = rel_y_min
                            patch_data['x_max'] = rel_x_max
                            patch_data['y_max'] = rel_y_max

                        if has_labels:
                            patch_data['labels'] = label

                        patches_data.append(patch_data)

                    patch_idx += 1

        return pd.DataFrame(patches_data)


def save_batch_images(
    patches: List[torch.Tensor],
    base_name: str,
    dest_dir: str,
    patch_size: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Save a batch of patches to disk.

    Args:
        patches: List of patch tensors
        base_name: Base name for the image
        dest_dir: Destination directory
        patch_size: Expected patch size (height, width). If provided, pads patches to this size.
    """
    os.makedirs(dest_dir, exist_ok=True)

    base_stem = Path(base_name).stem
    base_suffix = Path(base_name).suffix

    # Create padder if patch_size is specified
    padder = None
    if patch_size is not None:
        padder = PadIfNeeded(
            min_height=patch_size[0],
            min_width=patch_size[1],
            position="top_left",
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        )

    for idx, patch in enumerate(patches):
        patch_name = f"{base_stem}_patch_{idx:04d}{base_suffix}"
        patch_path = os.path.join(dest_dir, patch_name)

        # Convert tensor to numpy array
        if isinstance(patch, torch.Tensor):
            if patch.ndim == 3 and patch.shape[0] in [1, 3]:  # (C, H, W)
                patch = patch.permute(1, 2, 0)
            patch = patch.numpy()

        if patch.dtype != np.uint8:
            patch = (patch * 255).astype(np.uint8)

        # Pad if necessary
        if padder is not None:
            patch = padder(image=patch)['image']

        img = Image.fromarray(patch)
        img.save(patch_path)


def extract_patches(
    images_root: str,
    dest_dir: str,
    patch_size: Tuple[int, int],
    overlap: int = 0,
    csv_path: Optional[str] = None,
    min_visibility: float = 0.1,
    min_area_ratio: float = 0.0,
    save_all: bool = False,
) -> None:
    """
    Extract patches from images in a directory.

    Args:
        images_root: Root directory containing images
        dest_dir: Destination directory for patches
        patch_size: Size of patches as (height, width)
        overlap: Overlap between patches in pixels
        csv_path: Optional path to CSV with annotations
        min_visibility: Minimum visibility threshold for annotations
        min_area_ratio: Minimum ratio of cropped area to original bbox area
        save_all: If True, save all patches; if False, only annotated ones
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Get list of images
    image_paths = [
        os.path.join(images_root, p)
        for p in os.listdir(images_root)
        if not p.endswith('.csv') and Path(p).suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    ]

    patches_buffer = None
    if csv_path is not None:
        # Create patches buffer with annotations
        patches_buffer = PatchesBuffer(
            csv_path, images_root, patch_size, overlap, min_visibility, min_area_ratio
        ).buffer

        # Save updated annotations
        patches_buffer.drop(columns='limits').to_csv(
            os.path.join(dest_dir, 'gt.csv'), index=False
        )

        if not save_all:
            # Only process images with annotations
            df = pd.read_csv(csv_path)
            image_paths = [
                os.path.join(images_root, x)
                for x in df['images'].unique()
            ]

    # Process each image
    for img_path in tqdm(image_paths, desc='Extracting patches'):
        img_name = os.path.basename(img_path)
        pil_img = Image.open(img_path)

        # Convert to tensor
        img_array = np.array(pil_img)

        if csv_path is not None and not save_all:
            # Save only annotated patches
            padder = PadIfNeeded(
                patch_size[0], patch_size[1],
                position='top_left',
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            )

            img_patch_df = patches_buffer[patches_buffer['base_images'] == img_name]
            for _, row in img_patch_df[['images', 'limits']].iterrows():
                patch_name, limits = row['images'], row['limits']
                cropped_img = np.array(pil_img.crop(limits.get_tuple))
                padded_img = Image.fromarray(padder(image=cropped_img)['image'])
                padded_img.save(os.path.join(dest_dir, patch_name))
        else:
            # Save all patches
            patcher = ImagePatcher(img_array, patch_size, overlap)
            patches = patcher.make_patches()
            save_batch_images(patches, img_name, dest_dir, patch_size=patch_size)