"""HerdNet Inference class.

This module provides inference capabilities for HerdNet models, handling
both single images and batched inputs.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from animaldet.engine.inference import BaseInference
from animaloc.models import LossWrapper


class HerdNetInference(BaseInference):
    """
    Inference class for HerdNet models.

    This class handles inference for HerdNet models, including:
    - Heatmap prediction
    - Point extraction from heatmaps
    - Batch processing

    Args:
        model: HerdNet model (LossWrapper or nn.Module)
        device: Device to run inference on
        checkpoint_path: Optional path to checkpoint
        down_ratio: Downsampling ratio used by the model
        threshold: Confidence threshold for point detection
    """

    def __init__(
        self,
        model: Union[LossWrapper, torch.nn.Module],
        device: str = "cuda",
        checkpoint_path: Optional[Union[str, Path]] = None,
        down_ratio: int = 2,
        threshold: float = 0.5,
    ):
        super().__init__(model, device, checkpoint_path)
        self.down_ratio = down_ratio
        self.threshold = threshold

    def _predict_impl(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor], Any],
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run HerdNet inference.

        Args:
            inputs: Input images (B, C, H, W) or dict with 'image' key
            **kwargs: Additional arguments (threshold, etc.)

        Returns:
            Tuple of (heatmap, class_predictions)
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            images = inputs['image']
        elif isinstance(inputs, torch.Tensor):
            images = inputs
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")

        # Ensure batch dimension
        if images.ndim == 3:
            images = images.unsqueeze(0)

        # Get model predictions
        # HerdNet returns (heatmap, class_logits)
        if isinstance(self.model, LossWrapper):
            outputs = self.model.model(images)
        else:
            outputs = self.model(images)

        # Unpack outputs
        if isinstance(outputs, (list, tuple)):
            heatmap = outputs[0]
            class_preds = outputs[1] if len(outputs) > 1 else None
        else:
            heatmap = outputs
            class_preds = None

        return heatmap, class_preds

    def postprocess(
        self,
        outputs: Tuple[torch.Tensor, Optional[torch.Tensor]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Postprocess HerdNet outputs to extract detection points.

        Args:
            outputs: Tuple of (heatmap, class_predictions)
            **kwargs: Additional arguments
                - threshold: Detection threshold (default: self.threshold)
                - return_heatmap: Whether to include raw heatmap (default: False)

        Returns:
            List of detection dictionaries, one per image in batch:
                - points: (N, 2) array of detection coordinates
                - scores: (N,) array of confidence scores
                - classes: (N,) array of class predictions (if available)
                - heatmap: Raw heatmap (if return_heatmap=True)
        """
        threshold = kwargs.get('threshold', self.threshold)
        return_heatmap = kwargs.get('return_heatmap', False)

        heatmap, class_preds = outputs

        # Move to CPU for processing
        heatmap = heatmap.detach().cpu()
        if class_preds is not None:
            class_preds = class_preds.detach().cpu()

        batch_size = heatmap.shape[0]
        results = []

        for i in range(batch_size):
            # Get heatmap for this image
            hmap = heatmap[i, 0]  # (H, W)

            # Find peaks in heatmap
            points, scores = self._extract_points(hmap, threshold)

            # Get class predictions if available
            classes = None
            if class_preds is not None and len(points) > 0:
                # Get class predictions at detected points
                # Scale points to class prediction resolution
                class_map = class_preds[i]  # (num_classes, H, W)
                scaled_points = points / self.down_ratio
                scaled_points = scaled_points.long()

                # Clamp to valid range
                h, w = class_map.shape[1:]
                scaled_points[:, 0] = torch.clamp(scaled_points[:, 0], 0, h - 1)
                scaled_points[:, 1] = torch.clamp(scaled_points[:, 1], 0, w - 1)

                # Extract class predictions
                classes = class_map[:, scaled_points[:, 0], scaled_points[:, 1]]
                classes = torch.argmax(classes, dim=0).numpy()

            result = {
                'points': points.numpy() if len(points) > 0 else np.empty((0, 2)),
                'scores': scores.numpy() if len(scores) > 0 else np.empty(0),
            }

            if classes is not None:
                result['classes'] = classes

            if return_heatmap:
                result['heatmap'] = hmap.numpy()

            results.append(result)

        return results

    def _extract_points(
        self,
        heatmap: torch.Tensor,
        threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract point coordinates from heatmap.

        Args:
            heatmap: 2D heatmap tensor (H, W)
            threshold: Detection threshold

        Returns:
            Tuple of (points, scores)
                - points: (N, 2) tensor of (y, x) coordinates
                - scores: (N,) tensor of confidence scores
        """
        # Apply threshold
        mask = heatmap > threshold

        # Find peak coordinates
        coords = torch.nonzero(mask, as_tuple=False)  # (N, 2) in (y, x) format

        if len(coords) == 0:
            return torch.empty((0, 2)), torch.empty(0)

        # Get scores at detected points
        scores = heatmap[mask]

        # Scale coordinates by down_ratio to get original image coordinates
        points = coords.float() * self.down_ratio

        return points, scores

    def predict_image(
        self,
        image: Union[torch.Tensor, np.ndarray],
        threshold: Optional[float] = None,
        return_heatmap: bool = False
    ) -> Dict[str, Any]:
        """
        Convenience method for single image inference.

        Args:
            image: Input image (C, H, W) or (H, W, C)
            threshold: Detection threshold (default: self.threshold)
            return_heatmap: Whether to include raw heatmap

        Returns:
            Dictionary with detection results:
                - points: (N, 2) array of detection coordinates
                - scores: (N,) array of confidence scores
                - classes: (N,) array of class predictions (if available)
                - heatmap: Raw heatmap (if return_heatmap=True)
        """
        # Convert numpy to torch if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()

        # Handle channel ordering
        if image.ndim == 3 and image.shape[-1] == 3:
            # (H, W, C) -> (C, H, W)
            image = image.permute(2, 0, 1)

        # Normalize if needed (assuming 0-255 range)
        if image.max() > 1.0:
            image = image / 255.0

        # Run inference
        threshold = threshold or self.threshold
        outputs = self.predict(image.unsqueeze(0))
        results = self.postprocess(
            outputs,
            threshold=threshold,
            return_heatmap=return_heatmap
        )

        return results[0]
