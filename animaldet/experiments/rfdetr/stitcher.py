"""RF-DETR image stitcher for large image inference.

This module provides stitching capability for RF-DETR to handle large images
by dividing them into patches, running inference, and rescaling predictions
back to the original image coordinates.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from pathlib import Path
import sys

from animaldet.utils import get_device

# Add rf-detr to path
rfdetr_path = Path("/home/lmanrique/Do/rf-detr")
if str(rfdetr_path) not in sys.path:
    sys.path.insert(0, str(rfdetr_path))

# Import HerdNet's ImageToPatches for patching logic
herdnet_path = Path("/home/lmanrique/Do/HerdNet")
if str(herdnet_path) not in sys.path:
    sys.path.insert(0, str(herdnet_path))

from animaloc.data import ImageToPatches


class RFDETRStitcher(ImageToPatches):
    """Stitcher for RF-DETR to handle large images.

    This class divides large images into patches at the model's expected resolution,
    runs inference on each patch, and rescales the bounding box predictions back to
    the original image coordinates. Supports voting-based filtering for overlapping patches.

    Args:
        model: RF-DETR model instance (PyTorch nn.Module)
        size: Patch size tuple (height, width), typically (560, 560)
        overlap: Overlap between patches in pixels (default: 0 for non-overlapping)
        batch_size: Batch size for patch inference (default: 1)
        confidence_threshold: Minimum confidence score for detections (default: 0.5)
        nms_threshold: IoU threshold for non-maximum suppression (default: 0.45)
        device_name: Device for inference ('cuda' or 'cpu')
        voting_threshold: Minimum fraction of overlapping patches that must agree on a
            detection for it to be kept (default: 0.5, i.e., majority vote)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        size: Tuple[int, int] = (560, 560),
        overlap: int = 0,
        batch_size: int = 1,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device_name: str = "cuda",
        voting_threshold: float = 0.5,
    ) -> None:
        assert isinstance(model, torch.nn.Module), \
            "model argument must be an instance of nn.Module()"

        self.model = model
        self.size = size
        self.overlap = overlap
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.voting_threshold = voting_threshold

        # Get device using centralized utility
        self.device = get_device(device_name, verbose=False)

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply stitching algorithm to the image.

        Args:
            image: Input image tensor of shape [C, H, W]

        Returns:
            Dictionary containing:
                - 'boxes': Tensor of shape [N, 4] with boxes in (x1, y1, x2, y2) format
                - 'scores': Tensor of shape [N] with confidence scores
                - 'labels': Tensor of shape [N] with class labels
        """
        # Initialize patching
        super(RFDETRStitcher, self).__init__(image, self.size, self.overlap)

        self.image = image.to(torch.device('cpu'))

        # Step 1: Get patches
        patches = self.make_patches()

        # Step 2: Run inference on patches
        patch_detections = self._inference(patches)

        # Step 3: Rescale detections to original image coordinates
        stitched_detections = self._stitch_detections(patch_detections)

        # Remove patch_ids before returning (only used internally for voting)
        if 'patch_ids' in stitched_detections:
            stitched_detections = {
                'boxes': stitched_detections['boxes'],
                'scores': stitched_detections['scores'],
                'labels': stitched_detections['labels']
            }

        return stitched_detections

    @torch.no_grad()
    def _inference(self, patches: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Run inference on image patches.

        Args:
            patches: Tensor of patches [N, C, H, W]

        Returns:
            List of detection dictionaries, one per patch
        """
        dataset = TensorDataset(patches)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(dataset)
        )

        all_detections = []
        for patch_batch in dataloader:
            patch_batch = patch_batch[0].to(self.device)

            # RF-DETR forward pass returns dict with 'pred_logits' and 'pred_boxes'
            outputs = self.model(patch_batch)

            # Process outputs for each image in batch
            batch_size = patch_batch.shape[0]
            for i in range(batch_size):
                # Extract predictions for this image
                pred_logits = outputs['pred_logits'][i]  # [num_queries, num_classes]
                pred_boxes = outputs['pred_boxes'][i]    # [num_queries, 4]

                # Convert logits to scores (sigmoid for DETR-style models without background class)
                pred_scores = pred_logits.sigmoid()

                # Get max score and class for each query
                scores, labels = pred_scores.max(dim=-1)

                # Convert from 0-indexed to 1-indexed labels to match ground truth format
                labels = labels + 1

                # Filter by confidence threshold
                keep = scores > self.confidence_threshold
                boxes = pred_boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

                # Convert boxes from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
                boxes = self._box_cxcywh_to_xyxy(boxes)
                boxes = boxes * self.size[0]  # Denormalize to patch size

                all_detections.append({
                    'boxes': boxes.cpu(),
                    'scores': scores.cpu(),
                    'labels': labels.cpu()
                })

        return all_detections

    def _box_cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """Convert boxes from center format to corner format.

        Args:
            boxes: Tensor of shape [N, 4] in (cx, cy, w, h) format

        Returns:
            Tensor of shape [N, 4] in (x1, y1, x2, y2) format
        """
        if boxes.shape[0] == 0:
            return boxes

        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def _stitch_detections(
        self,
        patch_detections: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Rescale patch detections to original image coordinates.

        Args:
            patch_detections: List of detection dicts from each patch

        Returns:
            Dictionary with all detections in original image coordinates
        """
        all_boxes = []
        all_scores = []
        all_labels = []
        all_patch_ids = []

        # Get patch limits (locations in original image)
        limits = self.get_limits()

        for patch_idx, (detection, (_, limit)) in enumerate(zip(patch_detections, limits.items())):
            if detection['boxes'].shape[0] == 0:
                continue

            boxes = detection['boxes']

            # Rescale boxes to original image coordinates
            # Boxes are in patch coordinates [0, patch_size], need to offset by patch position
            boxes[:, 0] += limit.x_min  # x1
            boxes[:, 1] += limit.y_min  # y1
            boxes[:, 2] += limit.x_min  # x2
            boxes[:, 3] += limit.y_min  # y2

            all_boxes.append(boxes)
            all_scores.append(detection['scores'])
            all_labels.append(detection['labels'])
            all_patch_ids.append(torch.full((len(boxes),), patch_idx, dtype=torch.long))

        if len(all_boxes) == 0:
            # No detections
            return {
                'boxes': torch.empty((0, 4)),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long),
                'patch_ids': torch.empty(0, dtype=torch.long)
            }

        return {
            'boxes': torch.cat(all_boxes, dim=0),
            'scores': torch.cat(all_scores, dim=0),
            'labels': torch.cat(all_labels, dim=0),
            'patch_ids': torch.cat(all_patch_ids, dim=0)
        }

    def _apply_voting(
        self,
        detections: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Filter detections based on voting from overlapping patches.

        Groups overlapping detections and keeps only those that were detected
        by a sufficient fraction of patches that could have seen them.

        Args:
            detections: Dictionary with 'boxes', 'scores', 'labels', 'patch_ids'

        Returns:
            Filtered detections after voting
        """
        if detections['boxes'].shape[0] == 0:
            return detections

        import torchvision

        boxes = detections['boxes']
        scores = detections['scores']
        labels = detections['labels']
        patch_ids = detections['patch_ids']

        # Calculate IoU between all boxes
        ious = torchvision.ops.box_iou(boxes, boxes)

        # Group boxes with high overlap (IoU > 0.5)
        visited = torch.zeros(len(boxes), dtype=torch.bool)
        keep_indices = []

        for i in range(len(boxes)):
            if visited[i]:
                continue

            # Find all boxes that overlap with this one
            overlapping = (ious[i] > 0.5) & (labels == labels[i])
            group_indices = torch.where(overlapping)[0]
            visited[overlapping] = True

            # Get unique patches that detected this object
            patches_voted = patch_ids[group_indices].unique()
            n_votes = len(patches_voted)

            # Calculate how many patches could see this region
            box_center = torch.tensor([
                (boxes[i, 0] + boxes[i, 2]) / 2,
                (boxes[i, 1] + boxes[i, 3]) / 2
            ])
            n_possible_patches = self._count_patches_covering_point(box_center)

            # Apply voting threshold
            vote_ratio = n_votes / max(n_possible_patches, 1)
            if vote_ratio >= self.voting_threshold:
                # Keep the detection with highest score from the group
                best_idx = group_indices[scores[group_indices].argmax()]
                keep_indices.append(best_idx)

        if len(keep_indices) == 0:
            return {
                'boxes': torch.empty((0, 4)),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long),
                'patch_ids': torch.empty(0, dtype=torch.long)
            }

        keep_indices = torch.tensor(keep_indices, dtype=torch.long)

        return {
            'boxes': detections['boxes'][keep_indices],
            'scores': detections['scores'][keep_indices],
            'labels': detections['labels'][keep_indices],
            'patch_ids': detections['patch_ids'][keep_indices]
        }

    def _count_patches_covering_point(self, point: torch.Tensor) -> int:
        """Count how many patches cover a given point in the image.

        Args:
            point: Tensor of shape [2] containing (x, y) coordinates

        Returns:
            Number of patches that cover this point
        """
        limits = self.get_limits()
        count = 0

        for _, limit in limits.items():
            if (limit.x_min <= point[0] < limit.x_max and
                limit.y_min <= point[1] < limit.y_max):
                count += 1

        return count

    def _apply_nms(
        self,
        detections: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply non-maximum suppression to remove duplicate detections.

        Args:
            detections: Dictionary with 'boxes', 'scores', 'labels', optionally 'patch_ids'

        Returns:
            Filtered detections after NMS (without patch_ids)
        """
        if detections['boxes'].shape[0] == 0:
            return {
                'boxes': torch.empty((0, 4)),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long)
            }

        # Apply NMS per class
        keep_indices = []
        unique_labels = detections['labels'].unique()

        for label in unique_labels:
            mask = detections['labels'] == label
            class_boxes = detections['boxes'][mask]
            class_scores = detections['scores'][mask]

            # Get original indices
            original_indices = torch.where(mask)[0]

            # Apply NMS
            keep = self._batched_nms(
                class_boxes,
                class_scores,
                self.nms_threshold
            )

            keep_indices.append(original_indices[keep])

        if len(keep_indices) == 0:
            return {
                'boxes': torch.empty((0, 4)),
                'scores': torch.empty(0),
                'labels': torch.empty(0, dtype=torch.long)
            }

        keep_indices = torch.cat(keep_indices)

        # Return without patch_ids (only used for voting)
        return {
            'boxes': detections['boxes'][keep_indices],
            'scores': detections['scores'][keep_indices],
            'labels': detections['labels'][keep_indices]
        }

    def _batched_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """Perform non-maximum suppression.

        Args:
            boxes: Tensor of shape [N, 4]
            scores: Tensor of shape [N]
            threshold: IoU threshold

        Returns:
            Indices of boxes to keep
        """
        if boxes.shape[0] == 0:
            return torch.empty(0, dtype=torch.long)

        # Use torchvision's NMS
        import torchvision
        return torchvision.ops.nms(boxes, scores, threshold)
