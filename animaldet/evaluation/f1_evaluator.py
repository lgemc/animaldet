"""F1 Score Evaluator for object detection.

This module provides a wrapper to compute F1 metrics during evaluation.
"""

from typing import Dict, List, Optional, Any
import torch

from animaldet.evaluation.metrics.detection import evaluate_detection_f1


class F1Evaluator:
    """
    Computes F1 score metrics from model predictions and ground truths.

    Args:
        center_threshold: Maximum center distance in pixels for matching
        score_threshold: Minimum confidence score for predictions
        enabled: Whether F1 evaluation is enabled
        verbose: Print detailed debug information
    """

    def __init__(
        self,
        center_threshold: float = 50.0,
        score_threshold: float = 0.5,
        enabled: bool = True,
        verbose: bool = False
    ):
        self.center_threshold = center_threshold
        self.score_threshold = score_threshold
        self.enabled = enabled
        self.verbose = verbose

    def evaluate(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        class_names: Optional[Dict[int, str]] = None
    ) -> Optional[Dict]:
        """
        Evaluate F1 metrics from predictions and ground truths.

        Args:
            predictions: List of predictions with image_id, category_id, bbox, score
            ground_truths: List of ground truths with image_id, category_id, bbox
            class_names: Optional mapping of category_id to class names

        Returns:
            Dictionary with F1 metrics or None if disabled
        """
        if not self.enabled:
            return None

        if not predictions or not ground_truths:
            print("[F1 Evaluator] No predictions or ground truths available")
            return None

        return evaluate_detection_f1(
            predictions=predictions,
            ground_truths=ground_truths,
            center_threshold=self.center_threshold,
            score_threshold=self.score_threshold,
            class_names=class_names,
            verbose=self.verbose
        )

    @staticmethod
    def collect_predictions_and_gts(
        results: List[Dict],
        targets: List[Dict],
        convert_boxes: bool = True
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Collect predictions and ground truths from model outputs.

        Args:
            results: Model prediction results (after postprocessing)
            targets: Ground truth targets
            convert_boxes: Whether to convert box formats (from x1y1x2y2 to xywh)

        Returns:
            Tuple of (predictions, ground_truths)
        """
        all_predictions = []
        all_ground_truths = []

        for target, output in zip(targets, results):
            image_id = target["image_id"].item() if torch.is_tensor(target["image_id"]) else target["image_id"]

            # Add predictions
            if "boxes" in output and len(output["boxes"]) > 0:
                boxes = output["boxes"].float().cpu().numpy() if torch.is_tensor(output["boxes"]) else output["boxes"]

                # Convert from [x1, y1, x2, y2] to [x, y, w, h] if needed
                if convert_boxes:
                    boxes_xywh = boxes.copy()
                    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
                    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
                else:
                    boxes_xywh = boxes

                scores = output["scores"].float().cpu().numpy() if torch.is_tensor(output["scores"]) else output["scores"]
                labels = output["labels"].float().cpu().numpy() if torch.is_tensor(output["labels"]) else output["labels"]

                for box, score, label in zip(boxes_xywh, scores, labels):
                    all_predictions.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box.tolist() if hasattr(box, 'tolist') else box,
                        "score": float(score)
                    })

            # Add ground truths
            if "boxes" in target and len(target["boxes"]) > 0:
                gt_boxes = target["boxes"].float().cpu().numpy() if torch.is_tensor(target["boxes"]) else target["boxes"]

                # Denormalize if needed (check if boxes are in [0,1] range)
                if gt_boxes.max() <= 1.0:
                    orig_h, orig_w = target["orig_size"].cpu().numpy() if torch.is_tensor(target["orig_size"]) else target["orig_size"]
                    gt_boxes_denorm = gt_boxes.copy()
                    gt_boxes_denorm[:, [0, 2]] *= orig_w  # cx and width
                    gt_boxes_denorm[:, [1, 3]] *= orig_h  # cy and height
                else:
                    gt_boxes_denorm = gt_boxes

                # Convert from [cx, cy, w, h] to [x, y, w, h]
                gt_boxes_xywh = gt_boxes_denorm.copy()
                gt_boxes_xywh[:, 0] = gt_boxes_denorm[:, 0] - gt_boxes_denorm[:, 2] / 2  # x = cx - w/2
                gt_boxes_xywh[:, 1] = gt_boxes_denorm[:, 1] - gt_boxes_denorm[:, 3] / 2  # y = cy - h/2

                gt_labels = target["labels"].float().cpu().numpy() if torch.is_tensor(target["labels"]) else target["labels"]

                for box, label in zip(gt_boxes_xywh, gt_labels):
                    all_ground_truths.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box.tolist() if hasattr(box, 'tolist') else box
                    })

        return all_predictions, all_ground_truths
