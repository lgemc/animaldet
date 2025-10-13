"""RF-DETR evaluation with F1 metrics support."""

import torch
from typing import Any, Dict, Optional

from animaldet.evaluation.f1_evaluator import F1Evaluator
from animaldet.utils import get_autocast_kwargs


def evaluate_with_metrics(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessors: Dict[str, Any],
    data_loader: Any,
    base_ds: Any,
    device: torch.device,
    args: Any,
    collect_for_metrics: bool = True
) -> tuple[Dict[str, Any], Any, list, list, Optional[Dict]]:
    """
    Evaluate model and collect data for additional metrics.

    Args:
        model: Model to evaluate
        criterion: Loss criterion
        postprocessors: Output postprocessors
        data_loader: Validation data loader
        base_ds: Base dataset (for class names)
        device: Device to run on
        args: Training arguments
        collect_for_metrics: Whether to collect predictions/GTs for custom metrics

    Returns:
        Tuple of (stats, coco_evaluator, predictions, ground_truths, class_names)
    """

    model.eval()
    fp16_eval = getattr(args, 'fp16_eval', False)
    if fp16_eval:
        model.half()

    # Collect predictions and ground truths for F1
    all_predictions = []
    all_ground_truths = []

    # We need to run our own evaluation loop to collect data
    from rfdetr.datasets.coco_eval import CocoEvaluator
    import rfdetr.util.misc as utils

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())

    # Ensure COCO dataset has 'info' field for API compatibility
    if base_ds and hasattr(base_ds, 'dataset') and 'info' not in base_ds.dataset:
        base_ds.dataset['info'] = {
            'description': 'Custom Dataset',
            'version': '1.0',
            'year': 2025,
            'contributor': 'animaldet',
            'date_created': '2025/01/01'
        }

    coco_evaluator = CocoEvaluator(base_ds, iou_types) if base_ds else None

    from torch.amp import autocast, GradScaler

    # Get autocast kwargs for the device
    autocast_kwargs = get_autocast_kwargs(device, enabled=args.amp)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if fp16_eval:
            samples.tensors = samples.tensors.half()

        with autocast(**autocast_kwargs):
            outputs = model(samples)

        if fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][sub_key].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # Collect for custom metrics
        if collect_for_metrics:
            preds, gts = F1Evaluator.collect_predictions_and_gts(results, targets)
            all_predictions.extend(preds)
            all_ground_truths.extend(gts)

    # Gather stats
    metric_logger.synchronize_between_processes()
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None and "bbox" in postprocessors.keys():
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

    # Get class names from dataset
    class_names = None
    if base_ds and hasattr(base_ds, 'dataset') and 'categories' in base_ds.dataset:
        class_names = {cat['id']: cat['name'] for cat in base_ds.dataset['categories']}

    return stats, coco_evaluator, all_predictions, all_ground_truths, class_names
