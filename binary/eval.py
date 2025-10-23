"""Evaluation script for binary object detector with comprehensive metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BalancedBinaryDataset
from model import BinaryObjectDetector


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Comprehensive evaluation with multiple metrics.

    Returns metrics including:
    - Accuracy
    - Precision, Recall, F1
    - ROC-AUC (area under ROC curve)
    - Average Precision (AP / PR-AUC)
    - Confusion Matrix
    """
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.cpu().numpy()

        outputs = model(images)
        probabilities = torch.sigmoid(outputs).cpu().numpy()
        predictions = (probabilities > 0.5).astype(int)

        all_labels.extend(labels)
        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "precision": precision_score(all_labels, all_predictions, zero_division=0),
        "recall": recall_score(all_labels, all_predictions, zero_division=0),
        "f1": f1_score(all_labels, all_predictions, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probabilities),
        "average_precision": average_precision_score(all_labels, all_probabilities),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions).tolist(),
    }

    # Calculate per-class metrics
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    # Specificity (True Negative Rate)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Balanced Accuracy (average of recall and specificity)
    metrics["balanced_accuracy"] = (metrics["recall"] + metrics["specificity"]) / 2

    return metrics


def print_metrics(metrics: dict):
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\n--- Primary Metrics ---")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1 Score:           {metrics['f1']:.4f}")

    print("\n--- Advanced Metrics ---")
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"Average Precision:  {metrics['average_precision']:.4f}")
    print(f"Specificity:        {metrics['specificity']:.4f}")

    print("\n--- Confusion Matrix ---")
    cm = metrics["confusion_matrix"]
    print(f"                  Predicted")
    print(f"                  No Obj  |  Object")
    print(f"Actual  No Obj    {cm[0][0]:6d}  |  {cm[0][1]:6d}")
    print(f"        Object    {cm[1][0]:6d}  |  {cm[1][1]:6d}")

    print("\n--- Detailed Counts ---")
    print(f"True Negatives:     {metrics['true_negatives']}")
    print(f"False Positives:    {metrics['false_positives']}")
    print(f"False Negatives:    {metrics['false_negatives']}")
    print(f"True Positives:     {metrics['true_positives']}")

    print("\n" + "=" * 70)
    print("\nBest metric for binary classification: F1 Score")
    print(f"For imbalanced datasets, also consider: ROC-AUC and Average Precision")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate binary object detector")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-ann", required=True, help="Path to test annotations (COCO format)"
    )
    parser.add_argument(
        "--test-images", required=True, help="Path to test images directory"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Image size"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to use"
    )
    parser.add_argument(
        "--output", default=None, help="Save metrics to JSON file"
    )
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = BinaryObjectDetector(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from: {args.checkpoint}")

    if "f1" in checkpoint:
        print(f"Checkpoint F1 score: {checkpoint['f1']:.4f}")

    # Create dataset
    test_dataset = BalancedBinaryDataset(
        args.test_ann,
        args.test_images,
        image_size=args.image_size,
        is_train=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Evaluate
    metrics = evaluate(model, test_loader, device)

    # Print results
    print_metrics(metrics)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        json_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.integer, np.floating)):
                json_metrics[k] = float(v)
            else:
                json_metrics[k] = v

        with open(output_path, "w") as f:
            json.dump(json_metrics, f, indent=2)

        print(f"Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
