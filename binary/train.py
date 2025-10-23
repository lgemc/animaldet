"""Training script for binary object detection."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BalancedBinaryDataset, get_class_weights
from model import BinaryObjectDetector


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        correct += (predicted == labels.long()).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for images, labels in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device).float()

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        labels_long = labels.long()

        correct += (predicted == labels_long).sum().item()
        total += labels.size(0)

        true_positives += ((predicted == 1) & (labels_long == 1)).sum().item()
        false_positives += ((predicted == 1) & (labels_long == 0)).sum().item()
        false_negatives += ((predicted == 0) & (labels_long == 1)).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return epoch_loss, epoch_acc, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Train binary object detector")
    parser.add_argument(
        "--train-ann", required=True, help="Path to train annotations (COCO format)"
    )
    parser.add_argument(
        "--val-ann", required=True, help="Path to validation annotations (COCO format)"
    )
    parser.add_argument("--train-images", required=True, help="Path to train images directory")
    parser.add_argument("--val-images", required=True, help="Path to validation images directory")
    parser.add_argument("--output-dir", default="binary/checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")

    # Create datasets
    train_dataset = BalancedBinaryDataset(
        args.train_ann, args.train_images, image_size=args.image_size, is_train=True
    )
    val_dataset = BalancedBinaryDataset(
        args.val_ann, args.val_images, image_size=args.image_size, is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Calculate class weights for weighted loss
    class_weights = get_class_weights(train_dataset).to(device)
    print(f"Class weights (no-object, object): {class_weights}")

    # Create model
    model = BinaryObjectDetector(pretrained=True).to(device)

    # Weighted BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_f1 = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, precision, recall, f1 = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        # Save best model based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "val_acc": val_acc,
            }
            torch.save(checkpoint, output_dir / "best_model.pth")
            print(f"✓ Saved best model (F1: {f1:.4f})")

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            output_dir / "last_model.pth",
        )

    print(f"\nTraining completed! Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
