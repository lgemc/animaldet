# Binary Object Detection

Minimal binary classification system for object/no-object detection with ResNet18 backbone.

## Features

- **Balanced DataLoader**: 50% with objects, 50% without objects
- **Weighted Loss**: Class-imbalanced BCE loss for fair training
- **Data Augmentation**: Rotation, flips, color jitter, affine transforms
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, AP
- **Best Metric**: F1 Score (balanced between precision and recall)

## Structure

```
binary/
├── model.py        # ResNet18-based binary classifier
├── dataset.py      # Balanced dataset with augmentations
├── train.py        # Training script with weighted loss
├── eval.py         # Evaluation with comprehensive metrics
└── inference.py    # Lightweight inference for COCO datasets
```

## Usage

### Training

```bash
uv run python binary/train.py \
    --train-ann data/rfdetr/herdnet/560_all/annotations/instances_train2017.json \
    --val-ann data/rfdetr/herdnet/560_all/annotations/instances_val2017.json \
    --train-images data/rfdetr/herdnet/560_all/train \
    --val-images data/rfdetr/herdnet/560_all/val \
    --output-dir binary/checkpoints \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4
```

**Output per epoch:**
- Train Loss & Accuracy
- Validation Loss & Accuracy
- Precision, Recall, F1 Score
- Best model saved based on F1

### Evaluation

```bash
uv run python binary/eval.py \
    --checkpoint binary/checkpoints/best_model.pth \
    --test-ann data/rfdetr/herdnet/560_all/annotations/image_info_test-dev2017.json \
    --test-images data/rfdetr/herdnet/560_all/test \
    --output binary/results/metrics.json
```

**Metrics reported:**
- Accuracy & Balanced Accuracy
- Precision, Recall, F1 (primary metric)
- ROC-AUC, Average Precision
- Specificity
- Confusion Matrix
- True/False Positives/Negatives

### Inference

```bash
uv run python binary/inference.py \
    --checkpoint binary/checkpoints/best_model.pth \
    --annotations data/rfdetr/herdnet/560_all/annotations/image_info_test-dev2017.json \
    --images data/rfdetr/herdnet/560_all/test \
    --output binary/results/predictions.json \
    --threshold 0.5
```

## Model Architecture

**ResNet18 Backbone:**
- Pretrained on ImageNet
- Feature extraction: 512-dimensional
- Binary classification head: 512 → 256 → 1
- Dropout regularization (0.5, 0.3)

## Data Augmentation

**Training transforms:**
- Random horizontal/vertical flips
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine (translation, scale)
- ImageNet normalization

**Validation/Test transforms:**
- Resize only
- ImageNet normalization

## Class Imbalance Handling

1. **Balanced Sampling**: DataLoader ensures 50/50 split
2. **Weighted Loss**: `BCEWithLogitsLoss` with `pos_weight` based on class distribution
3. **Evaluation Metrics**: F1, ROC-AUC, and Average Precision handle imbalance better than accuracy

## Best Practices

- **Primary Metric**: F1 Score (harmonic mean of precision and recall)
- **Secondary Metrics**: ROC-AUC and Average Precision for imbalanced data
- **Threshold Tuning**: Adjust `--threshold` in inference based on precision/recall requirements
- **Early Stopping**: Monitor F1 score on validation set

## Requirements

Installed via `pyproject.toml`:
- torch >= 2.8.0
- torchvision
- pillow
- numpy
- scikit-learn
- tqdm
