# HerdNet Training Pipeline

Quick guide for training HerdNet on the Delplanque dataset following the two-stage approach from the paper.

## Overview

**Stage 1**: Train on animal-only patches (100 epochs)  
**HNP Generation**: Mine hard negative patches from full training images  
**Stage 2**: Retrain with animal patches + hard negatives (50 epochs)

## Prerequisites

```bash
# Data structure expected:
data-delplanque/
├── train/              # Full 24MP training images
├── train.csv           # Annotations for full images
├── train_patches/      # Pre-extracted animal patches (512x512)
├── train_patches.csv   # Annotations for patches
├── val/                # Full validation images
├── val.csv             # Validation annotations
├── test/               # Full test images
└── test.csv            # Test annotations
```

## Step 1: Stage 1 Training

Train on animal-only patches for 100 epochs:

```bash
python train_stage1.py \
  --root data-delplanque \
  --work-dir output/stage1 \
  --epochs 100 \
  --batch-size 4 \
  --num-workers 40 \
  --valid-freq 1 \
  --device cuda
```

**Key parameters:**
- `--root`: Directory containing `train_patches/`, `val/`, and CSVs
- `--epochs`: 100 (as per paper)
- `--valid-freq`: Validation frequency in epochs (default: 1)

**Output:**
- `output/stage1/best_model.pth`
- `output/stage1/latest_model.pth`

## Step 2: Generate Hard Negative Patches

Run inference on full 24MP training images to mine HNPs:

```bash
python generate_hnps.py \
  --checkpoint output/stage1/best_model.pth \
  --train-csv data-delplanque/train.csv \
  --train-root data-delplanque/train \
  --output-root output/hnp_patches \
  --patch-size 512 \
  --patch-overlap 160 \
  --min-score 0.0 \
  --batch-size 1 \
  --num-workers 4 \
  --device cuda
```

**Important:**
- `--batch-size 1` is required (full images have varying detection counts)
- `--num-workers 4` or less to avoid memory issues
- Generates ~10,000 HNP patches (number varies)

**Output:**
- `output/hnp_patches/*.JPG` - HNP patch images
- `output/hnp_patches/detections.csv` - All detections (TPs + FPs)
- `output/hnp_patches/gt.csv` - (Discard this, use original train_patches.csv)

## Step 3: Merge HNPs with Original Patches

Create a combined directory for Stage 2:

```bash
# Create combined directory
mkdir -p data-delplanque/train_patches_stage2

# Copy original patches
cp data-delplanque/train_patches/*.JPG data-delplanque/train_patches_stage2/

# Copy HNPs (skip duplicates with same name)
cp -n output/hnp_patches/*.JPG data-delplanque/train_patches_stage2/


## Step 4: Stage 2 Training

Train with combined dataset (original + HNPs):

```bash
python train_stage2.py \
  --checkpoint output/stage1/best_model.pth \
  --train-root data-delplanque/train_patches_stage2 \
  --train-csv data-delplanque/train_patches.csv \
  --val-csv data-delplanque/val.csv \
  --val-root data-delplanque/val \
  --work-dir output/stage2 \
  --epoch-count 50 \
  --batch-size 4 \
  --lr 1e-6 \
  --num-workers 40 \
  --valid-freq 5 
```

**Key parameters:**
- `--train-root`: Combined directory with original + HNPs
- `--train-csv`: **Original** train_patches.csv (HNPs not in this CSV)
- `--epoch-count`: 50 (as per paper)
- `--lr`: 1e-6 (lower than Stage 1, as per paper)

**How it works:**
- `FolderDataset` reads all `.JPG` files in `train_patches_stage2/`
- Patches in CSV → uses their annotations (originals)
- Patches NOT in CSV → treats as background (HNPs)

**Output:**
- `output/stage2/best_model.pth`
- `output/stage2/latest_model.pth`

## Step 5: Evaluation

### Evaluate on Test Set

```bash
python predict_evaluate_full_image.py \
  --checkpoint output/stage2/best_model.pth \
  --csv data-delplanque/test.csv \
  --root data-delplanque/test \
  --output-dir output/stage2/test_eval \
  --device cuda
```

**Important:** Do NOT use `--upsample` flag for evaluation to match training configuration.

**Output:**
- `output/stage2/test_eval/detections_no_upsample.csv`
- `output/stage2/test_eval/metrics_no_upsample.json`


## Visualization with FiftyOne

To visualize predictions, you need to scale coordinates to match image resolution:

```bash
# Scale detections to original resolution (×2)
python3 -c "
import pandas as pd
df = pd.read_csv('output/stage2/test_eval/detections_no_upsample.csv')
df['x'] = df['x'] * 2
df['y'] = df['y'] * 2
df.to_csv('output/stage2/test_eval/detections_scaled.csv', index=False)
"

# Visualize with FiftyOne
python scripts/view_full_fiftyone.py \
  --root data-delplanque/test \
  --gt-csv data-delplanque/test.csv \
  --detections-csv output/stage2/test_eval/detections_scaled.csv \
  --class-map classes.json
```

## Common Issues & Solutions

### 1. `--upsample` Flag Confusion

**Problem:** Using `--upsample` during evaluation gives low F1 score (~0.68) even though validation was good (~0.85).

**Explanation:**
- During **training/validation**: Model uses `up=False` (coordinates in reduced scale ÷2)
- With `--upsample` during **test**: Coordinates in original scale (×1)
- Mismatch in scales causes poor matching with ground truth

**Solution:** 
- ✅ **Evaluate WITHOUT `--upsample`** to match training configuration
- For visualization, scale coordinates manually (multiply by 2)

### 2. Out of Memory During HNP Generation

**Problem:** `torch.OutOfMemoryError` when running `generate_hnps.py`

**Solutions:**
- Use `--batch-size 1` (required)
- Reduce `--num-workers` to 4 or even 0
- Use smaller GPU or switch to CPU if necessary
- The script processes 24MP images, which are memory-intensive

