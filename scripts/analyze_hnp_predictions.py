"""Analyze model predictions on hard negative patches."""

import json
import torch
from pathlib import Path
from collections import defaultdict
import numpy as np
import sys

# Add rf-detr to path
rfdetr_path = Path("/home/lmanrique/Do/rf-detr")
if str(rfdetr_path) not in sys.path:
    sys.path.insert(0, str(rfdetr_path))

from rfdetr.detr import RFDETR


def load_annotations(annotation_path):
    """Load COCO annotations and identify background images."""
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Find images with annotations
    img_ids_with_annots = set()
    for ann in data['annotations']:
        img_ids_with_annots.add(ann['image_id'])

    # Separate background and foreground images
    background_imgs = []
    foreground_imgs = []

    for img in data['images']:
        if img['id'] in img_ids_with_annots:
            foreground_imgs.append(img)
        else:
            background_imgs.append(img)

    return background_imgs, foreground_imgs, data


def analyze_model_on_backgrounds(model_path, data_dir, annotation_file,
                                 confidence_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Analyze how many detections the model makes on background images."""

    # Load annotations
    ann_path = Path(data_dir) / annotation_file
    background_imgs, foreground_imgs, data = load_annotations(ann_path)

    print(f"\n{'='*80}")
    print(f"Dataset Statistics:")
    print(f"  Total images: {len(data['images'])}")
    print(f"  Foreground images: {len(foreground_imgs)}")
    print(f"  Background/HNP images: {len(background_imgs)}")
    print(f"{'='*80}\n")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Determine model variant from checkpoint or config
    # For now, assume base model
    from rfdetr.detr import RFDETRBase
    model = RFDETRBase(num_classes=6)

    # Load state dict - handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'ema' in checkpoint:
        state_dict = checkpoint['ema']['module']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Analyze predictions on background images
    print(f"\nAnalyzing predictions on {len(background_imgs)} background images...")
    print(f"(Using first 100 for faster analysis)\n")

    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((560, 560)),
        T.ToTensor(),
    ])

    # Track detections per confidence threshold
    detections_per_threshold = defaultdict(list)
    images_with_detections = defaultdict(set)

    # Sample for faster analysis
    sample_size = min(100, len(background_imgs))
    sample_imgs = background_imgs[:sample_size]

    with torch.no_grad():
        for idx, img_info in enumerate(sample_imgs):
            if idx % 20 == 0:
                print(f"  Processing {idx}/{sample_size}...")

            img_path = Path(data_dir) / "train2017" / img_info['file_name']
            if not img_path.exists():
                continue

            # Load and transform image
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            # Run inference
            outputs = model(img_tensor)

            # Get predictions (outputs are logits, need sigmoid for probabilities)
            pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
            pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]

            # Convert logits to probabilities
            pred_probs = pred_logits.sigmoid()

            # Get max class probability for each query
            max_probs = pred_probs.max(dim=-1)[0]  # [num_queries]

            # Count detections at each threshold
            for threshold in confidence_thresholds:
                num_dets = (max_probs > threshold).sum().item()
                detections_per_threshold[threshold].append(num_dets)

                if num_dets > 0:
                    images_with_detections[threshold].add(img_info['id'])

    # Print results
    print(f"\n{'='*80}")
    print(f"Results on Background Images (sample size: {sample_size}):")
    print(f"{'='*80}")

    for threshold in sorted(confidence_thresholds):
        dets = detections_per_threshold[threshold]
        total_dets = sum(dets)
        avg_dets = np.mean(dets)
        max_dets = max(dets) if dets else 0
        num_imgs_with_dets = len(images_with_detections[threshold])
        pct_imgs_with_dets = (num_imgs_with_dets / sample_size) * 100

        print(f"\nConfidence threshold: {threshold:.1f}")
        print(f"  Total detections: {total_dets}")
        print(f"  Avg detections per image: {avg_dets:.2f}")
        print(f"  Max detections in single image: {max_dets}")
        print(f"  Images with detections: {num_imgs_with_dets}/{sample_size} ({pct_imgs_with_dets:.1f}%)")
        print(f"  Expected FPs on full {len(background_imgs)} backgrounds: ~{int(avg_dets * len(background_imgs))}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/rfdetr/560', help='Data directory')
    parser.add_argument('--annotation', type=str, default='annotations/instances_train_plus_hnp2017.json',
                       help='Annotation file with HNP')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.6, 0.7],
                       help='Confidence thresholds to test')

    args = parser.parse_args()

    analyze_model_on_backgrounds(
        model_path=args.model,
        data_dir=args.data_dir,
        annotation_file=args.annotation,
        confidence_thresholds=args.thresholds
    )
