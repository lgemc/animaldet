"""Lightweight inference script for binary object detection on COCO format datasets."""

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from model import BinaryObjectDetector


class BinaryInference:
    """Lightweight inference wrapper for binary object detection."""

    def __init__(self, checkpoint_path: str, device: str = "cuda", image_size: int = 224):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size

        # Load model
        self.model = BinaryObjectDetector(pretrained=False).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Setup transforms
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        print(f"Model loaded from {checkpoint_path}")
        print(f"Using device: {self.device}")

    @torch.no_grad()
    def predict(self, image_path: str, threshold: float = 0.5) -> dict:
        """
        Predict if an image contains an object.

        Args:
            image_path: Path to image
            threshold: Probability threshold for classification

        Returns:
            dict with 'has_object' (bool), 'probability' (float), 'prediction' (int)
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        logit = self.model(image_tensor)
        probability = torch.sigmoid(logit).item()
        has_object = probability > threshold

        return {
            "has_object": has_object,
            "probability": probability,
            "prediction": int(has_object),
        }

    @torch.no_grad()
    def predict_batch(self, image_paths: list, threshold: float = 0.5) -> list:
        """Predict on a batch of images."""
        images = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            images.append(self.transform(image))

        batch = torch.stack(images).to(self.device)
        logits = self.model(batch)
        probabilities = torch.sigmoid(logits).cpu().numpy()

        results = []
        for prob in probabilities:
            has_object = prob > threshold
            results.append(
                {
                    "has_object": bool(has_object),
                    "probability": float(prob),
                    "prediction": int(has_object),
                }
            )

        return results


def run_on_coco_dataset(
    checkpoint_path: str,
    annotations_path: str,
    images_dir: str,
    output_path: str = None,
    threshold: float = 0.5,
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    Run inference on a COCO format dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        annotations_path: Path to COCO annotations JSON
        images_dir: Directory containing images
        output_path: Optional path to save predictions
        threshold: Classification threshold
        batch_size: Batch size for inference
        device: Device to use
    """
    # Load annotations
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    # Initialize inference
    inference = BinaryInference(checkpoint_path, device=device)

    # Process images
    results = []
    image_paths = []
    image_ids = []

    for img_info in coco_data["images"]:
        img_path = Path(images_dir) / img_info["file_name"]
        if img_path.exists():
            image_paths.append(str(img_path))
            image_ids.append(img_info["id"])

    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i : i + batch_size]
        batch_ids = image_ids[i : i + batch_size]

        predictions = inference.predict_batch(batch_paths, threshold=threshold)

        for img_id, img_path, pred in zip(batch_ids, batch_paths, predictions):
            results.append(
                {
                    "image_id": img_id,
                    "image_path": img_path,
                    "has_object": pred["has_object"],
                    "probability": pred["probability"],
                }
            )

    # Print summary
    num_with_objects = sum(1 for r in results if r["has_object"])
    num_without_objects = len(results) - num_with_objects

    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"Total images:         {len(results)}")
    print(f"Images with objects:  {num_with_objects} ({num_with_objects/len(results)*100:.1f}%)")
    print(f"Images without objects: {num_without_objects} ({num_without_objects/len(results)*100:.1f}%)")
    print(f"Threshold used:       {threshold}")
    print("=" * 60 + "\n")

    # Save results if requested
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Binary object detection inference on COCO format datasets"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--annotations", required=True, help="Path to COCO annotations JSON"
    )
    parser.add_argument("--images", required=True, help="Directory containing images")
    parser.add_argument(
        "--output", default=None, help="Path to save predictions JSON"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument(
        "--image-size", type=int, default=224, help="Image size (not used in dataset mode)"
    )

    args = parser.parse_args()

    run_on_coco_dataset(
        checkpoint_path=args.checkpoint,
        annotations_path=args.annotations,
        images_dir=args.images,
        output_path=args.output,
        threshold=args.threshold,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
