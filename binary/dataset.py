"""Dataset loader with balanced sampling and augmentations."""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class BalancedBinaryDataset(Dataset):
    """
    Binary classification dataset with balanced sampling (50% with objects, 50% without).
    Uses augmentations suitable for binary classification.
    """

    def __init__(
        self,
        annotations_path: str,
        images_dir: str,
        image_size: int = 224,
        is_train: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.is_train = is_train

        # Load annotations
        with open(annotations_path, "r") as f:
            coco_data = json.load(f)

        # Build image ID to annotations mapping
        self.image_id_to_anns = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_anns:
                self.image_id_to_anns[img_id] = []
            self.image_id_to_anns[img_id].append(ann)

        # Build image metadata
        self.images_with_objects = []
        self.images_without_objects = []

        for img in coco_data["images"]:
            img_id = img["id"]
            img_info = {
                "id": img_id,
                "file_name": img["file_name"],
                "has_object": img_id in self.image_id_to_anns,
            }

            if img_info["has_object"]:
                self.images_with_objects.append(img_info)
            else:
                self.images_without_objects.append(img_info)

        print(
            f"Dataset loaded: {len(self.images_with_objects)} images with objects, "
            f"{len(self.images_without_objects)} images without objects"
        )

        # Setup augmentations
        if is_train:
            self.transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomRotation(degrees=15),
                    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        # Return double the minimum to ensure 50/50 split
        return 2 * min(len(self.images_with_objects), len(self.images_without_objects))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Alternate between with/without objects for balanced sampling
        if idx % 2 == 0:
            # Sample from images WITH objects
            img_info = random.choice(self.images_with_objects)
            label = 1
        else:
            # Sample from images WITHOUT objects
            img_info = random.choice(self.images_without_objects)
            label = 0

        # Load and transform image
        img_path = self.images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label


def get_class_weights(dataset: BalancedBinaryDataset) -> torch.Tensor:
    """
    Calculate class weights for weighted loss.
    Since we balance 50/50, weights should be equal, but we compute from actual data.
    """
    num_with_objects = len(dataset.images_with_objects)
    num_without_objects = len(dataset.images_without_objects)
    total = num_with_objects + num_without_objects

    # Weight inversely proportional to class frequency
    weight_positive = total / (2 * num_with_objects)
    weight_negative = total / (2 * num_without_objects)

    return torch.tensor([weight_negative, weight_positive])
