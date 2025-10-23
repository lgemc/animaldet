"""Binary classification model for object/no-object detection."""

import torch
import torch.nn as nn
import torchvision.models as models


class BinaryObjectDetector(nn.Module):
    """Lightweight binary classifier using ResNet18 backbone."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Use ResNet18 as backbone
        resnet = models.resnet18(pretrained=pretrained)

        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)  # Shape: (batch_size,)
