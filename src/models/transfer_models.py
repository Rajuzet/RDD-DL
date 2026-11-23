"""
Transfer Learning CNN Architectures with ImageNet Pretrained Weights:
- ResNet-18
- VGG-16
- MobileNet-V2
- DenseNet-121

Each model replaces the final classification layer with a tailored medical head,
and explicitly defines the optimal target convolutional layer for Grad-CAM.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def get_transfer_model(
    model_name: str = "resnet18",
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.3
) -> Tuple[nn.Module, nn.Module]:
    """
    Factory function returning (model, target_layer_for_gradcam).
    
    Args:
        model_name: 'resnet18', 'vgg16', 'mobilenetv2', 'densenet121'
        num_classes: Number of target categories (default: 4)
        pretrained: If True, loads ImageNet pretrained weights
        freeze_backbone: If True, freezes backbone layers for feature extraction only
        dropout_rate: Dropout probability in the new classification head
    """
    model_key = model_name.lower().replace("-", "").replace("_", "")
    weights_arg = "DEFAULT" if pretrained else None

    if model_key == "resnet18":
        model = models.resnet18(weights=weights_arg)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
                
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        target_layer = model.layer4[-1].conv2

    elif model_key == "vgg16":
        model = models.vgg16(weights=weights_arg)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        # Last conv layer before pooling in VGG16
        target_layer = model.features[28]

    elif model_key == "mobilenetv2":
        model = models.mobilenet_v2(weights=weights_arg)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        target_layer = model.features[-1][0]

    elif model_key == "densenet121":
        model = models.densenet121(weights=weights_arg)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        target_layer = model.features.denseblock4.denselayer16.conv2

    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. Choose from ['resnet18', 'vgg16', 'mobilenetv2', 'densenet121']."
        )

    return model, target_layer
