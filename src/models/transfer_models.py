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
    strategy: str = "full_finetuning",
    dropout_rate: float = 0.3
) -> Tuple[nn.Module, nn.Module]:
    """
    Factory function returning (model, target_layer_for_gradcam).
    
    Args:
        model_name: 'resnet18', 'vgg16', 'mobilenetv2', 'densenet121'
        num_classes: Number of target categories (default: 4)
        pretrained: If True, loads ImageNet pretrained weights
        strategy: 'full_finetuning', 'feature_extraction', or 'partial_finetuning'
        dropout_rate: Dropout probability in the new classification head
    """
    model_key = model_name.lower().replace("-", "").replace("_", "")
    weights_arg = "DEFAULT" if pretrained else None

    if model_key == "resnet18":
        model = models.resnet18(weights=weights_arg)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        target_layer = model.layer4[-1].conv2

        if strategy == "feature_extraction":
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        elif strategy == "partial_finetuning":
            # Freeze layers 1-3, keep layer4 and fc trainable
            for name, param in model.named_parameters():
                if not ("layer4" in name or "fc" in name):
                    param.requires_grad = False

    elif model_key == "vgg16":
        model = models.vgg16(weights=weights_arg)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        target_layer = model.features[28]

        if strategy == "feature_extraction":
            for param in model.features.parameters():
                param.requires_grad = False
        elif strategy == "partial_finetuning":
            # Keep only the last conv block (features[24:30]) and classifier trainable
            for idx, layer in enumerate(model.features):
                if idx < 24:
                    for param in layer.parameters():
                        param.requires_grad = False

    elif model_key == "mobilenetv2":
        model = models.mobilenet_v2(weights=weights_arg)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        target_layer = model.features[-1][0]

        if strategy == "feature_extraction":
            for param in model.features.parameters():
                param.requires_grad = False
        elif strategy == "partial_finetuning":
            # Unfreeze top 4 bottleneck blocks (15-18)
            for idx, block in enumerate(model.features):
                if idx < 15:
                    for param in block.parameters():
                        param.requires_grad = False

    elif model_key == "densenet121":
        model = models.densenet121(weights=weights_arg)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        target_layer = model.features.denseblock4.denselayer16.conv2

        if strategy == "feature_extraction":
            for param in model.features.parameters():
                param.requires_grad = False
        elif strategy == "partial_finetuning":
            # Keep denseblock4, norm5, and classifier trainable
            for name, param in model.features.named_parameters():
                if not ("denseblock4" in name or "norm5" in name):
                    param.requires_grad = False

    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. Choose from ['resnet18', 'vgg16', 'mobilenetv2', 'densenet121']."
        )

    return model, target_layer


def get_differential_learning_rates(
    model: nn.Module,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4
) -> list:
    """
    Constructs differential parameter groups:
    Applies lower learning rate to pretrained backbone and higher learning rate to classification head.
    """
    head_params = []
    backbone_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(h in name for h in ["fc", "classifier"]):
            head_params.append(param)
        else:
            backbone_params.append(param)
            
    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay}
    ]
