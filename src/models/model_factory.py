"""
Model factory for building any of the 5 project architectures:
1. Custom CNN
2. ResNet-18
3. VGG-16
4. MobileNet-V2
5. DenseNet-121
"""

from typing import Dict, Tuple
import torch.nn as nn

from src.models.custom_cnn import CustomOCTCNN
from src.models.transfer_models import get_transfer_model


def build_model(
    model_name: str,
    num_classes: int = 4,
    pretrained: bool = True,
    strategy: str = "full_finetuning",
    freeze_backbone: bool = False,
    dropout_rate: float = 0.3
) -> Tuple[nn.Module, nn.Module]:
    """
    Unified builder returning (model, target_gradcam_layer).
    """
    normalized_name = model_name.lower().replace("-", "").replace("_", "")
    
    if freeze_backbone and strategy == "full_finetuning":
        strategy = "feature_extraction"
        
    if normalized_name in ["customcnn", "custom"]:
        model = CustomOCTCNN(num_classes=num_classes, dropout_rate=dropout_rate)
        target_layer = model.target_layer_for_gradcam
        return model, target_layer
    else:
        return get_transfer_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            strategy=strategy,
            dropout_rate=dropout_rate
        )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Counts total, trainable, and frozen parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen
    }
