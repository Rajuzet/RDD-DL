"""
Unit tests for all 5 Deep Learning Architectures:
1. Custom CNN
2. ResNet-18
3. VGG-16
4. MobileNet-V2
5. DenseNet-121
"""

import pytest
import torch

from src.models.model_factory import build_model, count_parameters


@pytest.mark.parametrize("model_name", ["custom_cnn", "resnet18", "vgg16", "mobilenetv2", "densenet121"])
def test_model_forward_pass_and_shape(model_name):
    model, target_layer = build_model(model_name, num_classes=4, pretrained=False)
    model.eval()
    
    # Batch of 2 synthetic OCT images
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (2, 4), f"Expected shape (2, 4), received {output.shape} for {model_name}"
    assert target_layer is not None, f"Target layer missing for {model_name}"
    
    params = count_parameters(model)
    assert params["total_parameters"] > 0
    assert params["trainable_parameters"] > 0


def test_transfer_learning_strategies():
    # Feature extraction mode should freeze backbone parameters
    model_fe, _ = build_model("resnet18", num_classes=4, pretrained=False, strategy="feature_extraction")
    params_fe = count_parameters(model_fe)
    assert params_fe["frozen_parameters"] > 0
    assert params_fe["trainable_parameters"] < params_fe["total_parameters"]

    # Partial finetuning mode
    model_pf, _ = build_model("resnet18", num_classes=4, pretrained=False, strategy="partial_finetuning")
    params_pf = count_parameters(model_pf)
    assert params_pf["frozen_parameters"] > 0
    assert params_pf["trainable_parameters"] > params_fe["trainable_parameters"]
