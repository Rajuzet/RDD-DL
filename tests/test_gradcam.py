"""
Unit test for Grad-CAM Explainable AI.
"""

import numpy as np
from PIL import Image
import torch

from src.explainability.gradcam import GradCAM
from src.models.model_factory import build_model


def test_gradcam_generation_and_overlay():
    model, target_layer = build_model("resnet18", num_classes=4, pretrained=False)
    gradcam = GradCAM(model, target_layer)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    cam, pred_class, confidence = gradcam.generate_cam(dummy_input, target_class=0)
    
    assert cam.shape == (224, 224)
    assert 0.0 <= np.min(cam) <= np.max(cam) <= 1.0
    assert 0 <= pred_class <= 3
    assert 0.0 <= confidence <= 1.0
    
    # Test Overlay
    raw_img = Image.new("RGB", (224, 224), color="gray")
    overlay = GradCAM.overlay_heatmap(raw_img, cam, alpha=0.5)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8
    
    gradcam.remove_hooks()
