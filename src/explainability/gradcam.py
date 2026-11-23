"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation.
Provides explainable AI visualizations highlighting pathological biomarkers
(subretinal fluid, macular edema cysts, drusen accumulations).
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn


class GradCAM:
    """
    Grad-CAM extractor for PyTorch Convolutional Neural Networks.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.hooks = []
        
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
        
    def remove_hooks(self):
        """Removes registered PyTorch hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates normalized Grad-CAM activation heatmap for the given input.
        
        Args:
            input_tensor: Tensor of shape (1, C, H, W)
            target_class: Class index to explain. If None, uses top predicted class.
            
        Returns:
            Tuple of (heatmap: np.ndarray [H, W], predicted_class: int, confidence: float)
        """
        self.model.eval()
        
        # Enable gradient tracking for backpropagation
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())
            
        confidence = float(probabilities[0, target_class].item())
        
        # Zero gradients and backpropagate targeted class score
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations from target layer.")
            
        gradients = self.gradients[0].cpu().numpy()     # (Channels, H, W)
        activations = self.activations[0].cpu().numpy() # (Channels, H, W)
        
        # Global Average Pooling on gradients across spatial dimensions
        weights = np.mean(gradients, axis=(1, 2))      # (Channels,)
        
        # Linear combination of weighted forward activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to focus on features that positively influence the target class
        cam = np.maximum(cam, 0.0)
        
        # Resize CAM to match input tensor spatial dimensions
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        
        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, target_class, confidence

    @staticmethod
    def overlay_heatmap(
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Resizes and blends Grad-CAM heatmap over original PIL Image.
        
        Returns:
            RGB blended image as uint8 numpy array.
        """
        width, height = original_image.size
        # Resize heatmap to match original image dimensions
        resized_cam = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        
        # Convert heatmap to 8-bit and apply colormap
        heatmap_uint8 = np.uint8(255 * resized_cam)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        
        original_np = np.array(original_image.convert("RGB"))
        
        # Alpha blending
        overlay = np.uint8(alpha * colored_heatmap + (1.0 - alpha) * original_np)
        return overlay
