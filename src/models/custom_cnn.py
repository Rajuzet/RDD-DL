"""
Custom Convolutional Neural Network (CNN) built from scratch using PyTorch.
Designed with 4 sequential Conv-BatchNorm-ReLU-Pooling stages, Adaptive Pooling,
and Dropout regularization for robust retinal OCT feature extraction.
"""

from typing import Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class CustomOCTCNN(nn.Module):
    """
    Custom 4-stage CNN architecture for Retinal OCT image classification.
    Input resolution: (3, 224, 224)
    Feature map transitions: 224 -> 112 -> 56 -> 28 -> 14
    """
    
    def __init__(self, num_classes: int = 4, in_channels: int = 3, dropout_rate: float = 0.4):
        super(CustomOCTCNN, self).__init__()
        
        self.block1 = ConvBlock(in_channels, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)
        
        # Adaptive pooling produces invariant (256, 1, 1) representation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Kaiming normal initialization for conv layers and Xavier for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @property
    def target_layer_for_gradcam(self) -> nn.Module:
        """Returns the final convolutional layer for Grad-CAM activation mapping."""
        return self.block4.conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits
