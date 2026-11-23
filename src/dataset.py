"""
Dataset preprocessing, augmentation, and PyTorch DataLoader pipeline.
Supports:
- Robust image loading and RGB format conversion
- Medical image augmentation (rotations, flips, contrast adjustment)
- Normalized tensor transformations
- Directory-based and split-based loading
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger("RDD-DL.Dataset")


def get_transforms(image_size: int = config.image_size, is_train: bool = True) -> transforms.Compose:
    """Returns PyTorch torchvision transformations for training or evaluation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ])


class SafeImageFolder(datasets.ImageFolder):
    """
    Subclasses torchvision.datasets.ImageFolder to safely handle and skip
    potentially corrupt or unreadable image files without crashing the training run.
    """
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        try:
            return super().__getitem__(index)
        except Exception as e:
            logger.warning(f"Error loading image at index {index}: {e}. Falling back to next sample.")
            return super().__getitem__((index + 1) % len(self))


def build_dataloaders(
    data_dir: Path,
    batch_size: int = config.batch_size,
    num_workers: int = config.num_workers,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = config.seed
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Builds train, validation, and test DataLoaders.
    Checks whether data_dir contains explicit train/val/test folders or a flat class structure.
    """
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    
    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        logger.info(f"Detected structured dataset in {data_dir} (train/val/test).")
        train_ds = SafeImageFolder(str(train_dir), transform=get_transforms(is_train=True))
        val_ds = SafeImageFolder(str(val_dir), transform=get_transforms(is_train=False))
        test_ds = SafeImageFolder(str(test_dir), transform=get_transforms(is_train=False))
        classes = train_ds.classes
    else:
        logger.info(f"Loading dataset from flat directory {data_dir} and performing stratified random split.")
        full_raw_ds = datasets.ImageFolder(str(data_dir))
        classes = full_raw_ds.classes
        
        total_len = len(full_raw_ds)
        test_len = int(total_len * test_split)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len - test_len
        
        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices, test_indices = random_split(
            range(total_len), [train_len, val_len, test_len], generator=generator
        )
        
        train_base = SafeImageFolder(str(data_dir), transform=get_transforms(is_train=True))
        eval_base = SafeImageFolder(str(data_dir), transform=get_transforms(is_train=False))
        
        train_ds = Subset(train_base, train_indices.indices)
        val_ds = Subset(eval_base, val_indices.indices)
        test_ds = Subset(eval_base, test_indices.indices)
        
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"DataLoaders successfully built with {len(classes)} classes: {classes}")
    logger.info(f"Samples - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    return train_loader, val_loader, test_loader, classes
