"""
Unit test for dataset transformations and SafeImageFolder loader.
"""

from pathlib import Path
import torch

from src.config import config
from src.dataset import build_dataloaders, get_transforms


def test_transforms_pipeline():
    train_tf = get_transforms(image_size=224, is_train=True)
    val_tf = get_transforms(image_size=224, is_train=False)
    
    assert train_tf is not None
    assert val_tf is not None


def test_dataloader_generation_on_samples():
    train_loader, val_loader, test_loader, classes = build_dataloaders(
        data_dir=config.samples_dir,
        batch_size=4,
        num_workers=0
    )
    
    assert len(classes) == 4
    batch_images, batch_labels = next(iter(train_loader))
    assert batch_images.shape == (4, 3, 224, 224)
    assert batch_labels.shape == (4,)
