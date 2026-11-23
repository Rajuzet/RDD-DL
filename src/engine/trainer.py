"""
Robust PyTorch Model Training Engine with Early Stopping,
Learning Rate Scheduling, and Metric History Logging.
"""

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import setup_logger

logger = setup_logger("RDD-DL.Trainer")


class ModelTrainer:
    """Orchestrates model training, validation, checkpointing, and early stopping."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        epochs: int = 15,
        early_stopping_patience: int = 4,
        early_stopping_min_delta: float = 1e-3,
        checkpoint_path: Optional[Path] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=2, factor=0.5
        )
        self.epochs = epochs
        self.patience = early_stopping_patience
        self.min_delta = early_stopping_min_delta
        self.checkpoint_path = checkpoint_path
        
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "learning_rate": []
        }
        
    def train_epoch(self) -> Tuple[float, float]:
        """Runs a single training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            
        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate_epoch(self) -> Tuple[float, float]:
        """Runs validation evaluation."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            
        val_loss = running_loss / max(total, 1)
        val_acc = correct / max(total, 1)
        return val_loss, val_acc

    def fit(self) -> Dict[str, List[float]]:
        """Executes full training with early stopping."""
        logger.info(f"Initiating training on device '{self.device}' for {self.epochs} epochs.")
        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self.model.state_dict())
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            
            # Step scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
                
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rate"].append(current_lr)
            
            duration = time.time() - epoch_start
            logger.info(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] ({duration:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {current_lr:.6f}"
            )
            
            # Checkpoint and Early Stopping Check
            if val_loss < (best_val_loss - self.min_delta):
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                if self.checkpoint_path:
                    self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_weights, self.checkpoint_path)
                    logger.info(f"Saved new best model checkpoint to {self.checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Restoring best model.")
                    break
                    
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes.")
        self.model.load_state_dict(best_weights)
        return self.history

    def plot_history(self, output_path: Path) -> None:
        """Plots and saves loss and accuracy learning curves."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss Plot
        ax1.plot(epochs, self.history["train_loss"], "b-o", label="Train Loss", linewidth=2)
        ax1.plot(epochs, self.history["val_loss"], "r--s", label="Val Loss", linewidth=2)
        ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross Entropy Loss")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()
        
        # Accuracy Plot
        ax2.plot(epochs, self.history["train_acc"], "b-o", label="Train Accuracy", linewidth=2)
        ax2.plot(epochs, self.history["val_acc"], "g--s", label="Val Accuracy", linewidth=2)
        ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=300)
        plt.close()
        logger.info(f"Saved learning curves to {output_path}")
