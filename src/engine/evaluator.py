"""
Model Evaluation and Benchmark Suite.
Computes:
- Accuracy, Weighted & Macro Precision, Recall, and F1-score
- Confusion Matrix visualizations with raw counts and normalized percentages
- Misclassification analysis and reporting
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import setup_logger

logger = setup_logger("RDD-DL.Evaluator")


class ModelEvaluator:
    """Evaluates PyTorch models on unseen test partitions with comprehensive diagnostics."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        classes: List[str],
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.classes = classes
        
    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs full test evaluation.
        
        Returns:
            metrics_dict, y_true, y_pred, y_probs
        """
        self.model.eval()
        all_preds: List[int] = []
        all_targets: List[int] = []
        all_probs: List[np.ndarray] = []
        
        for images, labels in self.test_loader:
            images = images.to(self.device, non_blocking=True)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_probs.append(probs)
            
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_probs = np.concatenate(all_probs, axis=0) if all_probs else np.array([])
        
        # Calculate global metrics
        acc = accuracy_score(y_true, y_pred)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        
        metrics = {
            "accuracy": float(acc),
            "precision_weighted": float(prec_w),
            "recall_weighted": float(rec_w),
            "f1_weighted": float(f1_w),
            "precision_macro": float(prec_m),
            "recall_macro": float(rec_m),
            "f1_macro": float(f1_m),
        }
        
        logger.info(f"Test Accuracy: {acc*100:.2f}% | F1 (Weighted): {f1_w:.4f} | Recall: {rec_w:.4f}")
        return metrics, y_true, y_pred, y_probs

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
        """Plots publication-grade raw count and normalized confusion matrices."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute counts
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=self.classes, yticklabels=self.classes, ax=ax1
        )
        ax1.set_title("Test Confusion Matrix (Sample Counts)", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Predicted Diagnosis", fontsize=11)
        ax1.set_ylabel("True Ground Truth", fontsize=11)
        
        # Normalized percentages
        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Blues", cbar=True,
            xticklabels=self.classes, yticklabels=self.classes, ax=ax2
        )
        ax2.set_title("Test Confusion Matrix (Normalized Recall %)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Predicted Diagnosis", fontsize=11)
        ax2.set_ylabel("True Ground Truth", fontsize=11)
        
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix visualization to {output_path}")

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generates sklearn classification report."""
        report = classification_report(y_true, y_pred, target_names=self.classes, zero_division=0)
        return report

    def extract_misclassifications(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
        max_samples: int = 10
    ) -> List[Dict]:
        """Identifies misclassified cases for qualitative diagnostic review."""
        misclassified = []
        mismatch_indices = np.where(y_true != y_pred)[0]
        
        for idx in mismatch_indices[:max_samples]:
            true_idx = int(y_true[idx])
            pred_idx = int(y_pred[idx])
            misclassified.append({
                "sample_index": int(idx),
                "true_label": self.classes[true_idx],
                "predicted_label": self.classes[pred_idx],
                "confidence": float(y_probs[idx][pred_idx]),
                "true_class_probability": float(y_probs[idx][true_idx])
            })
            
        return misclassified
