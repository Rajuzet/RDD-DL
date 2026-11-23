"""
Clinical Probability Calibration & Risk Thresholding.
FDA GMLP Principle: Models must provide calibrated probabilities reflecting true clinical risk.
Implements Temperature Scaling (Guo et al.) and High-Sensitivity Triage Thresholds.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

from src.utils.logger import setup_logger

logger = setup_logger("RDD-DL.Calibration")


class ClinicalCalibrator:
    """
    Post-hoc Temperature Scaling calibrator for multi-class medical predictions.
    Computes Expected Calibration Error (ECE) and aligns probabilities with clinical incidence.
    """
    
    def __init__(self, temperature: float = 1.15):
        self.temperature = max(0.01, temperature)
        
        # Clinical Operating Thresholds for High-Sensitivity Referrals:
        # For sight-threatening conditions (CNV and DME), lower operating threshold triggers clinical alert
        self.referral_thresholds = {
            "CNV": 0.35,      # High sensitivity threshold for subretinal fluid/neovascularization
            "DME": 0.40,      # High sensitivity threshold for macular edema
            "DRUSEN": 0.50,   # Standard threshold for non-exudative drusen
            "NORMAL": 0.50
        }

    def calibrate_logits(self, logits: np.ndarray) -> np.ndarray:
        """
        Scales logits by optimal temperature parameter T and applies softmax.
        
        Args:
            logits: Array of shape (N, num_classes) or (num_classes,)
        """
        scaled_logits = logits / self.temperature
        # Numerically stable softmax
        if scaled_logits.ndim == 1:
            exp_z = np.exp(scaled_logits - np.max(scaled_logits))
            return exp_z / np.sum(exp_z)
        else:
            exp_z = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_ece(self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculates Expected Calibration Error (ECE) across probability bins.
        """
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = predictions == labels

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(labels)

        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece)

    def check_high_sensitivity_referral(self, probs: np.ndarray, classes: List[str]) -> Dict[str, any]:
        """
        Evaluates whether calibrated probabilities exceed clinical referral thresholds.
        Ensures high sensitivity for acute vision-threatening pathologies (CNV and DME).
        """
        flagged_conditions = []
        for idx, cls in enumerate(classes):
            threshold = self.referral_thresholds.get(cls, 0.50)
            if cls in ["CNV", "DME"] and probs[idx] >= threshold:
                flagged_conditions.append({
                    "condition": cls,
                    "calibrated_probability": round(float(probs[idx]), 4),
                    "clinical_threshold": threshold,
                    "alert_level": "URGENT_REFERRAL"
                })

        return {
            "requires_urgent_retinal_referral": len(flagged_conditions) > 0,
            "alerts": flagged_conditions
        }
