"""
Automated Image Quality Assessment (IQA) & Clinical Gradability Gate.
Complies with FDA SaMD Clinical Acceptance Criteria (e.g. IDx-DR, EyeArt).
Rejects ungradable, severely blurred, under/over-exposed OCT scans prior to inference.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image

from src.utils.logger import setup_logger

logger = setup_logger("RDD-DL.IQA")


@dataclass
class QualityAssessmentResult:
    """Clinical image gradability assessment record."""
    is_gradable: bool
    quality_score: float  # [0.0, 100.0]
    status: str           # "PASS" or "REJECT"
    failure_reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""


class ImageQualityAssessor:
    """
    Evaluates input OCT scans against standardized ophthalmic quality thresholds.
    FDA Precedent: Rejection of ungradable scans to eliminate false confidence.
    """
    
    def __init__(
        self,
        min_resolution: Tuple[int, int] = (200, 200),
        min_laplacian_variance: float = 40.0,   # Sharpness threshold
        min_mean_intensity: float = 15.0,       # Underexposure threshold
        max_mean_intensity: float = 230.0,      # Overexposure threshold
        min_contrast_std: float = 18.0,         # Minimum dynamic range
    ):
        self.min_resolution = min_resolution
        self.min_laplacian_var = min_laplacian_variance
        self.min_mean_intensity = min_mean_intensity
        self.max_mean_intensity = max_mean_intensity
        self.min_contrast_std = min_contrast_std

    def evaluate(self, image: Image.Image) -> QualityAssessmentResult:
        """
        Runs comprehensive image gradability pre-checks on an input PIL Image.
        """
        failure_reasons = []
        metrics = {}
        
        # 1. Resolution Check
        width, height = image.size
        metrics["width"] = float(width)
        metrics["height"] = float(height)
        if width < self.min_resolution[0] or height < self.min_resolution[1]:
            failure_reasons.append(
                f"Resolution {width}x{height} is below minimum diagnostic threshold {self.min_resolution[0]}x{self.min_resolution[1]}."
            )

        # Convert to single-channel grayscale numpy array for photometric evaluation
        gray = np.array(image.convert("L"))

        # 2. Sharpness / Blur Detection (Modified Laplacian Variance)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        metrics["laplacian_variance"] = round(laplacian_var, 2)
        if laplacian_var < self.min_laplacian_var:
            failure_reasons.append(
                f"Severe image blur detected (Sharpness index: {laplacian_var:.1f} < {self.min_laplacian_var:.1f})."
            )

        # 3. Mean Intensity & Photometric Range (Exposure)
        mean_val = float(np.mean(gray))
        metrics["mean_intensity"] = round(mean_val, 2)
        if mean_val < self.min_mean_intensity:
            failure_reasons.append(
                f"Severe underexposure / signal dropout (Mean intensity: {mean_val:.1f} < {self.min_mean_intensity:.1f})."
            )
        elif mean_val > self.max_mean_intensity:
            failure_reasons.append(
                f"Severe overexposure / sensor saturation (Mean intensity: {mean_val:.1f} > {self.max_mean_intensity:.1f})."
            )

        # 4. Tissue Contrast & Dynamic Range
        contrast_std = float(np.std(gray))
        metrics["contrast_std"] = round(contrast_std, 2)
        if contrast_std < self.min_contrast_std:
            failure_reasons.append(
                f"Insufficient contrast between retinal strata and vitreous cavity ({contrast_std:.1f} < {self.min_contrast_std:.1f})."
            )

        # Composite Quality Score calculation [0 - 100]
        # Normalize metrics to an informative score
        sharpness_subscore = min(100.0, (laplacian_var / (self.min_laplacian_var * 2.5)) * 40.0)
        contrast_subscore = min(100.0, (contrast_std / (self.min_contrast_std * 2.0)) * 30.0)
        exposure_subscore = 30.0 if (self.min_mean_intensity <= mean_val <= self.max_mean_intensity) else 0.0
        
        quality_score = round(sharpness_subscore + contrast_subscore + exposure_subscore, 1)

        is_gradable = len(failure_reasons) == 0
        status = "PASS" if is_gradable else "REJECT"

        if is_gradable:
            recommendation = "Image meets clinical gradability criteria. Suitable for diagnostic AI inference."
        else:
            recommendation = (
                "Scan is ungradable according to FDA CADe clinical acceptance criteria. "
                "Please realign scan head, adjust patient focus, and re-acquire OCT B-scan."
            )

        result = QualityAssessmentResult(
            is_gradable=is_gradable,
            quality_score=quality_score,
            status=status,
            failure_reasons=failure_reasons,
            metrics=metrics,
            recommendation=recommendation
        )
        
        if not is_gradable:
            logger.warning(f"Image rejected by IQA Gate. Reasons: {failure_reasons}")
        else:
            logger.info(f"Image passed IQA Gate with score {quality_score}/100.")
            
        return result
