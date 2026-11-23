"""
Unit tests for FDA SaMD compliance modules:
- Automated Image Quality Assessment (IQA)
- Clinical Probability Calibration & Triage Alerting
- 21 CFR Part 11 Audit Trail Logging
"""

import io
import json
import numpy as np
from PIL import Image
import pytest

from src.compliance.audit_trail import AuditTrailLogger
from src.compliance.calibration import ClinicalCalibrator
from src.quality.image_quality_assessment import ImageQualityAssessor


def test_iqa_valid_oct_image():
    assessor = ImageQualityAssessor()
    # Create realistic contrast/sharpness gradient image
    arr = np.tile(np.linspace(20, 200, 256, dtype=np.uint8), (256, 1))
    # Add high-frequency edge
    arr[120:140, :] = 250
    img = Image.fromarray(arr).convert("RGB")
    
    result = assessor.evaluate(img)
    assert result.is_gradable is True
    assert result.status == "PASS"
    assert result.quality_score > 50.0
    assert len(result.failure_reasons) == 0


def test_iqa_reject_underexposed_image():
    assessor = ImageQualityAssessor()
    # Almost pitch black image
    arr = np.zeros((256, 256), dtype=np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    
    result = assessor.evaluate(img)
    assert result.is_gradable is False
    assert result.status == "REJECT"
    assert any("underexposure" in r.lower() or "contrast" in r.lower() for r in result.failure_reasons)


def test_clinical_calibration_and_referral():
    calibrator = ClinicalCalibrator(temperature=1.2)
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    
    # Raw uncalibrated logits
    logits = np.array([2.5, 0.5, 0.2, 0.1])
    calibrated_probs = calibrator.calibrate_logits(logits)
    
    assert np.isclose(np.sum(calibrated_probs), 1.0)
    # Scaled probabilities should be slightly smoothed by T > 1
    assert calibrated_probs[0] > 0.5
    
    # Check referral alert for CNV
    referral = calibrator.check_high_sensitivity_referral(calibrated_probs, classes)
    assert referral["requires_urgent_retinal_referral"] is True
    assert referral["alerts"][0]["condition"] == "CNV"


def test_audit_trail_logging(tmp_path):
    audit_file = tmp_path / "test_audit.jsonl"
    logger = AuditTrailLogger(log_path=audit_file)
    
    fake_bytes = b"OCT_TEST_IMAGE_BYTES_12345"
    record = logger.record_inference(
        image_bytes=fake_bytes,
        model_name="resnet18",
        iqa_status="PASS",
        iqa_score=88.5,
        prediction="CNV",
        confidence=94.2,
        probabilities={"CNV": 0.942, "DME": 0.03, "DRUSEN": 0.018, "NORMAL": 0.01},
        urgent_flag=True,
        operator_id="OP_DR_SMITH"
    )
    
    assert record.audit_id is not None
    assert len(record.image_sha256) == 64  # SHA-256 length
    
    # Verify file was written
    assert audit_file.exists()
    with open(audit_file, "r", encoding="utf-8") as f:
        line = f.readline()
        data = json.loads(line)
        assert data["primary_prediction"] == "CNV"
        assert data["operator_id"] == "OP_DR_SMITH"
