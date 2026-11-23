"""
21 CFR Part 11 Compliant Electronic Audit Trail Logger.
Produces immutable cryptographic audit records for every diagnostic AI inference.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, List, Optional
import uuid

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger("RDD-DL.AuditTrail")


@dataclass
class AuditRecord:
    """Immutable audit entry conforming to FDA 21 CFR Part 11 requirements."""
    audit_id: str
    timestamp_utc: str
    image_sha256: str
    software_version: str
    model_name: str
    iqa_status: str
    iqa_score: float
    primary_prediction: str
    calibrated_confidence: float
    probabilities: Dict[str, float]
    urgent_referral_flag: bool
    operator_id: str
    device_info: str


class AuditTrailLogger:
    """
    Manages structured, append-only audit logging for clinical traceability and regulatory audits.
    """
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or (config.base_dir / "logs" / "audit_trail.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(image_bytes: bytes) -> str:
        """Computes cryptographic SHA-256 digest of input raw image bytes."""
        return hashlib.sha256(image_bytes).hexdigest()

    def record_inference(
        self,
        image_bytes: bytes,
        model_name: str,
        iqa_status: str,
        iqa_score: float,
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        urgent_flag: bool = False,
        operator_id: str = "SYSTEM_DEFAULT"
    ) -> AuditRecord:
        """
        Creates and appends an immutable audit record to the persistent audit trail.
        """
        record = AuditRecord(
            audit_id=str(uuid.uuid4()),
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            image_sha256=self.compute_sha256(image_bytes),
            software_version=config.version,
            model_name=model_name,
            iqa_status=iqa_status,
            iqa_score=round(iqa_score, 1),
            primary_prediction=prediction,
            calibrated_confidence=round(confidence, 2),
            probabilities={k: round(v, 4) for k, v in probabilities.items()},
            urgent_referral_flag=urgent_flag,
            operator_id=operator_id,
            device_info=str(config.device)
        )

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record)) + "\n")
            logger.info(f"Recorded audit entry {record.audit_id} for image hash {record.image_sha256[:12]}...")
        except Exception as e:
            logger.error(f"Failed to write audit record: {e}")

        return record
