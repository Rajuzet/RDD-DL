"""
Production FastAPI Microservice for Retinal OCT Disease Detection & Grad-CAM Explainability.
Provides OpenAPI / Swagger UI endpoints for automated PACS and EHR inference integration.
"""

import base64
import io
import time
from typing import Dict, List, Optional
import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.config import config
from src.compliance.audit_trail import AuditTrailLogger
from src.compliance.calibration import ClinicalCalibrator
from src.explainability.gradcam import GradCAM
from src.models.model_factory import build_model, count_parameters
from src.quality.image_quality_assessment import ImageQualityAssessor
from src.utils.logger import setup_logger

logger = setup_logger("RDD-DL.API")

iqa_assessor = ImageQualityAssessor()
calibrator = ClinicalCalibrator(temperature=1.15)
audit_logger = AuditTrailLogger()

app = FastAPI(
    title="Retinal Disease Detection (RDD-DL) Clinical API",
    description="Deep Learning and Explainable AI Inference Service for Retinal OCT Scans (CNV, DME, Drusen, Normal).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware for web/clinical client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODEL_CACHE: Dict[str, tuple] = {}

def get_or_load_model(model_name: str = "resnet18"):
    model_key = model_name.lower().replace("-", "").replace("_", "")
    if model_key not in MODEL_CACHE:
        logger.info(f"Loading weights into cache for architecture: {model_name}")
        model, target_layer = build_model(model_name, num_classes=config.num_classes, pretrained=False)
        checkpoint_path = config.checkpoints_dir / f"best_{model_name}.pt"
        if checkpoint_path.exists():
            state_dict = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(state_dict)
        model.to(config.device)
        model.eval()
        gradcam = GradCAM(model, target_layer)
        MODEL_CACHE[model_key] = (model, gradcam)
    return MODEL_CACHE[model_key]

class ClassProbability(BaseModel):
    category: str
    probability: float

class PredictionResponse(BaseModel):
    model: str
    iqa_status: str
    iqa_score: float
    predicted_class: str
    confidence: float
    description: str
    probabilities: List[ClassProbability]
    urgent_referral_flag: bool
    audit_id: str
    inference_time_ms: float
    statutory_disclaimer: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    device: str
    supported_classes: List[str]

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """Liveness probe and environment health check."""
    return HealthResponse(
        status="HEALTHY",
        service="Retinal Disease Detection Inference Engine",
        version="1.0.0",
        device=str(config.device),
        supported_classes=config.classes
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_oct_scan(
    file: UploadFile = File(...),
    model_name: str = Query("resnet18", description="Choice of model: resnet18, densenet121, mobilenetv2, custom_cnn, vgg16")
):
    """
    Submits an OCT scan B-scan (PNG/JPEG) for automated diagnostic classification.
    Runs pre-analytical Image Quality Assessment (IQA) and Temperature Calibration.
    """
    start_time = time.time()
    try:
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Pre-analytical Image Quality Assessment Gate (FDA CADe Requirement)
    iqa_result = iqa_assessor.evaluate(pil_img)
    if not iqa_result.is_gradable:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "UNGRADABLE_IMAGE",
                "message": "Image failed FDA SaMD clinical acceptance criteria.",
                "quality_score": iqa_result.quality_score,
                "failure_reasons": iqa_result.failure_reasons,
                "recommendation": iqa_result.recommendation
            }
        )

    tf = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])
    input_tensor = tf(pil_img).unsqueeze(0).to(config.device)

    model, _ = get_or_load_model(model_name)
    with torch.no_grad():
        logits = model(input_tensor).cpu().numpy()[0]

    # Temperature Scaling Calibration
    probs = calibrator.calibrate_logits(logits)
    pred_idx = int(np.argmax(probs))
    pred_label = config.classes[pred_idx]
    confidence_pct = float(probs[pred_idx]) * 100.0

    prob_dict = {cls_name: float(prob_val) for cls_name, prob_val in zip(config.classes, probs)}
    triage_check = calibrator.check_high_sensitivity_referral(probs, config.classes)

    # 21 CFR Part 11 Audit Trail Logging
    audit_rec = audit_logger.record_inference(
        image_bytes=image_bytes,
        model_name=model_name,
        iqa_status=iqa_result.status,
        iqa_score=iqa_result.quality_score,
        prediction=pred_label,
        confidence=confidence_pct,
        probabilities=prob_dict,
        urgent_flag=triage_check["requires_urgent_retinal_referral"],
        operator_id="API_CLIENT"
    )

    prob_list = [
        ClassProbability(category=cls_name, probability=round(float(prob_val), 4))
        for cls_name, prob_val in zip(config.classes, probs)
    ]

    duration_ms = round((time.time() - start_time) * 1000.0, 2)

    return PredictionResponse(
        model=model_name,
        iqa_status=iqa_result.status,
        iqa_score=iqa_result.quality_score,
        predicted_class=pred_label,
        confidence=round(confidence_pct, 2),
        description=config.class_descriptions[pred_label],
        probabilities=prob_list,
        urgent_referral_flag=triage_check["requires_urgent_retinal_referral"],
        audit_id=audit_rec.audit_id,
        inference_time_ms=duration_ms,
        statutory_disclaimer="RESEARCH PROTOTYPE ONLY - NOT FOR DIRECT CLINICAL DIAGNOSIS."
    )

@app.post("/explain/gradcam", tags=["Explainability"])
async def explain_oct_scan(
    file: UploadFile = File(...),
    model_name: str = Query("resnet18", description="Choice of model: resnet18, densenet121, mobilenetv2, custom_cnn, vgg16"),
    alpha: float = Query(0.45, ge=0.1, le=0.9, description="Overlay blend opacity")
):
    """
    Computes analytical Grad-CAM heatmap and returns base64-encoded visual overlay.
    """
    try:
        image_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    tf = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])
    input_tensor = tf(pil_img).unsqueeze(0).to(config.device)

    model, gradcam = get_or_load_model(model_name)
    cam, pred_class, confidence = gradcam.generate_cam(input_tensor)
    overlay_rgb = GradCAM.overlay_heatmap(pil_img, cam, alpha=alpha, colormap=cv2.COLORMAP_JET)

    # Encode overlay image to base64 PNG
    overlay_pil = Image.fromarray(overlay_rgb)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "predicted_class": config.classes[pred_class],
        "confidence": round(float(confidence) * 100.0, 2),
        "overlay_base64_png": f"data:image/png;base64,{b64_str}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
