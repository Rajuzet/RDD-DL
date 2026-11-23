"""
Unit tests for the FastAPI clinical inference microservice.
"""

import io
from PIL import Image
from starlette.testclient import TestClient

from app.api import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "CNV" in data["supported_classes"]

def test_prediction_endpoint():
    # Create synthetic test OCT image in memory
    img = Image.new("RGB", (224, 224), color="gray")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    
    response = client.post(
        "/predict?model_name=resnet18",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert len(data["probabilities"]) == 4

def test_explain_gradcam_endpoint():
    img = Image.new("RGB", (224, 224), color="gray")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    
    response = client.post(
        "/explain/gradcam?model_name=resnet18&alpha=0.5",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "overlay_base64_png" in data
    assert data["overlay_base64_png"].startswith("data:image/png;base64,")
