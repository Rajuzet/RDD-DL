"""
Utility script to generate realistic synthetic retinal OCT sample scans
for each of the 4 diagnostic categories (NORMAL, CNV, DME, DRUSEN).
Allows immediate out-of-the-box local testing, unit tests, and Streamlit demo.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from src.config import config
from src.utils.logger import setup_logger

logger = setup_logger("RDD-DL.GenerateSamples")


def draw_base_retina(width: int = 512, height: int = 512) -> np.ndarray:
    """Generates standard baseline stratified retinal tissue cross-section."""
    # Dark vitreous cavity at top, subtle speckle noise
    img = np.random.normal(loc=18, scale=6, size=(height, width)).astype(np.float32)
    
    # Retinal nerve fiber and inner layers (y: 200 - 320)
    for y in range(190, 330):
        intensity = 65 + 30 * np.sin((y - 190) * np.pi / 140)
        img[y, :] += intensity
        
    # Hyper-reflective RPE (Retinal Pigment Epithelium) band (y: 330 - 345)
    img[330:345, :] += 140.0
    
    # Choroid layer (y: 345 - 420)
    img[345:420, :] += 50.0
    
    # Add speckle noise characteristic of optical coherence tomography
    speckle = np.random.gamma(shape=2.0, scale=8.0, size=(height, width))
    img += speckle
    
    return np.clip(img, 0, 255).astype(np.uint8)


def create_normal_sample(path: Path):
    """Normal retina with clean foveal contour."""
    img = draw_base_retina()
    # Add gentle foveal pit depression in center
    h, w = img.shape
    center_x = w // 2
    for x in range(center_x - 100, center_x + 100):
        depth = int(25 * np.cos((x - center_x) * np.pi / 200))
        if depth > 0:
            img[190:190 + depth, x] = np.random.normal(20, 5, depth)
            
    # Apply Gaussian smoothing for realistic tissue appearance
    img = cv2.GaussianBlur(img, (3, 3), 0)
    Image.fromarray(img).convert("RGB").save(str(path))


def create_cnv_sample(path: Path):
    """CNV: Neovascular complex breaching RPE with subretinal fluid accumulation."""
    img = draw_base_retina()
    # Subretinal fluid pocket (hyporeflective / dark cavity under retina)
    cv2.ellipse(img, (256, 310), (70, 25), 0, 0, 360, 25, -1)
    # Hyperreflective fibrovascular membrane lesion breaching RPE
    cv2.ellipse(img, (256, 335), (45, 18), 0, 0, 360, 220, -1)
    
    img = cv2.GaussianBlur(img, (3, 3), 0)
    Image.fromarray(img).convert("RGB").save(str(path))


def create_dme_sample(path: Path):
    """DME: Intraretinal cystoid macular edema (hyporeflective cystic fluid cavities)."""
    img = draw_base_retina()
    # Thicken retinal layer upward due to edema
    img[160:190, 160:360] += 70
    
    # Multiple distinct cystic cavities inside neurosensory retina
    cysts = [(210, 250, 22, 14), (260, 240, 30, 20), (310, 255, 20, 15), (250, 280, 25, 16)]
    for cx, cy, rx, ry in cysts:
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, 20, -1)
        
    img = cv2.GaussianBlur(img, (3, 3), 0)
    Image.fromarray(img).convert("RGB").save(str(path))


def create_drusen_sample(path: Path):
    """DRUSEN: Undulating domed mounds underneath RPE layer."""
    img = draw_base_retina()
    # Multiple dome-shaped mounds elevating RPE
    drusen_centers = [180, 230, 280, 330]
    for dc in drusen_centers:
        cv2.circle(img, (dc, 338), 16, 170, -1)
        
    img = cv2.GaussianBlur(img, (3, 3), 0)
    Image.fromarray(img).convert("RGB").save(str(path))


def generate_all_samples(samples_per_class: int = 5):
    """Populates samples directory with sample cases for all 4 classes."""
    base_dir = config.samples_dir
    for cls in config.classes:
        cls_dir = base_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(1, samples_per_class + 1):
            file_path = cls_dir / f"{cls.lower()}_sample_{i:02d}.jpeg"
            if cls == "NORMAL":
                create_normal_sample(file_path)
            elif cls == "CNV":
                create_cnv_sample(file_path)
            elif cls == "DME":
                create_dme_sample(file_path)
            elif cls == "DRUSEN":
                create_drusen_sample(file_path)
                
    logger.info(f"Generated {samples_per_class * 4} validated OCT scan samples in {base_dir}")


if __name__ == "__main__":
    generate_all_samples(samples_per_class=6)
