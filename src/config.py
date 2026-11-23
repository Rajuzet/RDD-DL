from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import torch

@dataclass
class ProjectConfig:
    """Centralized configuration for dataset, modeling, and evaluation."""
    
    # Project Info
    project_name: str = "Retinal Disease Detection Using Deep Learning on OCT Images"
    version: str = "1.0.0"
    
    # Class Definitions
    classes: List[str] = field(default_factory=lambda: ["CNV", "DME", "DRUSEN", "NORMAL"])
    num_classes: int = 4
    
    class_descriptions: dict = field(default_factory=lambda: {
        "CNV": "Choroidal Neovascularization - Pathological neovascular membrane with subretinal fluid or hemorrhage.",
        "DME": "Diabetic Macular Edema - Intraretinal fluid buildup causing cystic spaces and retinal thickening.",
        "DRUSEN": "Drusen Deposits - Extracellular lipid-protein accumulation between RPE and Bruch's membrane (AMD marker).",
        "NORMAL": "Normal Retina - Intact foveal contour, uniform retinal stratification without fluid accumulation or elevation."
    })
    
    # Data Specifications
    image_size: int = 224
    image_channels: int = 3
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Training Hyperparameters
    batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 15
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-3
    dropout_rate: float = 0.3
    
    # Random Seed
    seed: int = 42
    
    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    samples_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    
    # Device
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def __post_init__(self):
        self.data_dir = self.base_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.samples_dir = self.data_dir / "samples"
        self.checkpoints_dir = self.base_dir / "models" / "checkpoints"
        self.reports_dir = self.base_dir / "reports"
        
        # Ensure runtime directories exist
        for d in [self.data_dir, self.samples_dir, self.checkpoints_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

# Default singleton instance
config = ProjectConfig()
