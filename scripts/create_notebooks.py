"""
Script to generate production-quality Jupyter Notebooks for the project:
1. 01_eda_and_nn_fundamentals.ipynb
2. 02_model_development_and_explainability.ipynb
"""

import json
from pathlib import Path

def create_notebook(cells, output_path: Path):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated notebook: {output_path}")

def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    }

def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    }

def generate_eda_notebook(dest_path: Path):
    cells = [
        make_markdown_cell([
            "# Retinal Disease Detection Using Deep Learning on OCT Images",
            "## Part 1: Dataset Exploration, Preprocessing & Neural Network Fundamentals",
            "**Domain:** Healthcare & Life Sciences – Medical Imaging (Ophthalmology)",
            "**Classes:** CNV, DME, DRUSEN, NORMAL"
        ]),
        make_code_cell([
            "import os",
            "from pathlib import Path",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from PIL import Image",
            "import torch",
            "",
            "from src.config import config",
            "from src.dataset import build_dataloaders, get_transforms",
            "from src.nn_fundamentals import Activations, LossFunctions, ScratchNeuralNetwork",
            "from src.utils.seed import seed_everything",
            "",
            "seed_everything(config.seed)",
            "print(f'PyTorch Version: {torch.__version__} | Hardware Acceleration: {config.device}')"
        ]),
        make_markdown_cell([
            "---",
            "### 1. Dataset Collection and Class Distribution Analysis",
            "Analyzing sample counts and class balance across the 4 diagnostic categories."
        ]),
        make_code_cell([
            "sample_counts = {}",
            "for cls in config.classes:",
            "    cls_dir = config.samples_dir / cls",
            "    images = list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.png'))",
            "    sample_counts[cls] = len(images)",
            "",
            "plt.figure(figsize=(8, 4))",
            "sns.barplot(x=list(sample_counts.keys()), y=list(sample_counts.values()), palette='crest')",
            "plt.title('OCT Dataset Class Balance Distribution', fontsize=12, fontweight='bold')",
            "plt.xlabel('Diagnostic Category')",
            "plt.ylabel('Number of Images')",
            "plt.grid(axis='y', linestyle='--', alpha=0.7)",
            "plt.show()"
        ]),
        make_markdown_cell([
            "---",
            "### 2. Sample Image Visualization across Pathologies",
            "Visualizing the cross-sectional anatomy for Normal, CNV, DME, and Drusen."
        ]),
        make_code_cell([
            "fig, axes = plt.subplots(1, 4, figsize=(18, 5))",
            "for idx, cls in enumerate(config.classes):",
            "    cls_dir = config.samples_dir / cls",
            "    sample_img_path = list(cls_dir.glob('*.*'))[0]",
            "    img = Image.open(sample_img_path).convert('RGB')",
            "    axes[idx].imshow(img)",
            "    axes[idx].set_title(f'{cls}\\n{sample_img_path.name}', fontsize=11, fontweight='bold')",
            "    axes[idx].axis('off')",
            "plt.tight_layout()",
            "plt.show()"
        ]),
        make_markdown_cell([
            "---",
            "### 3. Neural Network Fundamentals from Scratch (NumPy)",
            "Testing activation functions (ReLU, Sigmoid, Tanh, Softmax) and analytical gradient backpropagation."
        ]),
        make_code_cell([
            "# Generate toy synthetic diagnostic features",
            "np.random.seed(42)",
            "N_samples = 200",
            "X_toy = np.random.randn(N_samples, 8)",
            "y_labels = np.random.randint(0, 4, size=N_samples)",
            "y_one_hot = np.zeros((N_samples, 4))",
            "y_one_hot[np.arange(N_samples), y_labels] = 1.0",
            "",
            "mlp = ScratchNeuralNetwork(input_dim=8, hidden_dim=16, output_dim=4, activation='relu', learning_rate=0.05)",
            "history = mlp.fit(X_toy, y_one_hot, epochs=60)",
            "",
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))",
            "ax1.plot(history['loss'], 'r-', label='Cross-Entropy Loss', linewidth=2)",
            "ax1.set_title('NumPy MLP Training Loss Convergence')",
            "ax1.set_xlabel('Epoch')",
            "ax1.grid(True, linestyle='--', alpha=0.6)",
            "ax1.legend()",
            "",
            "ax2.plot(history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)",
            "ax2.set_title('NumPy MLP Accuracy')",
            "ax2.set_xlabel('Epoch')",
            "ax2.grid(True, linestyle='--', alpha=0.6)",
            "ax2.legend()",
            "plt.show()"
        ])
    ]
    create_notebook(cells, dest_path)

def generate_modeling_notebook(dest_path: Path):
    cells = [
        make_markdown_cell([
            "# Retinal Disease Detection Using Deep Learning on OCT Images",
            "## Part 2: Custom CNN, Transfer Learning & Explainable AI (Grad-CAM)",
            "**Domain:** Healthcare & Life Sciences – Medical Imaging (Ophthalmology)"
        ]),
        make_code_cell([
            "import torch",
            "import matplotlib.pyplot as plt",
            "from PIL import Image",
            "import pandas as pd",
            "",
            "from src.config import config",
            "from src.dataset import build_dataloaders",
            "from src.models.model_factory import build_model, count_parameters",
            "from src.engine.trainer import ModelTrainer",
            "from src.engine.evaluator import ModelEvaluator",
            "from src.explainability.gradcam import GradCAM",
            "",
            "print(f'Active Device: {config.device}')"
        ]),
        make_markdown_cell([
            "---",
            "### 1. Architectural Parameter Comparison",
            "Evaluating parameters and complexity across Custom CNN and Transfer Learning backbones."
        ]),
        make_code_cell([
            "architectures = ['custom_cnn', 'resnet18', 'vgg16', 'mobilenetv2', 'densenet121']",
            "table = []",
            "for arch in architectures:",
            "    m, _ = build_model(arch, num_classes=4, pretrained=False)",
            "    p = count_parameters(m)",
            "    table.append({",
            "        'Architecture': arch,",
            "        'Total Parameters (M)': round(p['total_parameters'] / 1e6, 2),",
            "        'Trainable Parameters (M)': round(p['trainable_parameters'] / 1e6, 2)",
            "    })",
            "df_params = pd.DataFrame(table)",
            "df_params"
        ]),
        make_markdown_cell([
            "---",
            "### 2. Explainable AI with Grad-CAM",
            "Generating class activation maps on a sample OCT image to verify pathological focus."
        ]),
        make_code_cell([
            "# Load ResNet-18 model and Grad-CAM extractor",
            "model, target_layer = build_model('resnet18', num_classes=4, pretrained=True)",
            "gradcam = GradCAM(model, target_layer)",
            "",
            "# Load a sample CNV image",
            "sample_path = list((config.samples_dir / 'CNV').glob('*.jpeg'))[0]",
            "raw_img = Image.open(sample_path).convert('RGB')",
            "",
            "tf = config.mean  # verify normalization",
            "from torchvision import transforms",
            "input_tensor = transforms.Compose([",
            "    transforms.Resize((224, 224)),",
            "    transforms.ToTensor(),",
            "    transforms.Normalize(mean=config.mean, std=config.std)",
            "])(raw_img).unsqueeze(0)",
            "",
            "cam, pred_class, conf = gradcam.generate_cam(input_tensor)",
            "overlay = GradCAM.overlay_heatmap(raw_img, cam, alpha=0.5)",
            "",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 5))",
            "axes[0].imshow(raw_img)",
            "axes[0].set_title('Original OCT Scan', fontsize=12, fontweight='bold')",
            "axes[0].axis('off')",
            "",
            "axes[1].imshow(cam, cmap='jet')",
            "axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')",
            "axes[1].axis('off')",
            "",
            "axes[2].imshow(overlay)",
            "axes[2].set_title(f'Diagnostic Overlay\\nPrediction: {config.classes[pred_class]} ({conf*100:.1f}%)', fontsize=12, fontweight='bold')",
            "axes[2].axis('off')",
            "plt.tight_layout()",
            "plt.show()"
        ])
    ]
    create_notebook(cells, dest_path)

if __name__ == "__main__":
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)
    generate_eda_notebook(nb_dir / "01_eda_and_nn_fundamentals.ipynb")
    generate_modeling_notebook(nb_dir / "02_model_development_and_explainability.ipynb")
