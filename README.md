# Retinal Disease Detection Using Deep Learning on OCT Images

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Code Style](https://img.shields.io/badge/code%20style-PEP%208-green.svg)](https://peps.python.org/pep-0008/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(10%2F10)-brightgreen.svg)](#testing--verification)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, clinical-decision-support deep learning pipeline and interactive web application for automated detection, classification, and explainability of retinal abnormalities from **Optical Coherence Tomography (OCT)** images.

Distinguishes between four primary diagnostic categories:
1. **CNV (Choroidal Neovascularization)**
2. **DME (Diabetic Macular Edema)**
3. **DRUSEN (Age-Related Macular Degeneration precursor)**
4. **NORMAL (Intact retinal architecture)**

---

## Architecture & Pipeline

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                        OCT2017 Retinal Images                          │
 │                (CNV, DME, DRUSEN, NORMAL: ~84,000 Scans)               │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │               Data Preprocessing & Augmentation Pipeline               │
 │    - SafeImageFolder loader with error recovery                        │
 │    - Bilinear resize (224x224), RGB tensor standardization             │
 │    - Augmentations: Random Flips, Rotations (±10°), Color/Jitter       │
 │    - ImageNet Normalization (mean=[0.485,0.456,0.406], std=[0.229...]) │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
 ┌────────────────────────────────┐            ┌────────────────────────────────┐
 │ Neural Network Fundamentals    │            │ 5 Deep Learning Architectures  │
 │ - Multi-Layer Perceptron (MLP) │            │ 1. Custom OCT-CNN (from scratch)│
 │   from scratch in NumPy        │            │ 2. ResNet-18 (Residual)        │
 │ - Analytical Forward/Backward  │            │ 3. VGG-16 (Deep Classical)     │
 │ - ReLU, Sigmoid, Tanh, Softmax │            │ 4. MobileNet-V2 (Lightweight)  │
 │ - Cross-Entropy + Momentum SGD │            │ 5. DenseNet-121 (Dense Reuse)  │
 └────────────────────────────────┘            └────────────────┬───────────────┘
                                                                │
                                                                ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                  Training, Optimization & Regularization               │
 │    - AdamW Optimizer (lr=1e-4, weight_decay=1e-4)                      │
 │    - Multi-Class Cross-Entropy Loss with Label Smoothing               │
 │    - Early Stopping on Validation Loss (patience=4)                    │
 │    - Learning Rate Scheduling via ReduceLROnPlateau                    │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
 ┌────────────────────────────────┐            ┌────────────────────────────────┐
 │ Quantitative Evaluation Suite  │            │ Explainable AI (Grad-CAM)      │
 │ - Accuracy, Precision, Recall, │            │ - Hook-based gradient tracking │
 │   F1-Score (Weighted & Macro)  │            │ - Class activation heatmaps    │
 │ - Normalized Confusion Matrices│            │ - Anatomic overlay & alignment │
 │ - Latency profiling (ms/scan)  │            │   (fluid vs drusen validation) │
 └────────────────────────────────┘            └────────────────┬───────────────┘
                                                                │
                                                                ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │               Production Streamlit Web Studio (app/app.py)             │
 │    - Real-time drag-and-drop OCT image upload                          │
 │    - Pre-loaded curated clinical sample selector                       │
 │    - Dynamic multi-model prediction with confidence gauges             │
 │    - Interactive Grad-CAM heatmap overlay with opacity slider          │
 │    - Clinical pathology knowledge base & statutory disclaimer          │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```text
RDD-DL/
├── app/
│   └── streamlit_app.py         # Production Streamlit clinical studio
├── data/
│   ├── raw/                     # Raw downloaded OCT2017 dataset
│   ├── processed/               # Preprocessed train/val/test splits
│   └── samples/                 # Curated sample scans across all 4 classes
├── models/
│   └── checkpoints/             # Trained PyTorch model checkpoints (.pt)
├── notebooks/
│   ├── 01_eda_and_nn_fundamentals.ipynb           # EDA & NumPy from-scratch MLP
│   └── 02_model_development_and_explainability.ipynb # Model benchmarking & Grad-CAM
├── reports/
│   ├── PROJECT_REPORT.md        # Comprehensive technical & clinical documentation
│   ├── PRESENTATION.md          # Capstone presentation slide deck outline
│   ├── model_comparison.csv     # Benchmark metrics across all 5 models
│   ├── model_comparison.json    # JSON benchmark metrics
│   ├── confusion_matrix_*.png   # Test confusion matrices
│   └── learning_curves_*.png    # Loss and accuracy curves
├── scripts/
│   ├── download_data.py         # Hugging Face & Kaggle dataset acquisition tool
│   ├── generate_sample_data.py  # Generates verified sample scans for testing
│   ├── train_benchmark.py       # Full training, evaluation & benchmark orchestrator
│   └── create_notebooks.py      # Automated notebook generator
├── src/
│   ├── __init__.py
│   ├── config.py                # Strongly-typed project configuration dataclass
│   ├── dataset.py               # Robust PyTorch Dataset, transforms & DataLoaders
│   ├── nn_fundamentals.py       # NumPy neural network with analytical backpropagation
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loop with early stopping & checkpoints
│   │   └── evaluator.py         # Test evaluation, metrics & confusion matrix
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── gradcam.py           # Grad-CAM forward/backward hook engine & overlay
│   ├── models/
│   │   ├── __init__.py
│   │   ├── custom_cnn.py        # Custom 4-stage CNN built from scratch
│   │   ├── transfer_models.py   # ResNet18, VGG16, MobileNetV2, DenseNet121
│   │   └── model_factory.py     # Unified model builder & parameter counter
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Standardized console & file logging
│       └── seed.py              # Reproducibility seed management
├── tests/
│   ├── test_dataset.py          # Unit tests for transforms & dataloaders
│   ├── test_gradcam.py          # Unit tests for Grad-CAM & overlay blending
│   ├── test_models.py           # Unit tests for all 5 neural architectures
│   └── test_nn_fundamentals.py  # Unit tests for NumPy activations & backpropagation
├── requirements.txt             # Pinned production dependencies
├── setup.py                     # Python package distribution definition
├── .gitignore                   # Strict Git ignore configuration
└── README.md                    # Project documentation
```

---

## Benchmark Results

All five architectures were evaluated on unseen OCT test partitions:

| Architecture | Total Params (M) | Test Accuracy (%) | Weighted Precision | Weighted Recall | Weighted F1-Score | Inference Latency (ms) | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Custom CNN** | 0.42 | 89.2% | 0.895 | 0.892 | 0.893 | **12.4 ms** | `best_custom_cnn.pt` |
| **VGG-16** | 134.28 | 95.8% | 0.961 | 0.958 | 0.959 | 64.2 ms | `best_vgg16.pt` |
| **ResNet-18** | 11.18 | **98.4%** | **0.985** | **0.984** | **0.984** | **18.6 ms** | `best_resnet18.pt` |
| **MobileNet-V2** | 2.23 | 96.5% | 0.967 | 0.965 | 0.966 | 14.1 ms | `best_mobilenetv2.pt` |
| **DenseNet-121** | 6.96 | **98.8%** | **0.989** | **0.988** | **0.988** | 26.5 ms | `best_densenet121.pt` |

### Key Takeaways:
- **DenseNet-121** achieved the highest overall diagnostic accuracy (98.8%) and F1-score due to dense feature reuse across spatial scales.
- **ResNet-18** demonstrated an optimal balance between accuracy (98.4%) and computational efficiency (18.6 ms latency).
- **MobileNet-V2** proved to be the top choice for resource-constrained edge computing environments.

---

## Quickstart Guide

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/Rajuzet/RDD-DL.git
cd RDD-DL

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 2. Run Unit Test Suite

Execute the test suite covering data loaders, neural network fundamentals, all 5 deep learning architectures, and Grad-CAM explainability:

```bash
python -m pytest -v
```

All 10 tests run in under 25 seconds with 100% pass rate.

### 3. Generate Sample Data (Out-of-the-Box Demo)

The repository includes an automated sample generator so you can test immediately without downloading the full 84,000-image dataset:

```bash
python scripts/generate_sample_data.py
```

To download the full OCT2017 dataset from Hugging Face:

```bash
python scripts/download_data.py --download
```

### 4. Train & Benchmark Models

Train and evaluate the models with automated early stopping, checkpointing, and metric logging:

```bash
python scripts/train_benchmark.py --epochs 10 --models resnet18 densenet121 mobilenetv2 custom_cnn
```

Generated metrics, confusion matrices, and learning curves will be stored in `reports/`.

### 5. Launch Interactive Clinical Studio (Streamlit)

Start the web application:

```bash
streamlit run app/streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Explainable AI (Grad-CAM)

Clinical adoption of deep learning requires interpretability. The Grad-CAM implementation tracks gradients flowing into the final convolutional layer:
- **CNV:** Heatmap isolates the hyperreflective subretinal neovascular membrane and sub-RPE fluid pockets.
- **DME:** Strong activation occurs over intraretinal hyporeflective cystic spaces.
- **DRUSEN:** Focuses specifically on the undulating dome-shaped elevations of the RPE layer.
- **NORMAL:** Diffuse, uniform activation across intact parallel retinal strata.

---

## Clinical & Statutory Disclaimer

> **IMPORTANT NOTICE:** This software repository, deep learning models, and interactive web application are developed strictly for **academic research, educational prototyping, and portfolio demonstration** as part of the GUVI | HCL capstone curriculum. It is **not** an FDA-cleared, CE-marked, or certified medical device and must never be utilized for primary clinical diagnosis or medical decision-making.

---

## References & Citations

1. Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. **Cell**, 172(5), 1122-1131.
2. Selvaraju, R. R., Cogswell, M., Das, A., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. **IEEE ICCV**, 618-626.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. **IEEE CVPR**, 770-778.
4. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). *Densely Connected Convolutional Networks*. **IEEE CVPR**, 4700-4708.
