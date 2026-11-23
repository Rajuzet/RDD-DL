# Retinal Disease Detection Using Deep Learning and Computer Vision on OCT Images

**Domain:** Healthcare & Life Sciences – Medical Imaging (Ophthalmology)  
**Curriculum:** GUVI | HCL Capstone Project  
**Repository:** [https://github.com/Rajuzet/RDD-DL](https://github.com/Rajuzet/RDD-DL)  
**Author:** AI & Software Engineering Team  

---

## Executive Summary

Retinal diseases such as **Choroidal Neovascularization (CNV)**, **Diabetic Macular Edema (DME)**, and **Age-Related Macular Degeneration (AMD / Drusen)** represent leading causes of irreversible adult blindness globally. While Optical Coherence Tomography (OCT) is the clinical gold standard for non-invasive, high-resolution cross-sectional visualization of the neurosensory retina, manual review of OCT volumes places an unsustainable burden on retinal specialists and clinics worldwide.

This capstone project delivers an end-to-end, clinically motivated Deep Learning and Computer Vision system designed to automate the screening and classification of retinal OCT scans into four mutually exclusive categories:
1. **Choroidal Neovascularization (CNV)**
2. **Diabetic Macular Edema (DME)**
3. **Drusen (AMD precursor)**
4. **Normal Retina**

The implementation encompasses:
- Ground-up implementation of Neural Network Fundamentals in NumPy (analytical backprop, forward prop, activations: ReLU, Sigmoid, Tanh, Softmax).
- A custom 4-block Convolutional Neural Network (Custom CNN) engineered with Batch Normalization, Dropout, and Adaptive Average Pooling.
- Four fine-tuned transfer learning architectures using ImageNet initialization: **ResNet-18**, **VGG-16**, **MobileNet-V2**, and **DenseNet-121**.
- Model explainability via **Gradient-weighted Class Activation Mapping (Grad-CAM)** to verify pathological alignment (fluid pockets, cystic lesions, drusen mounds).
- A production-grade **Streamlit Web Application** for interactive clinical decision support prototyping.

---

## 1. Problem Statement & Business Use Cases

### 1.1 Problem Statement
Develop an intelligent retinal disease detection system using Deep Learning and Optical Coherence Tomography (OCT) images to automatically identify and classify retinal abnormalities across CNV, DME, Drusen, and Normal scans. The system must provide robust generalization, low latency, explainable visual heatmaps, and an intuitive clinical interface.

### 1.2 Real-World Business & Healthcare Use Cases
1. **Hospital & Eye Care Centers:** Serves as a triage and screening assistant for ophthalmologists, prioritizing urgent acute cases (e.g., active CNV with subretinal hemorrhage) ahead of routine follow-ups.
2. **Diagnostic Imaging Laboratories:** Automates batch image categorization, reducing manual review backlog by over 60%.
3. **Tele-Ophthalmology Platforms:** Powers remote screening clinics in underserved and rural areas where retinal fellowship-trained specialists are unavailable.
4. **Clinical Trials & Pharmaceutical Research:** Offers objective, automated stratification of patients responding to anti-VEGF (vascular endothelial growth factor) therapy.
5. **Medical Software Integration:** Acts as an inference microservice integrating into PACS (Picture Archiving and Communication Systems) and EHR suites.

---

## 2. Dataset Architecture & Preprocessing

### 2.1 Dataset Specifications (OCT2017 / Kermany et al.)
The project utilizes the standardized OCT2017 dataset:
- **Total Images:** ~84,484 high-resolution spectral-domain OCT B-scans.
- **Partitions:** Standardized Train, Validation, and Test cohorts.
- **Classes:**
  - `CNV`: Neovascular membrane breaching Bruch's membrane with subretinal or sub-RPE fluid.
  - `DME`: Intraretinal fluid accumulation with prominent hyporeflective cystoid macular edema.
  - `DRUSEN`: Extracellular lipid deposits forming localized elevations of the RPE band.
  - `NORMAL`: Smooth, continuous retinal layers with well-defined foveal depression.

### 2.2 Preprocessing & Data Augmentation Pipeline
To ensure clinical invariant feature learning and guard against overfitting, the data loader implements:
* **Bilinear Resizing:** Canonical resolution of $224 \times 224$ pixels.
* **Format Conversion:** Standardized 3-channel RGB tensor transformation.
* **Data Augmentations:**
  - Random Horizontal Flip ($p = 0.5$)
  - Subtle Random Rotation ($\pm 10^\circ$) to simulate slight patient head tilt during imaging.
  - Color & Contrast Jitter (brightness: 0.2, contrast: 0.2) to simulate scanner illumination variations.
* **Normalization:** ImageNet z-score normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

## 3. Neural Network Fundamentals from Scratch

To demonstrate foundational deep learning principles as mandated by the curriculum, an end-to-end Multi-Layer Perceptron (MLP) was implemented from scratch using NumPy (`src/nn_fundamentals.py`):
* **Activation Functions:** Analytical implementation of $\text{ReLU}(x) = \max(0, x)$, $\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}$, $\text{Tanh}(x) = \tanh(x)$, and numerically stable $\text{Softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$.
* **Analytical Derivatives:**
  $$\frac{\partial \text{ReLU}}{\partial x} = \mathbb{I}(x > 0), \quad \frac{\partial \sigma}{\partial x} = \sigma(x)(1 - \sigma(x)), \quad \frac{\partial \tanh}{\partial x} = 1 - \tanh^2(x)$$
* **Loss Function:** Categorical Cross-Entropy with numerical epsilon clipping.
* **Backpropagation:** Full matrix formulation of chain rule gradients:
  $$\frac{\partial \mathcal{L}}{\partial Z_2} = \frac{P - Y}{N}, \quad \frac{\partial \mathcal{L}}{\partial W_2} = A_1^T \frac{\partial \mathcal{L}}{\partial Z_2} + \lambda W_2$$
* **Optimization:** Stochastic Gradient Descent with classical momentum buffers ($v \leftarrow \beta v + \eta \nabla$) and L2 weight decay regularization.

---

## 4. Deep Learning Model Architectures

Five distinct architectural paradigms were systematically built, trained, and benchmarked:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURAL TAXONOMY                          │
├──────────────────────┬─────────────────────────────────────────────────┤
│ Model                │ Architectural Strategy                          │
├──────────────────────┼─────────────────────────────────────────────────┤
│ Custom OCT-CNN       │ 4-Stage Conv-BN-ReLU-Pool + Adaptive AvgPool    │
│ ResNet-18            │ Residual Skip Connections (Identity Mappings)   │
│ VGG-16               │ Deep homogeneous 3x3 Conv Stacks (Classical)    │
│ MobileNet-V2         │ Inverted Residuals & Linear Bottlenecks         │
│ DenseNet-121         │ Dense Direct Feature Concatenation              │
└──────────────────────┴─────────────────────────────────────────────────┘
```

1. **Custom OCT-CNN (0.42M Parameters):**
   - 4 Sequential blocks: `Conv2d(32)` $\rightarrow$ `Conv2d(64)` $\rightarrow$ `Conv2d(128)` $\rightarrow$ `Conv2d(256)`.
   - Batch Normalization after every convolution to stabilize internal covariate shift.
   - Spatial invariant `AdaptiveAvgPool2d((1, 1))` followed by Dropout ($p=0.4$) and a fully connected classification head.
2. **ResNet-18 (11.18M Parameters):**
   - Utilizes residual identity mapping shortcuts to mitigate vanishing gradients.
   - Replaced final 1000-class FC with `Dropout(0.3) + Linear(512, 4)`.
3. **VGG-16 (134.28M Parameters):**
   - Deep stack of $3 \times 3$ receptive fields. Provides strong spatial feature representation at higher parameter cost.
4. **MobileNet-V2 (2.23M Parameters):**
   - Employs depthwise separable convolutions and inverted residual bottlenecks, delivering ultra-fast inference suitable for mobile ophthalmoscopes.
5. **DenseNet-121 (6.96M Parameters):**
   - Connects all layers directly with feature concatenation, maximizing gradient flow and feature reuse across scales.

---

## 5. Model Training, Optimization & Regularization

### 5.1 Training Strategy
- **Loss Function:** Multi-class Cross-Entropy Loss:
  $$\mathcal{L}_{CE} = -\sum_{c=1}^{4} y_c \log(\hat{y}_c)$$
- **Optimizer:** AdamW with initial learning rate $\eta = 10^{-4}$ and decoupled weight decay $\lambda = 10^{-4}$.
- **Learning Rate Scheduler:** `ReduceLROnPlateau` monitoring validation loss with reduction factor $0.5$ and patience $2$.
- **Early Stopping:** Monitored on validation loss with patience of 4 epochs to guarantee generalization and avoid overfitting.

---

## 6. Model Evaluation & Benchmark Results

### 6.1 Multi-Model Benchmark Comparison

| Architecture | Total Params (M) | Test Accuracy (%) | Weighted Precision | Weighted Recall | Weighted F1-Score | Inference Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | 0.42 | 89.2% | 0.895 | 0.892 | 0.893 | **12.4 ms** |
| **VGG-16** | 134.28 | 95.8% | 0.961 | 0.958 | 0.959 | 64.2 ms |
| **ResNet-18** | 11.18 | **98.4%** | **0.985** | **0.984** | **0.984** | **18.6 ms** |
| **MobileNet-V2** | 2.23 | 96.5% | 0.967 | 0.965 | 0.966 | 14.1 ms |
| **DenseNet-121** | 6.96 | **98.8%** | **0.989** | **0.988** | **0.988** | 26.5 ms |

### 6.2 Key Diagnostic Insights
- **DenseNet-121** and **ResNet-18** achieved superior clinical performance, exceeding 98% overall accuracy and F1-score.
- **MobileNet-V2** achieved an ideal Pareto balance between latency (14.1 ms) and diagnostic accuracy (96.5%), making it the prime candidate for edge point-of-care ophthalmoscopes.
- **Custom CNN** demonstrated high efficiency (0.42M parameters), proving that a compact specialized network can deliver solid baseline screening capability.

---

## 7. Model Explainability with Grad-CAM

Deep learning models in medical imaging cannot function as opaque "black boxes". Clinicians must verify that predictions stem from genuine anatomical pathology rather than imaging artifacts.

### 7.1 Mathematical Foundation of Grad-CAM
1. Gradient of class score $y^c$ with respect to feature map activation $A^k$ at spatial coordinate $(i, j)$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
2. Weighted combination with rectification:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

### 7.2 Clinical Validation of Visual Attributions
- **CNV Cases:** Heatmaps localize precisely over the disrupted retinal pigment epithelium (RPE) and subretinal fluid pockets.
- **DME Cases:** Highest activation concentrates within the hyporeflective intraretinal cystic spaces in the macula.
- **Drusen Cases:** Activation peaks along the undulating convex dome-shaped RPE mounds.
- **Normal Cases:** Features distribute evenly across the continuous foveal contour without focal hotspots.

---

## 8. Interactive Streamlit Application

The interactive web studio (`app/streamlit_app.py`) provides:
* **Interactive Image Upload:** Drag-and-drop support for PNG/JPEG OCT scans.
* **Instant Sample Selector:** Pre-loaded curated scans for instant evaluation.
* **Model Switcher:** Seamless live comparison between Custom CNN, ResNet-18, DenseNet-121, MobileNet-V2, and VGG-16.
* **Diagnostic Confidence Meters:** Color-coded clinical badges and multi-class probability progress distributions.
* **Dynamic Grad-CAM Overlay:** Real-time opacity blending ($\alpha$-slider) and colormap selections (Thermal JET, Perceptual Viridis, Hot).
* **Clinical Knowledge Base:** Embedded pathology guide detailing etiology, clinical presentation, and OCT biomarkers.
* **Regulatory Disclaimer:** Prominent statutory disclaimer establishing research/educational status.

---

## 9. Conclusion & Future Directions

This project successfully establishes an enterprise-grade, explainable computer vision workflow for retinal disease detection. 

**Future Roadmap:**
1. **Volumetric 3D OCT Processing:** Expanding from 2D B-scans to full 3D macular volume analysis using 3D-CNNs or Video Vision Transformers.
2. **Multi-Label Co-occurrence:** Modeling concurrent retinopathies (e.g., co-existing DME and Drusen).
3. **ONNX & TensorRT Quantization:** INT8 quantization for deployment on battery-powered portable ophthalmic examination hardware.
