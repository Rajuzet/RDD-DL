# Capstone Project Presentation Deck Outline

**Project Title:** Retinal Disease Detection Using Deep Learning on OCT Images  
**Domain:** Healthcare & Life Sciences – Medical Imaging (Ophthalmology)  
**Presenter:** AI & Software Engineering Team  
**Evaluation Session:** GUVI | HCL Live Capstone Evaluation  

---

### Slide 1: Title & Executive Overview
* **Title:** Retinal Disease Detection Using Deep Learning on OCT Images
* **Subtitle:** An Explainable AI & Computer Vision Framework for Automated Retinal Screening
* **Key Highlights:**
  - 4 Diagnostic Categories: CNV, DME, DRUSEN, NORMAL
  - 5 CNN Architectures Evaluated (Custom CNN + 4 Transfer Learning models)
  - Explainable AI via Grad-CAM Biomarker Localization
  - Real-time Clinical Prototyping in Streamlit

---

### Slide 2: Clinical Problem Statement & Motivation
* **The Clinical Need:**
  - Retinal diseases (AMD, Diabetic Retinopathy, Neovascularization) cause severe vision loss if undiagnosed.
  - Optical Coherence Tomography (OCT) is the diagnostic standard, generating hundreds of scans per patient session.
  - Shortage of certified retinal specialists causes diagnostic bottlenecks and delayed treatment.
* **Project Objective:**
  - Build an end-to-end automated screening tool that accurately categorizes OCT B-scans with >98% accuracy and visualizes pathological evidence for clinician verification.

---

### Slide 3: The 4 Target Pathologies
* **Choroidal Neovascularization (CNV):**
  - Neovascular membrane breaking through Bruch's membrane; presence of subretinal fluid.
* **Diabetic Macular Edema (DME):**
  - Macular thickening with fluid-filled intraretinal cystoid spaces.
* **Drusen:**
  - Acellular lipid/protein deposits between RPE and Bruch's membrane; key precursor of dry AMD.
* **Normal Retina:**
  - Symmetrical foveal pit contour with uninterrupted retinal laminations.

---

### Slide 4: End-to-End System Architecture
* **Data Preprocessing & Augmentation:**
  - Canonical $224 \times 224$ resizing, RGB conversion, ImageNet normalization.
  - Data Augmentations: Random Flips, subtle rotations ($\pm 10^\circ$), and color/contrast jitter.
* **Foundational Modeling:**
  - NumPy implementation of Multi-Layer Perceptron (ReLU, Sigmoid, Tanh, Softmax, analytical backpropagation).
* **PyTorch Deep Learning Pipeline:**
  - Custom 4-block CNN built from scratch.
  - Transfer learning backbones: ResNet-18, VGG-16, MobileNet-V2, DenseNet-121.

---

### Slide 5: Model Optimization & Regularization Strategy
* **Cross-Entropy Loss** with AdamW optimizer ($\text{lr}=10^{-4}$, $\text{weight decay}=10^{-4}$).
* **Overfitting Safeguards:**
  - Spatial Dropout ($p=0.3 - 0.4$) in classification heads.
  - Batch Normalization in convolutional blocks.
  - Dynamic learning rate adjustment via `ReduceLROnPlateau`.
  - Automated `EarlyStopping` based on validation loss.

---

### Slide 6: Benchmark Results & Architecture Comparison
* **Comparative Summary:**
  - **DenseNet-121:** Highest overall accuracy (98.8%) and F1-score (0.988). Dense feature reuse excels at capturing subtle layer disruptions.
  - **ResNet-18:** Excellent clinical accuracy (98.4%) with low latency (18.6 ms).
  - **MobileNet-V2:** Ultra-lightweight (2.23M parameters) and fast (14.1 ms latency), optimal for edge devices.
  - **Custom CNN:** Lightweight footprint (0.42M parameters), verifying from-scratch design feasibility.
* **Evaluation Metrics:**
  - Confusion Matrices showing clear separation between fluid-filled lesions (CNV/DME) and dry drusen mounds.

---

### Slide 7: Explainable AI with Grad-CAM
* **Why Explainability Matters:**
  - Healthcare AI requires trust; "black box" decisions cannot be audited in patient care.
* **Grad-CAM Insights:**
  - Maps gradients back to the final convolutional feature maps.
  - Visual overlays confirm activation over active pathology:
    * CNV $\rightarrow$ highlights subretinal fluid and fibrovascular membranes.
    * DME $\rightarrow$ highlights intraretinal cystic fluid cavities.
    * Drusen $\rightarrow$ highlights undulating RPE elevations.

---

### Slide 8: Interactive Streamlit Application Demo
* **Features Demonstrated:**
  - Dual Input: Drag-and-drop file upload or pre-loaded clinical sample selector.
  - Dynamic Architecture Selector: Instant comparison of predictions across models.
  - Probability Breakdown: Confidence meter and multi-class distribution bars.
  - Real-time Grad-CAM Heatmap generation with opacity slider and colormap controls.
  - Clinical Knowledge Base & Statutory Medical Disclaimer.

---

### Slide 9: Challenges & Engineering Solutions
* **Data Imbalance & Volume:** Solved via stratified mini-batches and tensor pipelines.
* **Corrupted Image Handling:** Engineered `SafeImageFolder` wrapper to handle damaged files gracefully.
* **Interpretability Validation:** Verified Grad-CAM heatmap alignment with anatomical landmarks.

---

### Slide 10: Conclusion & Future Scope
* **Key Achievements:**
  - Complete end-to-end medical AI pipeline from foundational math to production UI.
  - Reliable performance across 5 diverse neural network architectures.
* **Next Steps:**
  - Extension to 3D volumetric OCT scans (OCT-Angiography).
  - Multi-label classification for concurrent retinopathies.
  - Edge deployment using ONNX runtime on embedded clinical equipment.
