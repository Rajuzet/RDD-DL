"""
Retinal OCT Disease Detection & Explainability Studio
Production Streamlit Application for Clinical Decision Support Prototyping.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.config import config
from src.explainability.gradcam import GradCAM
from src.models.model_factory import build_model, count_parameters


st.set_page_config(
    page_title="Retinal OCT AI Diagnostic Studio",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Clinical/FAANG aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        border-radius: 8px;
        padding: 1rem;
        color: #92400E;
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    .badge-normal { background-color: #DEF7EC; color: #03543F; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-cnv { background-color: #FDE8E8; color: #9B1C1C; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-dme { background-color: #FEECDC; color: #92400E; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
    .badge-drusen { background-color: #E1EFFE; color: #1E429F; padding: 4px 10px; border-radius: 6px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_oct_model(model_name: str):
    """Loads and caches selected architecture and checkpoints."""
    model, target_layer = build_model(model_name, num_classes=config.num_classes, pretrained=False)
    
    checkpoint_path = config.checkpoints_dir / f"best_{model_name}.pt"
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
    
    model.eval()
    gradcam = GradCAM(model, target_layer)
    return model, gradcam


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocesses input PIL image into normalized tensor."""
    tf = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std),
    ])
    tensor = tf(image.convert("RGB")).unsqueeze(0)
    return tensor


# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/visible.png", width=64)
    st.title("Diagnostic Settings")
    
    selected_model_name = st.selectbox(
        "Neural Network Architecture",
        options=["resnet18", "densenet121", "mobilenetv2", "custom_cnn", "vgg16"],
        index=0,
        format_func=lambda x: {
            "resnet18": "ResNet-18 (Balanced / Fast)",
            "densenet121": "DenseNet-121 (Dense Feature Reuse)",
            "mobilenetv2": "MobileNet-V2 (Lightweight / Edge)",
            "custom_cnn": "Custom 4-Block CNN (From Scratch)",
            "vgg16": "VGG-16 (Deep Classical)"
        }.get(x, x)
    )
    
    st.markdown("---")
    st.subheader("Explainability (Grad-CAM)")
    cam_alpha = st.slider("Heatmap Blend Opacity (α)", min_value=0.1, max_value=0.9, value=0.45, step=0.05)
    colormap_choice = st.selectbox(
        "Colormap",
        options=["JET (Thermal)", "VIRIDIS (Perceptual)", "HOT (Intense)"],
        index=0
    )
    cm_map = {
        "JET (Thermal)": cv2.COLORMAP_JET,
        "VIRIDIS (Perceptual)": cv2.COLORMAP_VIRIDIS,
        "HOT (Intense)": cv2.COLORMAP_HOT
    }
    
    st.markdown("---")
    st.subheader("Sample Scans")
    sample_options = ["None (Upload Custom)"]
    sample_files = []
    if config.samples_dir.exists():
        for f in config.samples_dir.rglob("*.jpeg"):
            sample_options.append(f"{f.parent.name} - {f.name}")
            sample_files.append(f)
            
    selected_sample = st.selectbox("Load Sample OCT Scan", options=sample_options)


# Header
st.markdown('<div class="main-header">Retinal OCT Disease Detection & Explainability Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning Screening for CNV, Diabetic Macular Edema, Drusen, and Normal Retinal Scans</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_predict, tab_benchmark, tab_pathology, tab_system = st.tabs([
    "🔬 Diagnostic Inference",
    "📊 Model Benchmarks",
    "📖 Clinical Pathology Guide",
    "⚙️ System Architecture"
])

model, gradcam = load_oct_model(selected_model_name)

with tab_predict:
    col_input, col_pred = st.columns([1, 1], gap="large")
    
    active_image = None
    image_source_label = ""
    
    with col_input:
        st.subheader("1. Input Retinal OCT Scan")
        uploaded_file = st.file_uploader(
            "Upload high-resolution OCT scan (DICOM-converted JPEG, PNG)",
            type=["jpeg", "jpg", "png"]
        )
        
        if uploaded_file is not None:
            active_image = Image.open(uploaded_file).convert("RGB")
            image_source_label = f"Uploaded File: {uploaded_file.name}"
        elif selected_sample != "None (Upload Custom)":
            sample_idx = sample_options.index(selected_sample) - 1
            sample_path = sample_files[sample_idx]
            active_image = Image.open(sample_path).convert("RGB")
            image_source_label = f"Sample: {sample_path.parent.name} / {sample_path.name}"
            
        if active_image is not None:
            st.image(active_image, caption=image_source_label, use_column_width=True)
            w, h = active_image.size
            st.caption(f"Dimensions: {w} × {h} pixels | Channels: 3 (RGB)")
        else:
            st.info("👆 Please upload an OCT cross-section image or pick a curated sample from the sidebar.")

    with col_pred:
        st.subheader("2. Diagnostic Analysis & Explainability")
        if active_image is not None:
            input_tensor = preprocess_image(active_image)
            
            # Prediction
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
            pred_class_idx = int(np.argmax(probs))
            pred_class_name = config.classes[pred_class_idx]
            confidence_score = float(probs[pred_class_idx]) * 100.0
            
            # Badge selection
            badge_class = f"badge-{pred_class_name.lower()}"
            st.markdown(f"""
            <div class="metric-card">
                <span class="{badge_class}">{pred_class_name}</span>
                <h2 style="margin: 0.4rem 0 0.1rem 0; color: #111827;">{confidence_score:.2f}% Confidence</h2>
                <p style="margin: 0; color: #6B7280; font-size: 0.95rem;">{config.class_descriptions[pred_class_name]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability distribution bars
            st.markdown("##### Prediction Probability Distribution")
            for cls_name, prob_val in zip(config.classes, probs):
                st.progress(float(prob_val), text=f"{cls_name}: {prob_val * 100:.2f}%")
                
            # Grad-CAM Generation
            st.markdown("##### Explainable AI (Grad-CAM Biomarker Localization)")
            cam, _, _ = gradcam.generate_cam(input_tensor, target_class=pred_class_idx)
            overlay = GradCAM.overlay_heatmap(
                original_image=active_image,
                heatmap=cam,
                alpha=cam_alpha,
                colormap=cm_map[colormap_choice]
            )
            
            col_cam1, col_cam2 = st.columns(2)
            with col_cam1:
                st.image(cam, caption="Raw Activation Heatmap", use_column_width=True, clamp=True)
            with col_cam2:
                st.image(overlay, caption="Grad-CAM Anatomic Overlay", use_column_width=True)
                
            st.caption("🔴 Red / Warm regions indicate high convolutional feature activation contributing most strongly to the predicted diagnosis.")
        else:
            st.write("Awaiting image input for classification...")

with tab_benchmark:
    st.subheader("Comprehensive Architecture Benchmarking")
    st.markdown("Comparative performance metrics across all 5 models evaluated on unseen OCT test partitions:")
    
    csv_path = config.reports_dir / "model_comparison.csv"
    if csv_path.exists():
        df_bench = pd.read_csv(csv_path)
        st.dataframe(df_bench.style.highlight_max(subset=["Accuracy (%)", "F1-Score (Weighted)"], color="#DCFCE7"), use_container_width=True)
    else:
        st.info("Run `python scripts/train_benchmark.py` to populate real-time evaluation logs.")
        
    st.markdown("#### Performance Visualizations")
    col_cm, col_lc = st.columns(2)
    with col_cm:
        cm_file = config.reports_dir / f"confusion_matrix_{selected_model_name}.png"
        if cm_file.exists():
            st.image(str(cm_file), caption=f"Test Confusion Matrix ({selected_model_name.upper()})", use_column_width=True)
    with col_lc:
        lc_file = config.reports_dir / f"learning_curves_{selected_model_name}.png"
        if lc_file.exists():
            st.image(str(lc_file), caption=f"Learning Curves ({selected_model_name.upper()})", use_column_width=True)

with tab_pathology:
    st.subheader("Ophthalmological Disease Reference")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
        ### 1. Choroidal Neovascularization (CNV)
        * **Etiology:** Proliferation of abnormal, fragile new blood vessels originating from the choroid that penetrate through Bruch's membrane into the sub-retinal pigment epithelium (sub-RPE) space.
        * **Clinical Presentation:** Rapid vision loss, metamorphopsia (distorted vision).
        * **OCT Biomarkers:** Hyperreflective subretinal or sub-RPE fibrovascular tissue mounds, adjacent hyporeflective subretinal fluid (SRF), and intraretinal cystoid fluid.
        
        ### 2. Diabetic Macular Edema (DME)
        * **Etiology:** Breakdown of the blood-retinal barrier secondary to chronic diabetic retinopathy, resulting in fluid leakage into the macular neurosensory layers.
        * **Clinical Presentation:** Central blur, decreased contrast sensitivity.
        * **OCT Biomarkers:** Hyporeflective intraretinal cystic spaces (cystoid macular edema), sponge-like retinal thickening, and loss of normal foveal dip.
        """)
    with col_d2:
        st.markdown("""
        ### 3. Drusen (Age-Related Macular Degeneration)
        * **Etiology:** Focal extracellular debris deposits containing lipids and proteins between the basal lamina of the retinal pigment epithelium (RPE) and the inner collagenous layer of Bruch's membrane.
        * **Clinical Presentation:** Early stages often asymptomatic; dry AMD can progress to geographic atrophy or wet CNV.
        * **OCT Biomarkers:** Distinct, undulating dome-shaped convex elevations of the RPE band with variable internal reflectivity and absence of subretinal fluid.
        
        ### 4. Normal Retina
        * **Anatomical Integrity:** Symmetrical, concave foveal depression with continuous, well-defined horizontal reflective layers (RNFL, IPL, INL, OPL, ONL, ELM, IS/OS ellipsoid zone, and intact RPE-Bruch's complex).
        """)

with tab_system:
    st.subheader("System Architecture & Specifications")
    st.markdown("""
    * **Inference Pipeline:** PyTorch 2.x Deep Learning Engine.
    * **Input Preprocessing:** Canonical 224×224 resolution with ImageNet normalization $\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$.
    * **Explainable AI (XAI):** Analytical Grad-CAM hook with spatial gradient Global Average Pooling.
    * **Compute Device:** CPU / CUDA Automatic hardware acceleration.
    """)
    param_info = count_parameters(model)
    st.json(param_info)

# Statutory Clinical Disclaimer Box
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ STATUTORY MEDICAL & REGULATORY DISCLAIMER:</strong><br>
    This web application and the underlying deep learning models are developed strictly for research, educational prototyping, and academic demonstration under the GUVI/HCL Capstone curriculum. This software is <strong>NOT</strong> an FDA-cleared, CE-marked, or certified medical diagnostic device. It must never be used as a substitute for professional clinical judgment, expert ophthalmic consultation, or official diagnostic imaging reports.
</div>
""", unsafe_allow_html=True)
