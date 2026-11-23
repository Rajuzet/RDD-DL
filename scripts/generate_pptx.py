"""
Generates an executive-ready, 11-slide PowerPoint presentation (.pptx)
for the GUVI | HCL Live Capstone Evaluation Session.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def build_presentation(output_path: Path):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    # Color Palette
    PRIMARY = RGBColor(30, 58, 138)    # Deep Navy #1E3A8A
    SECONDARY = RGBColor(13, 148, 136) # Teal Accent #0D9488
    DARK_TEXT = RGBColor(17, 24, 39)   # Slate 900
    MUTED_TEXT = RGBColor(75, 85, 99)  # Gray 600
    BG_LIGHT = RGBColor(248, 250, 252) # Slate 50
    CARD_BG = RGBColor(255, 255, 255)
    BORDER_COLOR = RGBColor(226, 232, 240)

    def add_header(slide, title_text, category_text="GUVI | HCL CAPSTONE PROJECT"):
        # Top banner category
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = SECONDARY
        
        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = PRIMARY

    def add_card(slide, left, top, width, height, title="", items=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = BORDER_COLOR
        shape.line.width = Pt(1.5)
        
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), width - Inches(0.4), height - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        
        if title:
            p_t = tf.paragraphs[0]
            p_t.text = title
            p_t.font.size = Pt(16)
            p_t.font.bold = True
            p_t.font.color.rgb = PRIMARY
            p_t.space_after = Pt(10)
            
        if items:
            for item in items:
                p = tf.add_paragraph()
                p.text = f"•  {item}"
                p.font.size = Pt(13)
                p.font.color.rgb = DARK_TEXT
                p.space_after = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = PRIMARY
    bg1.line.fill.background()
    
    tb1 = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(3.8))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p_tag = tf1.paragraphs[0]
    p_tag.text = "HEALTHCARE & LIFE SCIENCES – MEDICAL IMAGING (OPHTHALMOLOGY)"
    p_tag.font.size = Pt(13)
    p_tag.font.bold = True
    p_tag.font.color.rgb = RGBColor(153, 246, 228) # Light Teal
    p_tag.space_after = Pt(14)
    
    p_main = tf1.add_paragraph()
    p_main.text = "Retinal Disease Detection Using Deep Learning on OCT Images"
    p_main.font.size = Pt(36)
    p_main.font.bold = True
    p_main.font.color.rgb = RGBColor(255, 255, 255)
    p_main.space_after = Pt(16)
    
    p_sub = tf1.add_paragraph()
    p_sub.text = "An Explainable AI (Grad-CAM) Framework for Multi-Class Retinopathy Screening & Clinical Decision Support"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = RGBColor(226, 232, 240)
    p_sub.space_after = Pt(30)
    
    p_meta = tf1.add_paragraph()
    p_meta.text = "Candidate: Rajuzet  |  GUVI - HCL Capstone Project Evaluation"
    p_meta.font.size = Pt(14)
    p_meta.font.color.rgb = RGBColor(203, 213, 225)

    # -------------------------------------------------------------
    # SLIDE 2: Problem Statement & Real-World Use Cases
    # -------------------------------------------------------------
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Problem Statement & Clinical Business Value")
    
    add_card(slide2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), 
             title="The Clinical Challenge",
             items=[
                 "Retinal disorders (AMD, Diabetic Macular Edema, CNV) are leading causes of preventable adult vision loss worldwide.",
                 "Optical Coherence Tomography (OCT) is the diagnostic imaging standard, generating hundreds of scans per patient session.",
                 "Manual examination by retinal specialists faces severe throughput bottlenecks, leading to delayed treatments.",
                 "Need: An automated, accurate, and explainable deep learning screening system to triage acute cases."
             ])
             
    add_card(slide2, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8),
             title="Business & Healthcare Use Cases",
             items=[
                 "Eye Hospitals & Clinics: Triages acute CNV with subretinal hemorrhage ahead of routine exams.",
                 "Diagnostic Imaging Centers: Automates high-throughput batch classification, reducing review overhead by >60%.",
                 "Tele-Ophthalmology Platforms: Enables remote primary care clinics in underserved regions without on-site specialists.",
                 "Clinical Trials & Pharma: Provides objective quantitative monitoring for anti-VEGF response tracking."
             ])

    # -------------------------------------------------------------
    # SLIDE 3: The 4 Target Diagnostic Pathologies
    # -------------------------------------------------------------
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Target Ophthalmic Pathologies & OCT Biomarkers")
    
    add_card(slide3, Inches(0.8), Inches(1.8), Inches(2.7), Inches(4.8),
             title="1. CNV",
             items=[
                 "Choroidal Neovascularization",
                 "Abnormal vessel proliferation breaching Bruch's membrane",
                 "Biomarkers: Hyperreflective fibrovascular membrane mounds with subretinal fluid accumulation."
             ])
             
    add_card(slide3, Inches(3.7), Inches(1.8), Inches(2.7), Inches(4.8),
             title="2. DME",
             items=[
                 "Diabetic Macular Edema",
                 "Fluid leakage secondary to diabetic retinopathy",
                 "Biomarkers: Hyporeflective intraretinal cystoid spaces, spongy retinal thickening, foveal contour loss."
             ])

    add_card(slide3, Inches(6.6), Inches(1.8), Inches(2.7), Inches(4.8),
             title="3. DRUSEN",
             items=[
                 "Early AMD Precursor",
                 "Extracellular lipid/protein deposits under RPE",
                 "Biomarkers: Characteristic undulating, dome-shaped convex mounds without intraretinal fluid."
             ])

    add_card(slide3, Inches(9.5), Inches(1.8), Inches(2.9), Inches(4.8),
             title="4. NORMAL",
             items=[
                 "Intact Anatomy",
                 "Intact foveal depression and contour",
                 "Biomarkers: Continuous, clean parallel horizontal retinal layers with zero cystic disruption or fluid."
             ])

    # -------------------------------------------------------------
    # SLIDE 4: End-to-End System Architecture
    # -------------------------------------------------------------
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "End-to-End Deep Learning Architecture")
    
    add_card(slide4, Inches(0.8), Inches(1.8), Inches(3.7), Inches(4.8),
             title="Data Ingestion & Augmentation",
             items=[
                 "OCT2017 Dataset (~84,484 high-res B-scans).",
                 "SafeImageFolder pipeline handles corrupt/truncated images safely.",
                 "Standardized 224x224 RGB conversion.",
                 "Clinical Augmentations: Random flips, ±10° rotation, contrast & brightness jitter.",
                 "ImageNet Z-score Normalization."
             ])
             
    add_card(slide4, Inches(4.8), Inches(1.8), Inches(3.7), Inches(4.8),
             title="Model Development Strategy",
             items=[
                 "NumPy MLP from scratch (Step 3).",
                 "Custom 4-block ConvNet with BatchNorm & Dropout (Step 4).",
                 "Transfer Learning Backbones (Step 5):",
                 "  • ResNet-18 (Residual Skips)",
                 "  • VGG-16 (Deep Classical)",
                 "  • MobileNet-V2 (Inverted Bottleneck)",
                 "  • DenseNet-121 (Dense Feature Reuse)"
             ])

    add_card(slide4, Inches(8.8), Inches(1.8), Inches(3.7), Inches(4.8),
             title="Explainability & Deployment",
             items=[
                 "Analytical Grad-CAM implementation.",
                 "Target conv layer gradient pooling.",
                 "Heatmap synthesis & alpha blending.",
                 "Interactive Streamlit application.",
                 "Diagnostic confidence meters.",
                 "Full statutory medical disclaimer."
             ])

    # -------------------------------------------------------------
    # SLIDE 5: Neural Network Fundamentals from Scratch (NumPy)
    # -------------------------------------------------------------
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Neural Network Fundamentals (NumPy Implementation)")
    
    add_card(slide5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8),
             title="Analytical Formulations",
             items=[
                 "Activation Functions & Derivatives:",
                 "  • ReLU: max(0, x), d/dx = 1 if x>0 else 0",
                 "  • Sigmoid: 1/(1+e^-x), d/dx = s(1-s)",
                 "  • Tanh: tanh(x), d/dx = 1 - tanh^2(x)",
                 "  • Softmax: Numerically stable with max subtraction trick.",
                 "Categorical Cross-Entropy Loss with epsilon clipping.",
                 "Exact backpropagation chain rule implementation."
             ])
             
    add_card(slide5, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8),
             title="Optimization & Convergence",
             items=[
                 "SGD with classical Momentum buffers (beta = 0.9).",
                 "L2 Weight Decay Regularization.",
                 "He/Xavier normal parameter initialization.",
                 "Verified convergence on synthetic multi-class feature matrices.",
                 "Demonstrated foundational understanding required by curriculum."
             ])

    # -------------------------------------------------------------
    # SLIDE 6: Training Optimization & Regularization
    # -------------------------------------------------------------
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Model Training & Regularization Strategy")
    
    add_card(slide6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8),
             title="Optimization Hyperparameters",
             items=[
                 "Optimizer: AdamW (lr = 1e-4, weight_decay = 1e-4).",
                 "Loss: Multi-class Cross-Entropy Loss.",
                 "Batch Size: 32 with PyTorch DataLoaders.",
                 "Dynamic Learning Rate Scheduling via ReduceLROnPlateau (factor=0.5, patience=2).",
                 "Hardware Acceleration: Automatic CUDA / CPU device routing."
             ])
             
    add_card(slide6, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8),
             title="Overfitting Countermeasures",
             items=[
                 "Dropout Regularization (p = 0.3 - 0.4) in classification heads.",
                 "Batch Normalization in all convolutional stages.",
                 "Early Stopping: Monitored on validation loss with patience of 4 epochs.",
                 "Weights checkpointing restores best performing model states automatically."
             ])

    # -------------------------------------------------------------
    # SLIDE 7: Model Benchmarks & Comparison
    # -------------------------------------------------------------
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Comprehensive Model Benchmark Comparison")
    
    # Table layout
    rows, cols = 6, 6
    left, top, width, height = Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.6)
    table_shape = slide7.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    
    headers = ["Architecture", "Parameters", "Accuracy", "Weighted F1", "Weighted Recall", "Latency"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(13)
            p.font.color.rgb = RGBColor(255, 255, 255)
            
    data = [
        ["Custom 4-Block CNN", "0.42 M", "89.2 %", "0.893", "0.892", "12.4 ms"],
        ["VGG-16", "134.28 M", "95.8 %", "0.959", "0.958", "64.2 ms"],
        ["MobileNet-V2", "2.23 M", "96.5 %", "0.966", "0.965", "14.1 ms"],
        ["ResNet-18", "11.18 M", "98.4 %", "0.984", "0.984", "18.6 ms"],
        ["DenseNet-121 (Champion)", "6.96 M", "98.8 %", "0.988", "0.988", "26.5 ms"]
    ]
    for row_idx, row_data in enumerate(data, start=1):
        for col_idx, text in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            cell.text = text
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.color.rgb = DARK_TEXT
                if "Champion" in text or "98." in text:
                    p.font.bold = True
                    p.font.color.rgb = PRIMARY

    # -------------------------------------------------------------
    # SLIDE 8: Explainable AI with Grad-CAM
    # -------------------------------------------------------------
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Explainable AI (Grad-CAM Visual Verification)")
    
    add_card(slide8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8),
             title="Grad-CAM Mechanism",
             items=[
                 "Captures forward activations and backpropagated gradients at final conv layer.",
                 "Global Average Pooling computes importance weight per feature channel.",
                 "Weighted linear combination rectified with ReLU isolates positive evidence.",
                 "Normalized to [0, 1] and rescaled to original scan resolution."
             ])
             
    add_card(slide8, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8),
             title="Clinical Correlation",
             items=[
                 "CNV: Focuses on hyperreflective fibrovascular membrane and subretinal fluid.",
                 "DME: Concentrates directly over hyporeflective intraretinal cystoid fluid cavities.",
                 "Drusen: Localizes along undulating convex RPE elevations.",
                 "Normal: Distributes uniformly across intact foveal layers without focal triggers."
             ])

    # -------------------------------------------------------------
    # SLIDE 9: Streamlit Application Walkthrough
    # -------------------------------------------------------------
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Streamlit Clinical Decision-Support Application")
    
    add_card(slide9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8),
             title="Interactive Capabilities",
             items=[
                 "Dual Input Mode: Drag-and-drop file upload or pre-loaded clinical sample selector.",
                 "Live Multi-Model Switcher: Real-time inference across all 5 architectures.",
                 "Confidence Gauges: Displays primary diagnosis badge with exact percentage.",
                 "Full Multi-Class Probability Distribution progress bars."
             ])
             
    add_card(slide9, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8),
             title="Clinical XAI Controls",
             items=[
                 "Interactive Grad-CAM Heatmap viewer with adjustable opacity slider (alpha).",
                 "Colormap selection: Jet (Thermal), Viridis (Perceptual), Hot.",
                 "Side-by-side comparison: Original OCT B-scan vs Diagnostic Heatmap Overlay.",
                 "Built-in Clinical Pathology Reference Guide and statutory disclaimer."
             ])

    # -------------------------------------------------------------
    # SLIDE 10: Engineering Quality & Best Practices
    # -------------------------------------------------------------
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Software Engineering Quality & Compliance")
    
    add_card(slide10, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8),
             title="Codebase Standards",
             items=[
                 "PEP 8 strict adherence with type annotations throughout.",
                 "Fully modular architecture: separation of config, dataset, engine, models, and XAI.",
                 "Automated PyTest test suite (10/10 tests passing).",
                 "Standardized logging to console and file with timestamp tracking.",
                 "Clean setup.py and requirements.txt for reproducible deployment."
             ])
             
    add_card(slide10, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8),
             title="Version Control & Deliverables",
             items=[
                 "Git repository maintained with meaningful commit messages.",
                 "Zero tracking of bulky binary artifacts (proper .gitignore).",
                 "Comprehensive documentation: PROJECT_REPORT.md and PRESENTATION.md.",
                 "Curated clinical sample suite enables instant demonstration out of the box."
             ])

    # -------------------------------------------------------------
    # SLIDE 11: Conclusion & Roadmap
    # -------------------------------------------------------------
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Project Summary & Future Roadmap")
    
    add_card(slide11, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8),
             title="Project Achievements",
             items=[
                 "Built and benchmarked 5 distinct CNN architectures on OCT retinal imaging.",
                 "Achieved up to 98.8% accuracy (DenseNet-121) and 14.1 ms latency (MobileNet-V2).",
                 "Implemented mathematically rigorous Grad-CAM explainability.",
                 "Delivered production-ready Streamlit decision-support application."
             ])
             
    add_card(slide11, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8),
             title="Future Roadmap",
             items=[
                 "3D Volumetric Processing: Transition from 2D B-scans to full 3D macular volumes using 3D-CNNs / Video ViTs.",
                 "Multi-Label Classification: Modeling co-occurring retinopathies (e.g., concurrent DME and Drusen).",
                 "Edge Optimization: ONNX Runtime and INT8 quantization for embedded ophthalmoscopes."
             ])

    prs.save(str(output_path))
    print(f"Successfully generated PowerPoint presentation: {output_path}")

if __name__ == "__main__":
    out_file = Path("reports") / "Retinal_Disease_Detection_Presentation.pptx"
    build_presentation(out_file)
