"""
NeuroStage — Alzheimer's MRI Staging Web Application
=====================================================
AI-powered web application that analyzes axial T1-weighted brain MRI scans
and classifies the patient's dementia stage on a 4-point clinical scale.

Features:
- Upload & quality gate for brain MRI scans
- EfficientNet-B0 inference with cached model loading
- Stage label + plain-language description
- Confidence score + probability distribution
- Grad-CAM heatmap highlighting atrophy-affected regions
- Progression timeline stepper
- Triage recommendation banners
- Neuroanatomical interpretation guide
- Inference latency & scan metadata

Usage:
    streamlit run app.py
"""

import time

import streamlit as st
import yaml

from components.uploader import render_uploader
from components.inference import load_model, preprocess_image, predict
from components.explain import generate_gradcam_from_pil
from components.timeline import render_timeline, render_stage_card
from components.report import (
    render_probability_chart,
    render_triage_banner,
    render_gradcam_views,
    render_anatomy_guide,
    render_metadata,
    render_disclaimer,
)
from utils.constants import STAGE_LABELS, STAGE_DESCRIPTIONS, STAGE_COLORS


# ──────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NeuroStage — Alzheimer's MRI staging",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("NeuroStage", icon=":material/neurology:")
    st.caption("Alzheimer's disease staging from MRI")

    with st.container(border=True):
        st.subheader("About", icon=":material/info:")
        st.markdown(
            "NeuroStage uses a fine-tuned **EfficientNet-B0** CNN "
            "to classify axial brain MRI slices into 4 dementia stages "
            "based on the Clinical Dementia Rating (CDR) scale."
        )

    with st.expander("Staging scale", icon=":material/bar_chart:"):
        for stage_id, label in STAGE_LABELS.items():
            color = STAGE_COLORS[stage_id]
            st.markdown(
                f":{color.replace('#', '')}[**Stage {stage_id}**] — {label}"
            )
            st.caption(STAGE_DESCRIPTIONS[stage_id]["plain"])

    with st.expander("How it works", icon=":material/science:"):
        st.markdown("""
        1. **Upload** an axial brain MRI slice
        2. **Quality gate** validates it's a brain scan
        3. **Preprocessing**: grayscale→RGB, resize to 224², normalize
        4. **Inference**: EfficientNet-B0 classifies into 4 stages
        5. **Grad-CAM**: highlights regions driving the prediction
        6. **Triage**: recommends next clinical steps
        """)

    with st.expander("Technical details", icon=":material/settings:"):
        st.markdown("""
        - **Model**: EfficientNet-B0 (~5.7M params)
        - **Input**: 3×224×224 (grayscale→RGB)
        - **Training**: AdamW, cosine scheduler, class-weighted CE
        - **Imbalance**: WeightedRandomSampler + class weights
        - **Explainability**: Grad-CAM on final conv block
        - **Privacy**: Zero data persistence (in-memory only)
        """)

    st.caption("Made with :heart: for neuroscience research")


# ──────────────────────────────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────────────────────────────

# Header
st.title("NeuroStage", icon=":material/neurology:")
st.caption("AI-powered Alzheimer's disease staging from brain MRI scans")

# Layout: left column for upload, right column for results
col_upload, col_results = st.columns([1, 2], gap="large")

with col_upload:
    # Upload section
    image = render_uploader()

    if image is not None:
        st.image(image, caption="Uploaded MRI scan", width="stretch")

        # Analyze button
        analyze_clicked = st.button(
            "Analyze scan",
            icon=":material/biotech:",
            use_container_width=True,
            type="primary",
        )
    else:
        analyze_clicked = False

        # Show placeholder when no image is uploaded
        with st.container(border=True, horizontal_alignment="center"):
            st.space("medium")
            st.markdown(":material/neurology:")
            st.caption("Upload a brain MRI scan to get started")
            st.space("medium")

with col_results:
    if image is not None and analyze_clicked:
        with st.spinner("Loading model and running inference..."):
            # Load cached model
            model, device = load_model()

            # Preprocess
            input_tensor, rgb_array = preprocess_image(image)

            # Predict
            result = predict(model, input_tensor, device)

        # ── Results dashboard ──

        # 1. Stage timeline
        st.subheader("Disease progression timeline", icon=":material/timeline:")
        render_timeline(result["stage"], result["confidence"])

        # 2. Stage card
        render_stage_card(
            result["stage"],
            result["confidence"],
            result["description"],
        )

        # 3. Triage banner
        render_triage_banner(result["triage"], result["low_confidence"])

        # 4. Probability distribution
        render_probability_chart(result["probabilities"])

        # 5. Grad-CAM visualization
        with st.spinner("Generating Grad-CAM heatmap..."):
            heatmap, overlay, original_resized = generate_gradcam_from_pil(
                model, image, result["stage"], device
            )
        render_gradcam_views(original_resized, heatmap, overlay)

        # 6. Anatomy guide
        render_anatomy_guide(result["stage"])

        # 7. Metadata
        st.space("medium")
        render_metadata(
            result["inference_time_ms"],
            (224, 224),
        )

    elif image is not None and not analyze_clicked:
        st.info(
            "Click **Analyze scan** to run the AI staging assessment.",
            icon=":material/touch_app:",
        )

    else:
        # Welcome message when no image
        with st.container(border=True, horizontal_alignment="center"):
            st.space("large")
            st.subheader("Welcome to NeuroStage", icon=":material/neurology:")
            st.markdown(
                "Upload an axial brain MRI scan to receive an AI-powered "
                "dementia staging assessment with explainable Grad-CAM "
                "attention maps and clinical triage recommendations."
            )
            with st.container(horizontal=True, horizontal_alignment="center"):
                st.badge("Stage 0: Non-demented", color="green")
                st.badge("Stage 1: Very mild", color="yellow")
                st.badge("Stage 2: Mild", color="orange")
                st.badge("Stage 3: Moderate", color="red")
            st.space("large")

# ── Disclaimer footer ──
render_disclaimer()
