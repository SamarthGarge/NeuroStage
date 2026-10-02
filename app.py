"""
NeuroStage — Alzheimer's MRI Staging Web Application
=====================================================
AI-powered web application that analyzes axial T1-weighted brain MRI scans
and classifies the patient's dementia stage on a 4-point clinical scale.

Usage:
    streamlit run app.py
"""

import json

import streamlit as st
from streamlit_lottie import st_lottie

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
# Load Lottie animation (cached)
# ──────────────────────────────────────────────────────────────────────

@st.cache_data
def load_lottie_brain():
    """Load the brain Lottie animation from assets."""
    with open("assets/Brain.json", "r") as f:
        return json.load(f)


lottie_brain = load_lottie_brain()


# ──────────────────────────────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────────────────────────────

st.session_state.setdefault("analysis_result", None)
st.session_state.setdefault("analysis_image", None)
st.session_state.setdefault("gradcam_data", None)


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Lottie brain animation at the top
    st_lottie(lottie_brain, height=180, key="brain_lottie")

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
        badge_colors = {0: "green", 1: "yellow", 2: "orange", 3: "red"}
        for stage_id, label in STAGE_LABELS.items():
            st.badge(f"Stage {stage_id} — {label}", color=badge_colors[stage_id])
            st.caption(STAGE_DESCRIPTIONS[stage_id]["plain"])

    with st.expander("How it works", icon=":material/science:"):
        st.markdown("""
1. **Upload** an axial brain MRI slice
2. **Quality gate** validates it's a brain scan
3. **Preprocessing** — grayscale → RGB, resize to 224², normalize
4. **Inference** — EfficientNet-B0 classifies into 4 stages
5. **Grad-CAM** — highlights regions driving the prediction
6. **Triage** — recommends next clinical steps
        """)

    with st.expander("Technical details", icon=":material/settings:"):
        st.markdown("""
- **Model** — EfficientNet-B0 (~5.7M params)
- **Input** — 3 × 224 × 224 (grayscale → RGB)
- **Training** — AdamW, cosine scheduler, class-weighted CE
- **Imbalance** — WeightedRandomSampler + class weights
- **Explainability** — Grad-CAM on final conv block
- **Privacy** — Zero data persistence (in-memory only)
        """)



# ──────────────────────────────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────────────────────────────

st.title("NeuroStage", icon=":material/neurology:")
st.caption("AI-powered Alzheimer's disease staging from brain MRI scans")


# ── Upload section ──

image = render_uploader()

if image is not None:
    # Show uploaded image and analyze button in a compact row
    col_preview, col_action = st.columns([1, 2], vertical_alignment="center")

    with col_preview:
        st.image(image, caption="Uploaded MRI scan", width="stretch")

    with col_action:
        with st.container(border=True):
            st.subheader("Ready to analyze", icon=":material/biotech:")
            st.markdown(
                "The scan has passed the quality gate and is ready for "
                "AI-powered staging analysis."
            )
            analyze_clicked = st.button(
                "Analyze scan",
                icon=":material/play_arrow:",
                type="primary",
                key="analyze_btn",
            )

    # Run analysis
    if analyze_clicked:
        with st.status(
            "Running analysis...", expanded=True, state="running"
        ) as status:
            st.write("Loading model...")
            model, device = load_model()

            st.write("Preprocessing image...")
            input_tensor, rgb_array = preprocess_image(image)

            st.write("Running inference...")
            result = predict(model, input_tensor, device)

            st.write("Generating Grad-CAM heatmap...")
            heatmap, overlay, original_resized = generate_gradcam_from_pil(
                model, image, result["stage"], device
            )

            # Store in session state
            st.session_state["analysis_result"] = result
            st.session_state["analysis_image"] = image
            st.session_state["gradcam_data"] = (heatmap, overlay, original_resized)

            status.update(label="Analysis complete", state="complete", expanded=False)

    # ── Results dashboard ──

    result = st.session_state.get("analysis_result")
    gradcam_data = st.session_state.get("gradcam_data")

    if result is not None and gradcam_data is not None:
        st.header("Results", icon=":material/analytics:")

        # 1. Stage timeline
        render_timeline(result["stage"], result["confidence"])

        # 2. Stage card + triage side by side
        col_stage, col_triage = st.columns([3, 2], gap="medium")

        with col_stage:
            render_stage_card(
                result["stage"],
                result["confidence"],
                result["description"],
            )

        with col_triage:
            with st.container(border=True, height="stretch"):
                st.subheader("Triage recommendation", icon=":material/emergency:")
                render_triage_banner(result["triage"], result["low_confidence"])

        # 3. Probability distribution + Grad-CAM side by side
        col_chart, col_gradcam = st.columns(2, gap="medium")

        with col_chart:
            with st.container(border=True, height="stretch"):
                render_probability_chart(result["probabilities"])

        with col_gradcam:
            with st.container(border=True, height="stretch"):
                heatmap, overlay, original_resized = gradcam_data
                render_gradcam_views(original_resized, heatmap, overlay)

        # 4. Anatomy guide + metadata
        render_anatomy_guide(result["stage"])

        render_metadata(result["inference_time_ms"], (224, 224))

else:
    # ── Welcome state (no image uploaded) ──

    st.space("medium")

    with st.container(horizontal_alignment="center"):
        with st.container(border=True):
            st.subheader("Welcome to NeuroStage", icon=":material/neurology:")
            st.markdown(
                "Upload an axial brain MRI scan above to receive an AI-powered "
                "dementia staging assessment with explainable Grad-CAM "
                "attention maps and clinical triage recommendations."
            )

            with st.container(horizontal=True, horizontal_alignment="center"):
                st.badge("Stage 0: Non-demented", color="green")
                st.badge("Stage 1: Very mild", color="yellow")
                st.badge("Stage 2: Mild", color="orange")
                st.badge("Stage 3: Moderate", color="red")


# ── Disclaimer footer ──
st.space("medium")
render_disclaimer()
