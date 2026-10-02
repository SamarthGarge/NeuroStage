"""
NeuroStage Report Component
=============================
Results dashboard rendering: probability distribution chart,
triage banner, Grad-CAM visualization, and metadata display.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from utils.constants import (
    STAGE_LABELS,
    STAGE_COLORS,
    STAGE_SHORT_LABELS,
    ANATOMY_GUIDE,
)


def render_probability_chart(probabilities: dict):
    """
    Render a horizontal bar chart showing the probability distribution
    across all 4 dementia stages.

    Args:
        probabilities: Dict of {stage_label: probability}
    """
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [STAGE_COLORS[i] for i in range(len(labels))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=[v * 100 for v in values],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
            opacity=0.85,
        ),
        text=[f"{v*100:.1f}%" for v in values],
        textposition="auto",
        textfont=dict(color="white", size=13, family="Inter"),
    ))

    fig.update_layout(
        title=dict(
            text="Probability distribution",
            font=dict(size=16, color="#c9d1d9"),
        ),
        xaxis=dict(
            title="Probability (%)",
            range=[0, 105],
            showgrid=True,
            gridcolor="#21262d",
            color="#8b949e",
        ),
        yaxis=dict(
            autorange="reversed",
            color="#c9d1d9",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        height=220,
        margin=dict(l=10, r=20, t=45, b=20),
    )

    st.plotly_chart(fig)


def render_triage_banner(triage: dict, low_confidence: bool):
    """
    Render the clinical triage recommendation banner.

    Uses Streamlit's native alert components:
    - st.success for stage 0
    - st.warning for stages 1–2
    - st.error for stage 3 (urgent)
    """
    if low_confidence:
        st.warning(
            "**Low confidence warning** — Model confidence is below 60%. "
            "This prediction may be unreliable. Consider specialist review "
            "regardless of the predicted stage.",
            icon=":material/warning:",
        )

    level = triage["level"]
    action = triage["action"]
    message = triage["message"]

    full_message = f"**{action}**\n\n{message}"

    if level == "success":
        st.success(full_message, icon=":material/check_circle:")
    elif level == "warning":
        st.warning(full_message, icon=":material/warning:")
    elif level == "error":
        st.error(full_message, icon=":material/emergency:")


def render_gradcam_views(
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
):
    """
    Render side-by-side Grad-CAM visualization with tabs:
    Original | Heatmap | Blended overlay

    Args:
        original: Original image array (H, W, 3), uint8
        heatmap: Grad-CAM heatmap array (H, W, 3), uint8
        overlay: Blended overlay array (H, W, 3), uint8
    """
    st.subheader("Grad-CAM attention map", icon=":material/biotech:")
    st.caption("Warmer colors (red/yellow) = higher model attention")

    view_mode = st.segmented_control(
        "View",
        ["Original", "Heatmap", "Overlay"],
        default="Overlay",
        label_visibility="collapsed",
        key="gradcam_view",
    )

    if view_mode == "Original":
        st.image(original, caption="Uploaded MRI scan", width="stretch")
    elif view_mode == "Heatmap":
        st.image(heatmap, caption="Grad-CAM heatmap", width="stretch")
    else:
        st.image(overlay, caption="Blended overlay (50/50)", width="stretch")


def render_anatomy_guide(predicted_stage: int):
    """
    Display the neuroanatomical interpretation guide — maps Grad-CAM
    hotspots to clinical meaning based on the predicted stage.
    """
    with st.expander(
        "Anatomy guide — What is the model looking at?",
        icon=":material/science:",
        expanded=False,
    ):
        st.markdown(
            "The model focuses on specific brain regions whose atrophy patterns "
            "correlate with each dementia stage:"
        )

        relevant_regions = []
        for region, info in ANATOMY_GUIDE.items():
            if predicted_stage in info["stages"]:
                relevant_regions.append((region, info))

        if relevant_regions:
            for region_name, info in relevant_regions:
                st.markdown(
                    f"**:material/location_on: {region_name.title()}** "
                    f"({info['location']})\n\n"
                    f"{info['meaning']}"
                )
        else:
            st.info(
                "For non-demented scans (Stage 0), the model should show "
                "relatively uniform, low activation — no specific atrophy "
                "regions are highlighted.",
                icon=":material/info:",
            )


def render_metadata(inference_time_ms: float, image_size: tuple):
    """Render inference metadata."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Inference time", f"{inference_time_ms:.0f} ms")
    with col2:
        st.metric("Input size", f"{image_size[0]}×{image_size[1]}")
    with col3:
        st.metric("Model", "EfficientNet-B0")


def render_disclaimer():
    """Render the mandatory medical disclaimer footer."""
    st.warning(
        "**Educational & research use only** — This tool is NOT a medical device "
        "and should NOT be used for clinical diagnosis. Single-slice screening only; "
        "full volumetric clinical review is required. Always consult a qualified "
        "healthcare professional for medical decisions.\n\n"
        "*Dataset derived from OASIS-1. Model: EfficientNet-B0 (ImageNet transfer learning).*",
        icon=":material/gavel:",
    )
