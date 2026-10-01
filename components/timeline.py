"""
NeuroStage Timeline Component
===============================
Custom HTML/CSS stage progression stepper for the Streamlit dashboard.
Renders a visual 4-stage timeline showing the patient's current position
on the Alzheimer's Disease staging scale.
"""

import streamlit as st
from utils.constants import STAGE_LABELS, STAGE_COLORS, STAGE_EMOJIS


def render_timeline(predicted_stage: int, confidence: float):
    """
    Render a visual stage progression timeline.

    Shows 4 connected nodes (stages 0–3) with:
    - Active node highlighted and enlarged
    - Progress bar colored up to the current stage
    - Stage labels below each node
    - Confidence displayed at the active node

    Args:
        predicted_stage: Predicted stage (0–3)
        confidence: Confidence score (0–1)
    """
    active_color = STAGE_COLORS[predicted_stage]

    # Build node HTML for each stage
    nodes_html = ""
    for stage_id in range(4):
        is_active = stage_id == predicted_stage
        is_passed = stage_id < predicted_stage
        label = STAGE_LABELS[stage_id]
        color = STAGE_COLORS[stage_id]

        if is_active:
            node_style = f"background: {color}; border-color: {color}; transform: scale(1.3); box-shadow: 0 0 20px {color}66;"
            label_style = f"color: {color}; font-weight: 700;"
        elif is_passed:
            node_style = f"background: {color}; border-color: {color}; opacity: 0.7;"
            label_style = f"color: {color}; opacity: 0.7;"
        else:
            node_style = "background: #2a2a3e; border-color: #444466;"
            label_style = "color: #666688;"

        confidence_badge = ""
        if is_active:
            confidence_badge = f'<div class="ns-conf-badge">{confidence*100:.1f}%</div>'

        nodes_html += f"""
        <div class="ns-tl-node">
            <div class="ns-tl-circle" style="{node_style}">
                <span class="ns-tl-num">{stage_id}</span>
            </div>
            {confidence_badge}
            <div class="ns-tl-label" style="{label_style}">
                {label.replace(" ", "<br>")}
            </div>
        </div>
        """

        # Add connector line between nodes (except after last)
        if stage_id < 3:
            if stage_id < predicted_stage:
                connector_style = f"background: linear-gradient(90deg, {STAGE_COLORS[stage_id]}, {STAGE_COLORS[stage_id+1]});"
            elif stage_id == predicted_stage:
                connector_style = f"background: linear-gradient(90deg, {color}88, #2a2a3e);"
            else:
                connector_style = "background: #2a2a3e;"

            nodes_html += f'<div class="ns-tl-conn" style="{connector_style}"></div>'

    # Assemble full timeline HTML — using st.html since no native equivalent
    timeline_html = f"""
    <style>
        .ns-tl-wrap {{
            display: flex;
            align-items: flex-start;
            justify-content: center;
            padding: 30px 10px 20px 10px;
            margin: 8px 0;
        }}
        .ns-tl-node {{
            display: flex;
            flex-direction: column;
            align-items: center;
            position: relative;
            min-width: 80px;
        }}
        .ns-tl-circle {{
            width: 44px;
            height: 44px;
            border-radius: 50%;
            border: 3px solid;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
            z-index: 2;
        }}
        .ns-tl-num {{
            color: white;
            font-size: 16px;
            font-weight: 700;
            font-family: Inter, sans-serif;
        }}
        .ns-tl-label {{
            margin-top: 10px;
            font-size: 11px;
            text-align: center;
            line-height: 1.3;
            max-width: 90px;
            font-family: Inter, sans-serif;
        }}
        .ns-conf-badge {{
            margin-top: 6px;
            background: {active_color}22;
            border: 1px solid {active_color}66;
            color: {active_color};
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            font-family: Inter, sans-serif;
        }}
        .ns-tl-conn {{
            height: 3px;
            flex: 1;
            min-width: 40px;
            margin-top: 22px;
            border-radius: 2px;
        }}
    </style>
    <div class="ns-tl-wrap">
        {nodes_html}
    </div>
    """

    st.html(timeline_html)


def render_stage_card(predicted_stage: int, confidence: float, description: dict):
    """
    Render a styled stage information card with clinical and plain-language
    descriptions using native Streamlit elements.

    Args:
        predicted_stage: Predicted stage (0–3)
        confidence: Confidence score (0–1)
        description: Dict with 'clinical' and 'plain' keys
    """
    label = STAGE_LABELS[predicted_stage]

    # Map stage to badge color
    badge_colors = {0: "green", 1: "yellow", 2: "orange", 3: "red"}
    badge_color = badge_colors[predicted_stage]

    with st.container(border=True):
        col_title, col_conf = st.columns([3, 1])
        with col_title:
            st.subheader(
                f"Stage {predicted_stage} — {label}",
                icon=":material/clinical_notes:",
            )
        with col_conf:
            st.metric("Confidence", f"{confidence*100:.1f}%")

        st.badge(f"Stage {predicted_stage}", color=badge_color)
        st.markdown(description["clinical"])
        st.caption(f":material/chat: **In plain language:** {description['plain']}")
