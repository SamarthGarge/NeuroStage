"""
NeuroStage Upload Component
=============================
File uploader with quality gate validation for brain MRI scans.
Checks for valid image format, size, aspect ratio, and basic
brain-scan heuristics before passing to inference.
"""

from typing import Tuple, Optional

import numpy as np
from PIL import Image
import streamlit as st

from utils.constants import QUALITY_GATE


def validate_upload(uploaded_file) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Quality gate: validate the uploaded file is a plausible brain MRI scan.

    Checks:
    1. File is a valid image (PIL can open it)
    2. File size ≤ 10 MB
    3. Aspect ratio between 0.4–2.5
    4. Minimum resolution ≥ 64×64
    5. Mean intensity within plausible range (not blank/saturated)

    Returns:
        (is_valid, message, pil_image_or_none)
    """
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > QUALITY_GATE["max_file_size_mb"]:
        return (
            False,
            f"File too large ({file_size_mb:.1f} MB). "
            f"Maximum allowed: {QUALITY_GATE['max_file_size_mb']} MB.",
            None,
        )

    # Try to open as image
    try:
        image = Image.open(uploaded_file)
        image.load()  # force full load to catch corrupt files
    except Exception as e:
        return (
            False,
            f"Cannot open file as an image. Please upload a valid "
            f"PNG or JPG brain MRI scan. Error: {str(e)}",
            None,
        )

    # Check dimensions
    width, height = image.size
    if width < QUALITY_GATE["min_size_pixels"] or height < QUALITY_GATE["min_size_pixels"]:
        return (
            False,
            f"Image too small ({width}×{height}). "
            f"Minimum: {QUALITY_GATE['min_size_pixels']}×"
            f"{QUALITY_GATE['min_size_pixels']} pixels.",
            None,
        )

    # Check aspect ratio
    aspect_ratio = width / height
    if (aspect_ratio < QUALITY_GATE["min_aspect_ratio"]
            or aspect_ratio > QUALITY_GATE["max_aspect_ratio"]):
        return (
            False,
            f"Unusual aspect ratio ({aspect_ratio:.2f}). Brain MRI axial "
            f"slices should have an aspect ratio between "
            f"{QUALITY_GATE['min_aspect_ratio']}–{QUALITY_GATE['max_aspect_ratio']}.",
            None,
        )

    # Check intensity (grayscale heuristic)
    gray = np.array(image.convert("L"), dtype=np.float32)
    mean_intensity = gray.mean()

    if mean_intensity < QUALITY_GATE["min_mean_intensity"]:
        return (
            False,
            "Image appears nearly black. Please upload a properly "
            "exposed brain MRI scan.",
            None,
        )

    if mean_intensity > QUALITY_GATE["max_mean_intensity"]:
        return (
            False,
            "Image appears nearly white/blank. Please upload a valid "
            "brain MRI scan.",
            None,
        )

    # All checks passed
    return (
        True,
        f"Valid brain MRI scan ({width}×{height}, {file_size_mb:.1f} MB)",
        image,
    )


def render_uploader() -> Optional[Image.Image]:
    """
    Render the Streamlit file uploader with quality gate validation.

    Returns:
        PIL Image if valid upload, None otherwise
    """
    st.subheader("Upload brain MRI scan", icon=":material/upload:")
    st.caption(
        "Upload an axial (top-down) T1-weighted brain MRI slice in PNG or JPG format. "
        "Maximum file size: 10 MB."
    )

    uploaded_file = st.file_uploader(
        "Choose an MRI scan",
        type=["png", "jpg", "jpeg"],
        help="Axial brain MRI slice (PNG/JPG, max 10 MB)",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        return None

    # Run quality gate
    is_valid, message, image = validate_upload(uploaded_file)

    if not is_valid:
        st.error(f"{message}", icon=":material/error:")
        st.info(
            "**Tips for a good upload:**\n"
            "- Use an axial (top-down) brain MRI slice\n"
            "- Grayscale PNG or JPG format\n"
            "- Roughly square aspect ratio\n"
            "- Clear, well-exposed image",
            icon=":material/lightbulb:",
        )
        return None

    st.success(message, icon=":material/check_circle:")

    # Show scan metadata
    width, height = image.size
    file_size = uploaded_file.size / 1024  # KB
    st.caption(f":material/straighten: Dimensions: {width}×{height} | Size: {file_size:.0f} KB")

    return image
