"""
NeuroStage Inference Component
===============================
Model loading (cached) and prediction wrapper for the Streamlit app.
Handles loading the trained EfficientNet-B0, preprocessing uploaded images,
and returning stage predictions with confidence scores.
"""

import time
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import streamlit as st
import yaml

from model.architecture import AlzheimerStagingModel, build_model
from utils.preprocess import get_val_transforms, grayscale_to_rgb, min_max_normalize
from utils.constants import STAGE_LABELS, STAGE_DESCRIPTIONS, TRIAGE_RULES


@st.cache_resource
def load_model(weights_path: str = "model/best_model.pt") -> Tuple[AlzheimerStagingModel, torch.device]:
    """
    Load the trained model weights. Cached with @st.cache_resource so it's
    loaded only once per Streamlit session (singleton pattern).

    Returns:
        (model, device) tuple
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(weights_path).exists():
        st.error(
            f"❌ Model weights not found at `{weights_path}`. "
            "Please train the model first with `python train.py`."
        )
        st.stop()

    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", None)

    # Build model from saved config or defaults
    if config:
        model = build_model(config)
    else:
        model = AlzheimerStagingModel()

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


def preprocess_image(
    image: Image.Image, image_size: int = 224
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess a PIL image for model inference.

    Steps:
    1. Convert to grayscale (force single-channel)
    2. Per-image min-max normalization
    3. Grayscale → RGB (3-channel replication)
    4. Resize to 224×224 + ImageNet normalization

    Returns:
        (input_tensor, original_rgb_array) — tensor for model and RGB array
        for Grad-CAM overlay
    """
    # Force grayscale
    gray = image.convert("L")
    gray_array = np.array(gray, dtype=np.uint8)

    # Per-image min-max normalization
    normalized = min_max_normalize(gray_array)

    # Grayscale → RGB
    rgb_array = grayscale_to_rgb(normalized)

    # Apply inference transforms
    transform = get_val_transforms(image_size)
    augmented = transform(image=rgb_array)
    input_tensor = augmented["image"].unsqueeze(0)  # add batch dim

    return input_tensor, rgb_array


@torch.no_grad()
def predict(
    model: AlzheimerStagingModel,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> Dict:
    """
    Run inference and return prediction results.

    Returns dictionary with:
    - stage: predicted stage (0–3)
    - label: human-readable stage label
    - confidence: max probability (0–1)
    - probabilities: dict of {stage_label: probability} for all 4 stages
    - logits: raw logit values
    - triage: triage recommendation dict
    - description: clinical + plain-language description
    - low_confidence: bool flag if confidence < 60%
    """
    start_time = time.perf_counter()

    input_tensor = input_tensor.to(device)
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    inference_time = time.perf_counter() - start_time

    predicted_stage = int(probs.argmax())
    confidence = float(probs[predicted_stage])

    # Build probability distribution
    probabilities = {}
    for stage_id, label in STAGE_LABELS.items():
        probabilities[label] = float(probs[stage_id])

    result = {
        "stage": predicted_stage,
        "label": STAGE_LABELS[predicted_stage],
        "confidence": confidence,
        "probabilities": probabilities,
        "probs_array": probs,
        "logits": logits.squeeze(0).cpu().numpy(),
        "triage": TRIAGE_RULES[predicted_stage],
        "description": STAGE_DESCRIPTIONS[predicted_stage],
        "low_confidence": confidence < 0.60,
        "inference_time_ms": inference_time * 1000,
    }

    return result
