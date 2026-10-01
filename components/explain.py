"""
NeuroStage Explainability Component
=====================================
Grad-CAM heatmap generation and overlay for interpretable predictions.
Hooks into the final convolutional block of EfficientNet-B0 to produce
a 224×224 attention heatmap highlighting brain regions the model focuses on.
"""

from typing import Tuple, Optional

import numpy as np
from PIL import Image
import torch
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model.architecture import AlzheimerStagingModel
from utils.preprocess import get_val_transforms, grayscale_to_rgb, min_max_normalize


def get_target_layer(model: AlzheimerStagingModel):
    """
    Get the target layer for Grad-CAM from EfficientNet-B0.
    We hook the final convolutional block — this captures high-level
    features like atrophy patterns, ventricle enlargement, and cortical
    thinning.
    """
    # EfficientNet-B0 in timm: the last feature block
    # backbone.blocks[-1] is the final MBConv block
    return [model.backbone.blocks[-1]]


def create_brain_mask(gray_image: np.ndarray, threshold: int = 15) -> np.ndarray:
    """
    Create a binary mask of the brain region by thresholding the
    grayscale image. Pixels above the threshold are considered brain
    tissue; the dark background is masked out.

    Args:
        gray_image: Grayscale image array (H, W), uint8
        threshold: Intensity threshold to separate brain from background

    Returns:
        Binary mask (H, W), float32, values 0.0 or 1.0
    """
    # Threshold to separate brain from dark background
    _, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find the largest contour (the brain) and fill it
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Keep only the largest contour (the brain)
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
    else:
        mask = binary

    # Apply slight Gaussian blur to soften mask edges for a smooth overlay
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    return (mask / 255.0).astype(np.float32)


def generate_gradcam(
    model: AlzheimerStagingModel,
    input_tensor: torch.Tensor,
    original_rgb: np.ndarray,
    predicted_class: int,
    device: torch.device,
    image_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM heatmap for the predicted class, masked to
    brain tissue only.

    Args:
        model: Trained AlzheimerStagingModel
        input_tensor: Preprocessed input tensor (1, 3, 224, 224)
        original_rgb: Original RGB image array (H, W, 3), uint8
        predicted_class: Predicted stage (0–3)
        device: torch device
        image_size: Target size for heatmap

    Returns:
        Tuple of (heatmap_colored, overlay, original_resized):
        - heatmap_colored: Standalone heatmap as RGB (H, W, 3), uint8
        - overlay: Blended original + heatmap (H, W, 3), uint8
        - original_resized: Original image resized to match (H, W, 3), uint8
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    target_layers = get_target_layer(model)
    targets = [ClassifierOutputTarget(predicted_class)]

    # Generate Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # (H, W) float [0, 1]

    # Resize original to match heatmap size
    original_resized = cv2.resize(
        original_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR
    )

    # Create brain mask to suppress background attention
    gray_resized = cv2.cvtColor(original_resized, cv2.COLOR_RGB2GRAY)
    brain_mask = create_brain_mask(gray_resized)

    # Apply mask to Grad-CAM — zero out attention on background
    grayscale_cam = grayscale_cam * brain_mask

    # Re-normalize the masked cam to use full [0, 1] range
    cam_max = grayscale_cam.max()
    if cam_max > 0:
        grayscale_cam = grayscale_cam / cam_max

    # Normalize original to [0, 1] float for overlay
    original_float = original_resized.astype(np.float32) / 255.0

    # Create colored heatmap (jet colormap)
    heatmap_colored = cv2.applyColorMap(
        (grayscale_cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Mask the heatmap so background stays dark
    mask_3ch = np.stack([brain_mask] * 3, axis=-1)
    heatmap_colored = (heatmap_colored * mask_3ch).astype(np.uint8)

    # Create overlay (blend original + heatmap)
    overlay = show_cam_on_image(
        original_float, grayscale_cam, use_rgb=True, image_weight=0.5
    )

    return heatmap_colored, overlay, original_resized


def generate_gradcam_from_pil(
    model: AlzheimerStagingModel,
    pil_image: Image.Image,
    predicted_class: int,
    device: torch.device,
    image_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper: generate Grad-CAM directly from a PIL image.

    Returns:
        (heatmap_colored, overlay, original_resized) — all uint8 RGB arrays
    """
    # Preprocess
    gray = pil_image.convert("L")
    gray_array = np.array(gray, dtype=np.uint8)
    normalized = min_max_normalize(gray_array)
    rgb_array = grayscale_to_rgb(normalized)

    # Create tensor
    transform = get_val_transforms(image_size)
    augmented = transform(image=rgb_array)
    input_tensor = augmented["image"].unsqueeze(0)

    return generate_gradcam(
        model, input_tensor, rgb_array, predicted_class, device, image_size
    )
