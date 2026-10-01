"""
NeuroStage Preprocessing
========================
MRI-specific image transforms for training (with augmentation) and
inference (deterministic). All transforms are anatomy-preserving —
no elastic deformations or heavy crops that could erase atrophy signals.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.constants import IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Training transforms with anatomy-preserving augmentation.

    Augmentation rationale (from TRD §3.2):
    - Horizontal flip: brain is roughly symmetric; safe augmentation
    - Small rotation (±10°): mimics slight head tilt during scanning
    - Affine shear (±5°): minor deformation within safe bounds
    - Gamma shift: simulates scanner intensity variation
    - Gaussian noise: models sensor noise at acquisition
    - NO elastic deformation or heavy crops — these destroy atrophy patterns
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=0, p=0.5),
        A.Affine(shear=(-5, 5), mode=0, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3,
        ),
        A.GaussNoise(std_range=(0.005, 0.01), p=0.2),
        # Normalize: per-image min-max is handled in __getitem__,
        # then ImageNet mean/std applied here
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Validation / test / inference transforms — deterministic, no augmentation.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image to 3-channel RGB by replicating the single
    channel. EfficientNet (ImageNet-pretrained) expects 3 channels.

    Args:
        image: numpy array, shape (H, W) or (H, W, 1) or (H, W, 3)

    Returns:
        numpy array, shape (H, W, 3), dtype uint8
    """
    if image.ndim == 2:
        # (H, W) -> (H, W, 3)
        return np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        # (H, W, 1) -> (H, W, 3)
        return np.concatenate([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 3:
        # Already RGB
        return image
    else:
        raise ValueError(
            f"Unexpected image shape: {image.shape}. "
            "Expected (H, W), (H, W, 1), or (H, W, 3)."
        )


def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """
    Per-image min-max normalization to [0, 255] range (uint8).
    Handles edge case of constant-intensity images.
    """
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min < 1e-8:
        # Constant image — return zeros to avoid division by zero
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image.astype(np.float32) - img_min) / (img_max - img_min)
    return (normalized * 255).astype(np.uint8)
