"""
NeuroStage Dataset
==================
PyTorch Dataset and DataLoader factory for Alzheimer's MRI classification.

Handles:
- Loading images from class-based directory structure
- Stratified train/val/test splitting
- Grayscale → RGB conversion
- Per-image min-max normalization
- Class-weighted sampling to handle severe imbalance
  (67K Non-Demented vs. 488 Moderate Dementia)
"""

import os
import random
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from utils.constants import DIR_TO_LABEL
from utils.preprocess import (
    get_train_transforms,
    get_val_transforms,
    grayscale_to_rgb,
    min_max_normalize,
)


class MRIDataset(Dataset):
    """
    PyTorch Dataset for Alzheimer's MRI staging.

    Each item returns a preprocessed 3-channel tensor and its integer label (0–3).
    Grayscale MRI slices are replicated to 3 channels for ImageNet-pretrained
    backbones, and per-image min-max normalized before augmentation.
    """

    def __init__(
        self,
        image_paths: list,
        labels: list,
        transform=None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("L")  # force grayscale
        image = np.array(image, dtype=np.uint8)

        # Per-image min-max normalization to [0, 255]
        image = min_max_normalize(image)

        # Grayscale → RGB (3-channel replication)
        image = grayscale_to_rgb(image)

        # Apply transforms (augmentation + ImageNet normalization + to_tensor)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback: basic tensor conversion
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label


def collect_image_paths(data_root: str) -> Tuple[list, list]:
    """
    Walk the class-based directory structure and collect all image paths
    with their corresponding labels.

    Expected structure:
        data_root/
            Non Demented/       → label 0
            Very mild Dementia/ → label 1
            Mild Dementia/      → label 2
            Moderate Dementia/  → label 3

    Returns:
        (image_paths, labels) — parallel lists
    """
    image_paths = []
    labels = []

    for dir_name, label_id in DIR_TO_LABEL.items():
        class_dir = os.path.join(data_root, dir_name)
        if not os.path.isdir(class_dir):
            print(f"WARNING: Directory not found: {class_dir}")
            continue

        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label_id)

    print(f"Collected {len(image_paths)} images across {len(DIR_TO_LABEL)} classes")
    for dir_name, label_id in DIR_TO_LABEL.items():
        count = labels.count(label_id)
        print(f"  Stage {label_id} ({dir_name}): {count} images")

    return image_paths, labels


def stratified_split(
    image_paths: list,
    labels: list,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Tuple[list, list]]:
    """
    Perform stratified train/val/test split to preserve class distribution
    in each subset. Critical for imbalanced datasets.

    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing
        (image_paths, labels) tuple.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # First split: train vs. (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
        image_paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs. test (from the remaining portion)
    relative_test_ratio = test_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        valtest_paths, valtest_labels,
        test_size=relative_test_ratio,
        stratify=valtest_labels,
        random_state=seed,
    )

    splits = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

    for split_name, (paths, labs) in splits.items():
        print(f"\n{split_name.upper()} split: {len(paths)} images")
        for label_id in sorted(set(labs)):
            count = labs.count(label_id)
            print(f"  Stage {label_id}: {count} ({count/len(labs)*100:.1f}%)")

    return splits


def compute_class_weights(labels: list) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropy loss.
    This ensures the loss penalizes errors on minority classes
    (e.g., Moderate Dementia with only 488 samples) proportionally more.

    Formula: weight_c = N_total / (N_classes × N_c)

    This is the standard sklearn-style "balanced" weighting.
    """
    labels_arr = np.array(labels)
    num_classes = len(set(labels))
    total = len(labels)
    weights = []

    for c in range(num_classes):
        count = (labels_arr == c).sum()
        if count == 0:
            weights.append(1.0)
        else:
            weights.append(total / (num_classes * count))

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"\nClass weights (inverse frequency): {weights_tensor.tolist()}")
    return weights_tensor


def compute_sample_weights(labels: list) -> torch.Tensor:
    """
    Compute per-sample weights for WeightedRandomSampler.
    Each sample's weight is the inverse frequency of its class,
    so minority classes are drawn more frequently per epoch.

    This effectively creates balanced mini-batches during training
    without duplicating data or modifying the dataset.
    """
    class_weights = compute_class_weights(labels)
    sample_weights = torch.tensor(
        [class_weights[label].item() for label in labels],
        dtype=torch.float64,
    )
    return sample_weights


def create_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders with proper class imbalance handling.

    Class imbalance strategy (3-layer defense):
    1. WeightedRandomSampler: oversample minority classes during training
    2. Class-weighted CrossEntropy: penalize errors on rare classes more
    3. Macro-F1 early stopping: ensures minority class performance matters

    Args:
        data_root: Path to dataset root (e.g., "alz_dataset/Data")
        image_size: Target image size (224 for EfficientNet-B0)
        batch_size: Training batch size
        num_workers: DataLoader workers
        train_ratio, val_ratio, test_ratio: Split ratios (must sum to 1)
        use_weighted_sampler: Whether to use WeightedRandomSampler
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
        and 'class_weights' tensor.
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Collect all image paths and labels
    image_paths, labels = collect_image_paths(data_root)

    # Stratified split
    splits = stratified_split(
        image_paths, labels,
        train_ratio, val_ratio, test_ratio, seed
    )

    # Create transforms
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    # Create datasets
    train_dataset = MRIDataset(*splits["train"], transform=train_transform)
    val_dataset = MRIDataset(*splits["val"], transform=val_transform)
    test_dataset = MRIDataset(*splits["test"], transform=val_transform)

    # Compute class weights for loss function
    train_labels = splits["train"][1]
    class_weights = compute_class_weights(train_labels)

    # Create weighted sampler for training
    train_sampler = None
    train_shuffle = True
    if use_weighted_sampler:
        sample_weights = compute_sample_weights(train_labels)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False  # sampler and shuffle are mutually exclusive
        print("Using WeightedRandomSampler for balanced training batches")

    # Create dataloaders
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "class_weights": class_weights,
    }

    return dataloaders
