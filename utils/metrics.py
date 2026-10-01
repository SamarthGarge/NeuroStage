"""
NeuroStage Metrics
==================
Evaluation utilities for model training and validation. Provides Macro-F1,
per-class precision/recall, confusion matrix plotting, and the critical
stage-3 recall safety metric.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    accuracy_score,
)

from utils.constants import STAGE_LABELS


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged F1 score across all 4 stages."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def compute_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """
    Compute per-class precision, recall, F1, and support.

    Returns:
        Dictionary with keys 'precision', 'recall', 'f1', 'support',
        each mapping to a dict of {stage_label: value}.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    result = {"precision": {}, "recall": {}, "f1": {}, "support": {}}
    for i, label in STAGE_LABELS.items():
        if i < len(precision):
            result["precision"][label] = float(precision[i])
            result["recall"][label] = float(recall[i])
            result["f1"][label] = float(f1[i])
            result["support"][label] = int(support[i])
    return result


def compute_stage3_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute recall specifically for stage 3 (Moderate Dementia).
    This is the critical clinical safety metric — target ≥ 95%.
    Missing a moderate dementia case is the most dangerous failure mode.
    """
    stage3_mask = y_true == 3
    if stage3_mask.sum() == 0:
        return 0.0
    return float((y_pred[stage3_mask] == 3).sum() / stage3_mask.sum())


def compute_safety_check(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Clinical safety audit: check for dangerous misclassifications.
    Stage 3 → 0 is the most dangerous error (severe dementia missed entirely).

    Returns:
        Dictionary with safety metrics and flagged cases.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    return {
        "stage3_recall": compute_stage3_recall(y_true, y_pred),
        "stage3_as_stage0": int(cm[3, 0]) if cm.shape[0] > 3 else 0,
        "stage3_as_stage1": int(cm[3, 1]) if cm.shape[0] > 3 else 0,
        "stage0_as_stage3": int(cm[0, 3]) if cm.shape[1] > 3 else 0,
        "passed": (
            (cm[3, 0] == 0 if cm.shape[0] > 3 else True)  # no 3→0
            and compute_stage3_recall(y_true, y_pred) >= 0.95
        ),
    }


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray):
    """Print a formatted classification report with stage labels."""
    target_names = [STAGE_LABELS[i] for i in range(4)]
    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORT")
    print("=" * 65)
    print(classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
    ))
    print(f"Macro-F1: {compute_macro_f1(y_true, y_pred):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    safety = compute_safety_check(y_true, y_pred)
    print(f"\nStage-3 Recall: {safety['stage3_recall']:.4f} "
          f"(target ≥ 0.95) {'✅' if safety['stage3_recall'] >= 0.95 else '❌'}")
    print(f"Stage 3→0 errors: {safety['stage3_as_stage0']} "
          f"{'✅' if safety['stage3_as_stage0'] == 0 else '❌ CRITICAL'}")
    print(f"Safety check: {'PASSED ✅' if safety['passed'] else 'FAILED ❌'}")
    print("=" * 65 + "\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap with stage labels.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Optional file path to save the figure
        normalize: If True, show percentages; otherwise counts
    """
    labels = [0, 1, 2, 3]
    target_names = [STAGE_LABELS[i] for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        # Normalize per row (true labels) to show recall per class
        cm_display = cm.astype(float)
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm_display = cm_display / row_sums
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Recall" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted Stage", fontsize=12)
    ax.set_ylabel("True Stage", fontsize=12)
    ax.set_title(
        "Confusion Matrix" + (" (Normalized)" if normalize else ""),
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    return fig
