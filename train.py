"""
NeuroStage Training Script
===========================
End-to-end training pipeline for the Alzheimer's Disease staging model.

Features:
- EfficientNet-B0 with differential learning rates (backbone vs. head)
- 3-layer class imbalance defense:
    1. WeightedRandomSampler (balanced batches)
    2. Class-weighted CrossEntropy loss
    3. Macro-F1 early stopping
- Cosine annealing with linear warmup
- Mixed precision training (GPU)
- Comprehensive evaluation with safety audit

Usage:
    python train.py
    python train.py --epochs 30 --batch-size 16 --no-gpu
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

from model.architecture import build_model, count_parameters
from utils.dataset import create_dataloaders
from utils.metrics import (
    compute_macro_f1,
    compute_stage3_recall,
    compute_safety_check,
    print_classification_report,
    plot_confusion_matrix,
)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
):
    """
    Cosine annealing LR schedule with linear warmup.
    Warmup linearly increases LR from 0 to base_lr over warmup steps.
    Cosine then decays LR to 0 over the remaining steps.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model, dataloader, criterion, optimizer, scheduler, scaler, device, epoch
):
    """Run one training epoch with optional mixed precision."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        if scaler is not None:
            with autocast(device.type):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = compute_macro_f1(np.array(all_labels), np.array(all_preds))

    return epoch_loss, epoch_f1


@torch.no_grad()
def validate(model, dataloader, criterion, device, split_name="Val"):
    """Evaluate model on validation or test set."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"[{split_name}]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    epoch_f1 = compute_macro_f1(y_true, y_pred)
    stage3_recall = compute_stage3_recall(y_true, y_pred)

    return epoch_loss, epoch_f1, stage3_recall, y_true, y_pred


def train(config: dict):
    """Full training pipeline."""
    # ----- Setup -----
    seed = config["data"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"NeuroStage Training")
    print(f"{'='*65}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ----- Data -----
    print(f"\n--- Data Pipeline ---")
    data_cfg = config["data"]
    train_cfg = config["training"]

    dataloaders = create_dataloaders(
        data_root=data_cfg["root_dir"],
        image_size=data_cfg["image_size"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        train_ratio=data_cfg["split_ratios"]["train"],
        val_ratio=data_cfg["split_ratios"]["val"],
        test_ratio=data_cfg["split_ratios"]["test"],
        use_weighted_sampler=train_cfg["use_weighted_sampler"],
        seed=seed,
    )

    # ----- Model -----
    print(f"\n--- Model ---")
    model = build_model(config).to(device)
    params = count_parameters(model)
    print(f"Architecture: {config['model']['backbone']}")
    print(f"Parameters: {params['total_millions']} total, "
          f"{params['trainable_millions']} trainable")

    # ----- Loss (class-weighted CrossEntropy) -----
    class_weights = dataloaders["class_weights"].to(device)
    if train_cfg["use_class_weights"]:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Loss: CrossEntropy with class weights {class_weights.cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"Loss: CrossEntropy (unweighted)")

    # ----- Optimizer (differential LR: backbone slower, head faster) -----
    opt_cfg = train_cfg["optimizer"]
    optimizer = torch.optim.AdamW([
        {"params": model.get_backbone_params(), "lr": opt_cfg["backbone_lr"]},
        {"params": model.get_head_params(), "lr": opt_cfg["head_lr"]},
    ], weight_decay=opt_cfg["weight_decay"])
    print(f"Optimizer: AdamW (backbone LR={opt_cfg['backbone_lr']}, "
          f"head LR={opt_cfg['head_lr']})")

    # ----- Scheduler -----
    sched_cfg = train_cfg["scheduler"]
    steps_per_epoch = len(dataloaders["train"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=sched_cfg["warmup_epochs"],
        total_epochs=train_cfg["epochs"],
        steps_per_epoch=steps_per_epoch,
    )
    print(f"Scheduler: Cosine annealing, {sched_cfg['warmup_epochs']} warmup epochs")

    # ----- Mixed precision -----
    scaler = None
    if train_cfg["mixed_precision"] and device.type == "cuda":
        scaler = GradScaler(device.type)
        print(f"Mixed precision: enabled (AMP)")
    else:
        print(f"Mixed precision: disabled")

    # ----- Training loop -----
    es_cfg = train_cfg["early_stopping"]
    best_metric = 0.0
    patience_counter = 0
    weights_path = config["model"]["weights_path"]
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    history = {
        "train_loss": [], "train_f1": [],
        "val_loss": [], "val_f1": [],
        "val_stage3_recall": [], "lr": [],
    }

    print(f"\n--- Training ({train_cfg['epochs']} epochs, "
          f"patience={es_cfg['patience']}) ---\n")

    for epoch in range(1, train_cfg["epochs"] + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_f1 = train_one_epoch(
            model, dataloaders["train"], criterion,
            optimizer, scheduler, scaler, device, epoch
        )

        # Validate
        val_loss, val_f1, val_stage3_recall, _, _ = validate(
            model, dataloaders["val"], criterion, device, "Val"
        )

        # Current LR
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_stage3_recall"].append(val_stage3_recall)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:>2}/{train_cfg['epochs']} │ "
            f"Train Loss: {train_loss:.4f}  F1: {train_f1:.4f} │ "
            f"Val Loss: {val_loss:.4f}  F1: {val_f1:.4f} │ "
            f"S3-Recall: {val_stage3_recall:.4f} │ "
            f"LR: {current_lr:.6f} │ {elapsed:.1f}s"
        )

        # Early stopping on macro-F1
        metric = val_f1
        if metric > best_metric:
            best_metric = metric
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_loss": val_loss,
                "val_stage3_recall": val_stage3_recall,
                "config": config,
            }, weights_path)
            print(f"  ↑ New best Macro-F1: {best_metric:.4f} — saved to {weights_path}")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{es_cfg['patience']})")
            if patience_counter >= es_cfg["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (best F1: {best_metric:.4f})")
                break

    # ----- Final evaluation on test set -----
    print(f"\n{'='*65}")
    print("FINAL EVALUATION (Test Set)")
    print(f"{'='*65}")

    # Load best model
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from epoch {checkpoint['epoch']} "
          f"(val F1: {checkpoint['val_f1']:.4f})")

    test_loss, test_f1, test_stage3_recall, y_true, y_pred = validate(
        model, dataloaders["test"], criterion, device, "Test"
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Macro-F1: {test_f1:.4f}")
    print(f"Test Stage-3 Recall: {test_stage3_recall:.4f}")

    # Detailed report
    print_classification_report(y_true, y_pred)

    # Safety audit
    safety = compute_safety_check(y_true, y_pred)
    print(f"\nClinical Safety Audit:")
    print(f"  Stage 3→0 misclassifications: {safety['stage3_as_stage0']}")
    print(f"  Safety check: {'PASSED ✅' if safety['passed'] else 'FAILED ❌'}")

    # Confusion matrix
    cm_path = os.path.join(os.path.dirname(weights_path), "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, save_path=cm_path, normalize=True)

    print(f"\n{'='*65}")
    print(f"Training complete!")
    print(f"Best model: {weights_path}")
    print(f"Confusion matrix: {cm_path}")
    print(f"{'='*65}\n")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train NeuroStage model")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU training")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    train(config)


if __name__ == "__main__":
    main()
