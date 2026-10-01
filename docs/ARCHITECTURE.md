# Architecture Document
## Alzheimer's Disease Staging from MRI Scans

**Version:** 1.0
**Date:** 2026-09-30
**Status:** Draft

---

## 1. System Context Diagram

```
                    ┌──────────────────────────────┐
                    │   User (Neurologist / GP /   │
                    │   Researcher / Caregiver)    │
                    └──────────────┬───────────────┘
                                   │ HTTPS (browser)
                                   ▼
                    ┌──────────────────────────────┐
                    │      Streamlit Web App       │
                    │  ┌────────────────────────┐  │
                    │  │  UI Layer (app.py)     │  │
                    │  │  upload · stage · XAI  │  │
                    │  └──────────┬─────────────┘  │
                    │  ┌──────────▼─────────────┐  │
                    │  │  Inference Service     │  │
                    │  │  (cached model, CPU)   │  │
                    │  └──────────┬─────────────┘  │
                    └─────────────┼────────────────┘
                                  │ loads weights
                                  ▼
                    ┌──────────────────────────────┐
                    │  Pretrained Model Artifact   │
                    │      (best_model.pt)         │
                    └──────────────────────────────┘

┌─────────────────────────── TRAINING SIDE (offline) ───────────────────────────┐
│                                                                                │
│  Kaggle MRI ──→ Preprocessing ──→ Augmentation ──→ EfficientNet-B0 ──→ Stage  │
│  (~6.4k imgs)   Pipeline          Pipeline           + Custom Head     Head    │
│                                                        │                       │
│                                              ┌─────────▼─────────┐            │
│                                              │  Loss: CE (+      │            │
│                                              │  ordinal soft     │            │
│                                              │  labels option)   │            │
│                                              └─────────┬─────────┘            │
│                                                        ▼                       │
│                                              Checkpoint → best_model.pt        │
│                                                        │                       │
│                                              Evaluation → Macro-F1, CM,       │
│                                                         stage-3 recall         │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

### 2.1 Training Pipeline (Offline)

```
┌─────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ MRI Slices  │ → │  Preprocessing   │ → │  Augmented      │
│ (~6,400,    │   │  · Grayscale→RGB │   │  Batches        │
│  balanced)  │   │  · Resize 224²   │   │  (Albumentations:│
│ + labels    │   │  · Min-max norm  │   │  flip, rotate,  │
└─────────────┘   │  · ImageNet std  │   │  gamma, noise)  │
                  └──────────────────┘   └────────┬────────┘
                                                  ▼
                                         ┌─────────────────┐
                                         │ EfficientNet-B0 │
                                         │  (ImageNet)     │
                                         └────────┬────────┘
                                                  ▼
                                         ┌─────────────────┐
                                         │  Custom Head:   │
                                         │  GAP→DO→FC→DO→  │
                                         │  FC(4 logits)   │
                                         └────────┬────────┘
                                                  ▼
                            ┌───────────────────────────────────┐
                            │  Loss: CrossEntropy               │
                            │  Optimizer: AdamW                 │
                            │  (backbone 3e-4 · head 1e-3)      │
                            │  Scheduler: cosine + warmup       │
                            └─────────────────┬─────────────────┘
                                              ▼
                            ┌───────────────────────────────────┐
                            │  Early stop on val macro-F1       │
                            │  → save best_model.pt             │
                            │  → eval: F1, CM, stage-3 recall   │
                            └───────────────────────────────────┘
```

**Key design decisions:**
- **Grayscale→RGB replication:** EfficientNet expects 3 channels; MRI is grayscale. Replicating channels preserves ImageNet pretrained features better than training from scratch on 1-channel.
- **Anatomy-preserving augmentation:** No elastic deformation or heavy cropping — these could erase the atrophy patterns (enlarged ventricles, shrunken hippocampus) that are the model's signal.
- **Macro-F1 early stopping:** Ensures minority clinically-critical stage 3 drives model selection, not just overall accuracy.
- **Ordinal soft-label option:** Optionally train with soft targets (e.g., true=2 gets [0, 0.1, 0.8, 0.1]) to penalize adjacent-stage errors less than distant-stage errors — clinically aligned loss shaping.

### 2.2 Inference Pipeline (Online)

```
Upload (PIL MRI slice)
       │
       ▼
┌──────────────────┐
│ Quality Gate     │ ──reject──▶ "Please upload an axial brain MRI slice"
│ · Brain-like?    │
│ · Grayscale?     │
│ · Aspect ratio   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Preprocessing    │
│ · grayscale→RGB  │
│ · resize 224²    │
│ · normalize      │
└────────┬─────────┘
         ▼
┌──────────────────┐     ┌──────────────────┐
│ Model Forward    │ ──→ │ softmax probs[4] │
│ (cached)         │     │ argmax → stage   │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐     ┌──────────────────┐
│ Grad-CAM Hook    │     │ Post-processing  │
│ (final conv blk) │     │ · stage 3 →      │
│ → 224² heatmap   │     │   URGENT banner  │
└────────┬─────────┘     │ · confidence %   │
         │               └────────┬─────────┘
         └───────────┬────────────┘
                     ▼
         ┌───────────────────────────┐
         │   Streamlit Dashboard     │
         │  · Stage card + timeline  │
         │  · prob bar chart         │
         │  · original/heatmap views │
         │  · triage banner          │
         │  · latency metric         │
         │  · disclaimer footer      │
         └───────────────────────────┘
```

### 2.3 Streamlit Page Layout

```
┌────────────────────────────────────────────────────────────┐
│  🧠 Alzheimer's MRI Staging      [Sidebar: About · Stages]  │
├──────────────────────┬───────────────────────────────────────┤
│                      │                                       │
│  [Upload Widget]     │   STAGE TIMELINE                      │
│                      │   ●──────●══════●──────○               │
│  🖼️ MRI preview      │   0   1   [2]   3  ← You are here      │
│                      │                                       │
│  [Analyze button]    │   ┌─────────────────────────────┐     │
│                      │   │ Stage 2 — Mild Dementia     │     │
│                      │   │ Confidence: 91.2%           │     │
│                      │   │ ⚠ Refer to specialist       │     │
│                      │   └─────────────────────────────┘     │
│                      │                                       │
│                      │   Probability distribution:           │
│                      │   NonDem 4% ███ vMild 5%             │
│                      │   ██████████ Mild 91% Moderate 0%    │
│                      │                                       │
│                      │   Grad-CAM: [Original|Heatmap|Blend]  │
│                      │   ℹ Inference: 1.1s · 224×224        │
└──────────────────────┴───────────────────────────────────────┘
  ⚠ Educational/research use only — not a medical device.
```

---

## 3. Model Architecture Detail

```
Input: 3 × 224 × 224 (grayscale replicated to RGB)
│
├─▶ EfficientNet-B0 Backbone (ImageNet pretrained)
│     Conv stem → MBConv blocks (squeeze-excite) → 1280 × 7 × 7 feature map
│
├─▶ Global Average Pooling → 1280-d vector
│
├─▶ Dropout(0.4)
├─▶ Linear(1280 → 256) + ReLU
├─▶ Dropout(0.2)
└─▶ Linear(256 → 4)   ← stage logits (0–3)

Params: ~5.3M (backbone) + ~0.4M (head) ≈ 5.7M total
```

**Why EfficientNet-B0:**
- ~5.3M params — smallest viable strong backbone; meets ≤4s CPU inference target easily
- Squeeze-excite attention aligns with localized atrophy detection (ventricles, hippocampus are spatially localized)
- ImageNet features transfer well to medical imaging (texture/edge priors)

**Optional ensemble (high-accuracy mode):**
- Soft-vote: EfficientNet-B0 + ResNet18 + DenseNet121
- Expected +0.01–0.02 Macro-F1; 3× latency — present as a UI toggle "⚡ Fast mode / 🎯 Accurate mode"

---

## 4. Neuroanatomical Interpretation Layer

A unique component of this project — mapping Grad-CAM regions to clinical meaning:

| Expected Hotspot | Stage Correlation | Clinical Meaning |
|------------------|-------------------|------------------|
| Hippocampus (medial temporal lobe) | 1 → 2 | Earliest atrophy site; memory circuits |
| Lateral ventricles | 1 → 3 | Enlargement indicates tissue loss (hydrocephalus ex vacuo) |
| Cortical ribbon (parietal/temporal) | 2 → 3 | Cortical thinning; sulcal widening |
| Global brain volume | 3 | Diffuse atrophy in moderate stage |

The app includes an "Anatomy Guide" info box that explains which regions the model is focusing on — turning a heatmap into a teachable moment.

---

## 5. Data Flow Summary

| Stage | Data Format | Transformation |
|-------|------------|----------------|
| Raw | Grayscale PNG/JPG | — |
| Channel | 3×H×W | Grayscale replicated to RGB |
| Tensor | Float32, CHW | Resize 224², min-max + ImageNet normalization |
| Augmented (train only) | Float32, CHW | Flip/rotate/gamma/noise (anatomy-preserving) |
| Features | 1280-d | GAP of final conv block |
| Output | 4-prob vector | Softmax over stage logits |
| Explanation | 224² heatmap | Grad-CAM on final conv block |

---

## 6. Error Handling & Edge Cases

| Case | Handling |
|------|----------|
| Non-image file | `st.error` — "Upload a valid MRI image (PNG/JPG)" |
| Non-brain image (photo, text) | Quality gate rejects: "This doesn't look like a brain MRI slice" |
| Wrong orientation (coronal/sagittal) | Heuristic aspect + structure check → warning with guidance |
| Very low resolution | Upscaling warning; note that staging confidence may be reduced |
| Model file missing | Startup validation with setup instructions |
| Confidence < 60% | Display "Low confidence — consider specialist review" notice regardless of stage |

---

## 7. Scalability & Extensibility

- **Stateless inference** → containerizable, horizontally scalable
- **3D volumetric extension (v2):** Swap 2D CNN for 3D ResNet / nnU-Net on full MRI volumes; aggregate per-slice predictions
- **Longitudinal extension (v2):** Accept two scans (baseline + follow-up) → progression rate classifier (stable/slow/fast decliner)
- **Multi-task head extension:** Auxiliary outputs (age estimation, brain-volume regression) can improve staging accuracy via multi-task learning
- **Model versioning:** `model_v{f1}_{date}.pt`; `config.yaml` selects active version

---

## 8. Roadmap: v2 Enhancements

| Feature | Description | Effort |
|---------|-------------|--------|
| 3D volumetric inference | Full MRI volume analysis (3D CNN) | High |
| Longitudinal comparison | Two-scan progression tracking | Medium |
| DICOM support | Native medical imaging format parsing | Medium |
| Multi-dementia types | Extend to Lewy body / vascular dementia classes | High |
| Report generation | Automated clinical-style PDF reports | Low |
