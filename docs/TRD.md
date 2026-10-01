# Technical Requirements Document (TRD)
## Alzheimer's Disease Staging from MRI Scans

**Version:** 1.0
**Date:** 2026-09-30
**Status:** Draft

---

## 1. Technical Overview

A **CNN-based 4-class classifier** (dementia stages 0–3) on axial T1-weighted MRI slices, wrapped in a **Streamlit web app**. Training uses PyTorch with ImageNet transfer learning; the app features a stage-progression timeline UI and Grad-CAM overlays localized to neuroanatomically meaningful regions.

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
│  Dataset     │ → │  Preprocess  │ → │  Training   │ → │  Evaluation │
│  (Kaggle MRI)│   │  & Augment   │   │  (PyTorch)  │   │  (F1, CM)   │
└─────────────┘   └──────────────┘   └─────────────┘   └──────────────┘
                                            ↓
                              ┌─────────────────────────┐
                              │  Exported Model (.pt)   │
                              └───────────┬─────────────┘
                                          ↓
                              ┌─────────────────────────┐
                              │   Streamlit App (UI)    │
                              │  Upload → Stage → XAI   │
                              └─────────────────────────┘
```

---

## 2. Technology Stack

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| Language | Python | 3.10+ | Ecosystem compatibility |
| DL Framework | PyTorch | 2.x | Flexible CAM hooks, strong pretrained zoo |
| Vision models | `timm` (EfficientNet-B0) | Latest | Best accuracy/FLOPs ratio; CPU-feasible |
| Augmentation | Albumentations | 1.4+ | Medical imaging transforms |
| Frontend | Streamlit | 1.3x+ | Rapid UI, session state, native charts |
| Timeline UI | Custom HTML/CSS stepper in Streamlit | — | Visual stage progression |
| Explainability | `pytorch-grad-cam` | Latest | Drop-in Grad-CAM for timm models |
| Metrics | `scikit-learn` | Latest | Macro-F1, confusion matrix, per-class reports |
| Containerization | Docker (optional) | — | Reproducible deployment |

**System Requirements:** 8 GB RAM (CPU inference); CUDA GPU recommended for training.

---

## 3. Data Pipeline

### 3.1 Dataset
- **Source:** Alzheimer's MRI Dataset (Kaggle) — derived from OASIS-1
- **Size:** ~6,400 axial MRI slices (1,600 per class — perfectly balanced)
- **Labels:** `0 = Non-Demented`, `1 = Very Mild Dementia`, `2 = Mild Dementia`, `3 = Moderate Dementia`
- **Format:** Grayscale PNG/JPG, pre-cropped to brain region (~128×128 to 256×256)
- **Alternative (scale-up):** OASIS Images Kaggle set (~80,000 slices)
- **Split:** 80% train / 10% validation / 10% test, **stratified by class**

### 3.2 Preprocessing
| Step | Specification |
|------|---------------|
| Grayscale → RGB | Replicate channel ×3 (ImageNet models expect 3 channels) |
| Resize | 224×224 (EfficientNet-B0 native input) |
| Intensity normalization | Per-image min-max normalize to [0, 1], then ImageNet mean/std |
| Skull-strip check | Dataset is pre-skull-stripped; add validation heuristic (center-mass brightness) |
| Augmentation (train only) | Horizontal flip (p=0.5), rotation ±10°, affine shear ±5°, gamma shift (0.9–1.1), Gaussian noise (σ ≤ 0.01) |

**Augmentation rationale:** MRI augmentations must be *anatomy-preserving* — no elastic deformations or heavy crops that could erase atrophy signals (the very features the model needs).

### 3.3 Class Handling
- Dataset is balanced → **no oversampling needed** with primary Kaggle set
- If using OASIS (imbalanced): class-weighted CrossEntropy + `WeightedRandomSampler`
- Adjacent-stage confusion tolerance: misclassifying stage 1 as 2 is clinically far less harmful than 0 as 3 → consider **ordinal-aware loss** (soft labels targeting neighboring classes)

---

## 4. Model Architecture

### 4.1 Backbone
- **Primary:** `EfficientNet-B0` (timm), ImageNet-pretrained
- **Head:** Global average pooling → Dropout(0.4) → Linear(1280 → 256) → ReLU → Dropout(0.2) → Linear(256 → 4)
- **Rationale:** ~5.3M params; light enough for fast CPU inference while retaining strong feature extraction

### 4.2 Alternative / Ensemble Option
- 3-model soft-voting ensemble: EfficientNet-B0 + ResNet18 + DenseNet121
- Adds ~0.01–0.02 Macro-F1; increases inference latency ~3× — offer as "high-accuracy mode" toggle

### 4.3 Training Configuration
| Parameter | Value |
|-----------|-------|
| Loss | CrossEntropy (optionally soft ordinal labels) |
| Optimizer | AdamW, lr = 3e-4 (backbone), 1e-3 (head), weight_decay = 1e-4 |
| Scheduler | Cosine annealing, 2-epoch warmup |
| Batch size | 32 (GPU) / 16 (CPU) |
| Epochs | 25 with early stopping (patience = 5) on val macro-F1 |
| Mixed precision | `torch.cuda.amp` (GPU only) |
| Checkpointing | Best-val-F1 model → `best_model.pt` |

### 4.4 Evaluation Metrics
- **Primary:** Macro-F1 (equal weight per class — critical since stage 3 is clinically most important despite being smallest)
- **Secondary:** Per-class precision/recall, confusion matrix, weighted accuracy
- **Clinical safety metric:** Recall on stage 3 (moderate) — target ≥ 95% ("never miss severe cases")

**Target:** Macro-F1 ≥ 0.90 on held-out test set.

---

## 5. Inference Pipeline (Runtime)

```python
# Pseudo-flow
1. Receive uploaded MRI slice (PIL)
2. Quality gate: brain-like? (aspect ratio + grayscale structure + brightness heuristics)
3. Preprocess: grayscale→RGB → resize 224² → normalize
4. Forward pass → logits → softmax → 4-stage probabilities
5. Post-process: argmax → stage; max prob → confidence
6. Triage rule: stage == 3 → "Urgent Specialist Review" banner
7. Grad-CAM: hook final conv block → 224² heatmap → upscale to original
8. Render: original | heatmap | blended overlay + stage timeline + prob chart
```

### 5.1 Performance Requirements
| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end latency (CPU) | ≤ 4 s | `time.perf_counter()`, logged to UI |
| Model load time | ≤ 8 s (cached) | `@st.cache_resource` singleton |
| Peak RAM | ≤ 3 GB | Monitored during inference |

### 5.2 Caching & Session Strategy
- Model: `@st.cache_resource` (loaded once per session)
- Preprocessing: `@st.cache_data` for repeated uploads
- Zero persistence — images never touch disk

---

## 6. Streamlit Application Structure

```
app.py                 — Main entry: page config, layout, tabs
├── components/
│   ├── uploader.py    — st.file_uploader + brain-scan validation
│   ├── inference.py   — Model loading + prediction wrapper
│   ├── explain.py     — Grad-CAM generation + overlay
│   ├── timeline.py    — Stage progression stepper (HTML/CSS)
│   └── report.py      — Results dashboard
├── utils/
│   ├── preprocess.py  — MRI-specific transforms
│   ├── metrics.py     — F1, confusion matrix (train-time)
│   └── constants.py   — Stage labels, descriptions, triage rules
├── model/
│   └── best_model.pt  — Trained weights
├── config.yaml        — Model path, image size, thresholds
└── requirements.txt
```

### 6.1 Key UI Components
| Component | Implementation |
|-----------|----------------|
| File upload | `st.file_uploader` (accept `image/*`) |
| Scan preview | `st.image` with caption (side-by-side original/heatmap) |
| Stage timeline | HTML/CSS stepper: 4 nodes, active node highlighted, connecting progress bar |
| Probability chart | Plotly express horizontal bar (sorted, colored by stage) |
| Triage banner | `st.error` (urgent) / `st.warning` (referral) / `st.success` (normal) |
| Latency metric | `st.metric("Inference time", f"{ms:.0f} ms")` |
| Disclaimer | `st.warning` — research-use-only footer |

---

## 7. Security, Privacy & Compliance

| Concern | Mitigation |
|---------|------------|
| Patient privacy | In-memory processing only; no logging of image content |
| Medical disclaimer | Persistent footer: "Educational/research use. Not a medical device. Not for diagnosis." |
| Dataset license | OASIS-derived data is research-use; raw data NOT redistributed — only model weights |
| Dependency vulnerabilities | Pinned `requirements.txt`; `pip-audit` in CI |

---

## 8. Testing & Validation

| Test Type | Method | Pass Criteria |
|-----------|--------|---------------|
| Unit | `pytest` on preprocess, metrics, triage logic | 100% pass on utils |
| Model | Held-out test set | Macro-F1 ≥ 0.90; stage-3 recall ≥ 95% |
| Robustness | Low-contrast, rotated, noised test images | ≤ 5% F1 drop |
| Safety audit | All stage-3 test scans | Zero misclassified as stage 0 |
| UI | Manual checklist against all FRs | All P0 features functional |
| Edge cases | Non-brain image, wrong orientation, oversize file | Graceful error with guidance |

---

## 9. Deployment

| Option | Spec |
|--------|------|
| Local | `streamlit run app.py` |
| Cloud | Streamlit Community Cloud / Hugging Face Spaces (model via Git LFS or release asset) |
| Container | `Dockerfile`: `python:3.10-slim`, `EXPOSE 8501` |
| Config | Secrets via Streamlit Secrets / env vars |

### 9.1 Reproducibility
- Seeds: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`
- Dataset version pinned: Kaggle dataset ID + download date in `data/README.md`
- Model card: `MODEL_CARD.md` with intended use, performance, and limitations
