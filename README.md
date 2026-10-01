# 🧠 NeuroStage — Alzheimer's MRI Staging

AI-powered web application that analyzes axial T1-weighted brain MRI scans and classifies the patient's dementia stage on a 4-point clinical scale (CDR 0–3).

> ⚠️ **Disclaimer:** This is an educational/research tool — NOT a medical device. Not for clinical diagnosis.

---

## ✨ Features

- **4-Stage Classification** — Non-Demented, Very Mild, Mild, Moderate Dementia
- **Grad-CAM Explainability** — Heatmaps highlighting atrophy-affected brain regions
- **Progression Timeline** — Visual stepper showing disease stage positioning
- **Clinical Triage** — Automated recommendations (routine → urgent referral)
- **Plain-Language Mode** — Descriptions for non-clinical users
- **Anatomy Guide** — Maps model attention to neuroanatomical meaning
- **Quality Gate** — Validates uploads are brain MRI scans

---

## 🏗️ Architecture

```
EfficientNet-B0 (ImageNet pretrained)
    ├── Backbone: ~5.3M params (squeeze-excite attention)
    ├── GAP → Dropout(0.4) → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→4)
    └── Total: ~5.7M parameters
```

---

## 📁 Project Structure

```
NeuroStage/
├── app.py                  # Streamlit web application
├── train.py                # Model training pipeline
├── config.yaml             # Configuration (hyperparams, paths)
├── requirements.txt        # Python dependencies
├── components/
│   ├── uploader.py         # File upload + quality gate
│   ├── inference.py        # Model loading + prediction
│   ├── explain.py          # Grad-CAM generation
│   ├── timeline.py         # Stage progression stepper
│   └── report.py           # Results dashboard
├── model/
│   ├── architecture.py     # EfficientNet-B0 + custom head
│   └── best_model.pt       # Trained weights (after training)
├── utils/
│   ├── constants.py        # Stage labels, descriptions, triage rules
│   ├── preprocess.py       # MRI-specific transforms
│   ├── metrics.py          # Evaluation (F1, confusion matrix)
│   └── dataset.py          # Data loading + stratified splits
└── alz_dataset/            # Dataset (not committed)
    └── Data/
        ├── Non Demented/
        ├── Very mild Dementia/
        ├── Mild Dementia/
        └── Moderate Dementia/
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

Overrides available:
```bash
python train.py --epochs 30 --batch-size 16 --no-gpu
```

### 3. Launch the App

```bash
streamlit run app.py
```

---

## 📊 Class Imbalance Strategy

The dataset is heavily imbalanced:

| Class | Count |
|---|---|
| Non Demented | 67,222 |
| Very Mild Dementia | 13,725 |
| Mild Dementia | 5,002 |
| Moderate Dementia | 488 |

**3-layer defense:**
1. **WeightedRandomSampler** — oversample minority classes in each batch
2. **Class-weighted CrossEntropy** — penalize errors on rare classes proportionally
3. **Macro-F1 early stopping** — ensures minority class performance drives model selection

---

## 🔬 Explainability

Grad-CAM hooks the final convolutional block to produce attention heatmaps mapped to clinical meaning:

| Region | Stages | Clinical Meaning |
|---|---|---|
| Hippocampus | 1–2 | Earliest atrophy site; memory circuits |
| Ventricles | 1–3 | Enlargement indicates tissue loss |
| Cortex | 2–3 | Cortical thinning; sulcal widening |
| Global | 3 | Diffuse atrophy in moderate stage |

---

## 📈 Target Metrics

- **Macro-F1 ≥ 0.90** across 4 classes
- **Stage-3 Recall ≥ 95%** (never miss severe cases)
- **Inference ≤ 4 seconds** on CPU
- **Zero Stage 3→0 misclassifications** (safety audit)

---

## 📜 Dataset

Derived from OASIS-1 (Open Access Series of Imaging Studies). Alzheimer's MRI Dataset from Kaggle (~86K axial slices).

---

## 📄 License

Research and educational use only.
