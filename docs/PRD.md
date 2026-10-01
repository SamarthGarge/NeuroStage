# Product Requirements Document (PRD)
## Alzheimer's Disease Staging from MRI Scans

**Version:** 1.0
**Date:** 2026-09-30
**Product Owner:** [Your Name]
**Status:** Draft

---

## 1. Overview

### 1.1 Product Summary
An AI-powered web application that analyzes axial T1-weighted brain MRI scans and classifies the patient's dementia stage on a 4-point clinical scale. The application provides clinicians, researchers, and caregivers with an instant, explainable staging assessment — including a disease progression timeline, brain-region attention maps, and triage recommendations.

### 1.2 Problem Statement
Alzheimer's Disease affects over 55 million people worldwide, and early intervention is the only proven way to slow progression. Yet:
- **Specialist shortage:** Neurologists and neuroradiologists are scarce; wait times for MRI interpretation can exceed months
- **Subtle early signs:** Very mild cognitive impairment (vMCI) is easily missed by the human eye in routine scans
- **Monitoring burden:** Tracking progression across repeated scans requires consistent, objective staging
- **Caregiver anxiety:** Families often lack accessible, understandable explanations of scan results

An automated staging tool provides consistent first-pass triage, flags rapid progressors for urgent specialist review, and translates clinical staging into plain language for non-specialists.

### 1.3 Goals & Objectives
| Goal | Success Metric |
|------|----------------|
| Accurate staging | Macro-F1 ≥ 0.90 across 4 classes |
| Clinical sensitivity | ≥ 95% sensitivity on "Moderate Dementia" (never miss severe cases) |
| Fast inference | ≤ 4 seconds per scan on CPU |
| Interpretability | Region-based attention (Grad-CAM) shown for every prediction |
| Usability | Works in any modern browser, no installation required |

### 1.4 Non-Goals (Out of Scope)
- **Not a diagnostic device** — decision-support / triage tool only
- No patient longitudinal tracking in v1 (single-scan assessment)
- No 3D volume analysis in v1 (single-slice classification; 3D CNN in v2 roadmap)
- No other dementia types (Lewy body, vascular dementia) in v1
- No raw DICOM support in v1 (PNG/JPEG uploads only)

---

## 2. Target Users

| Persona | Description | Key Need |
|---------|-------------|----------|
| Screening Neurologist | Specialist reviewing high scan volumes | Fast triage of which patients need urgent follow-up |
| General Practitioner | Primary care doctor in non-specialist settings | Objective flag for dementia referral |
| MRI Technician | Operator at imaging centers | Quality-check + preliminary staging note |
| Caregiver / Patient (indirect) | Family member of a diagnosed patient | Plain-language explanation of what the stage means |
| ML Researcher / Student | Portfolio, academic use | Reproducible pipeline with explainable outputs |

---

## 3. Functional Requirements

### 3.1 Core Features (Must-Have)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Upload brain MRI scan (PNG/JPG, axial view, max 10 MB) | P0 |
| FR-2 | Run deep learning inference → return stage 0–3 | P0 |
| FR-3 | Display stage label + plain-language description | P0 |
| FR-4 | Display confidence score + probability distribution across 4 stages | P0 |
| FR-5 | Render Grad-CAM heatmap highlighting atrophy-affected regions (hippocampus, ventricles, cortex) | P0 |
| FR-6 | Render **progression timeline** showing where the scan sits on the 4-stage scale | P0 |
| FR-7 | Triage flag: "Urgent Specialist Review" if stage = 3 | P0 |
| FR-8 | Show side-by-side comparison: uploaded scan vs. heatmap overlay | P1 |
| FR-9 | Display inference latency + scan metadata (dimensions, file size) | P1 |
| FR-10 | Downloadable summary report (PDF/PNG) of results | P2 |
| FR-11 | Batch mode: process multiple slices from one scan (optional) | P2 |

### 3.2 Staging Scale (Clinical Dementia Rating — CDR)

| Stage | Label | Clinical Description | Typical Findings on MRI | Recommended Action |
|-------|-------|---------------------|------------------------|--------------------|
| 0 | **Non-Demented** | No cognitive impairment | Normal brain volume | Routine annual check |
| 1 | **Very Mild Dementia** | Subtle memory lapses; independence retained | Mild hippocampal atrophy, slight ventricular enlargement | Baseline established; 6–12 month follow-up |
| 2 | **Mild Dementia** | Noticeable memory/cognitive deficits; daily tasks affected | Clear hippocampal + cortical atrophy, enlarged ventricles | Refer to specialist; begin intervention |
| 3 | **Moderate Dementia** | Significant impairment; needs assistance with daily living | Severe atrophy, markedly enlarged ventricles, widened sulci | **Refer — urgent; full care plan needed** |

### 3.3 User Flow

```
Upload MRI Scan → Quality Gate (check it's a brain axial slice)
               → Model Inference (EfficientNet-B0 ensemble)
               → Results Dashboard
                   ├── Stage label + description card
                   ├── Probability bar chart (4 stages)
                   ├── Progression timeline (stepper UI)
                   ├── Grad-CAM heatmap overlay
                   ├── Triage recommendation banner
                   └── Inference time + scan metadata
```

---

## 4. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| Performance | P95 inference ≤ 4 s on CPU; ≤ 1 s on GPU |
| Accuracy | Macro-F1 ≥ 0.90; Sensitivity ≥ 95% on stage 3 |
| Robustness | Graceful error for non-brain images, wrong-orientation scans, low-quality uploads |
| Privacy | Zero data persistence — all processing in memory |
| Compatibility | Chrome, Firefox, Safari, Edge (latest 2 versions) |
| Accessibility | Plain-language mode toggle for non-clinical users |

---

## 5. Success Metrics

- **Model:** Macro-F1 ≥ 0.90; per-class recall reported; confusion matrix inspected for adjacent-stage confusions (1↔2 is acceptable; 0↔3 is not)
- **Product:** ≥ 75% of users view the Grad-CAM overlay (explainability engagement)
- **Clinical safety:** 0% of moderate-dementia scans misclassified as non-demented on test set

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Dataset provenance ambiguity (Kaggle mirror of OASIS) | Document source clearly; state "derived from OASIS-1" in disclaimers; cite original dataset |
| Class imbalance (stage 3 is rare) | Class-weighted loss + oversampling; report per-class metrics, not just accuracy |
| Single-slice ≠ full diagnosis | Clear disclaimer: "Single-slice screening only; full volumetric clinical review required" |
| Regulatory sensitivity (medical device) | Footer disclaimer: "Educational/research tool — not for clinical diagnosis" |
| Orientation sensitivity (axial vs. coronal slices) | Quality gate rejects non-axial-looking images with guidance message |

---

## 7. Release Plan

| Milestone | Deliverable |
|-----------|-------------|
| M1 (Week 1) | Data exploration + baseline model (ResNet18) |
| M2 (Week 2–3) | Fine-tuned EfficientNet-B0, Macro-F1 ≥ 0.88 |
| M3 (Week 4) | Streamlit app with all P0 features |
| M4 (Week 5) | Grad-CAM + progression timeline UI + testing |
| M5 (Week 6) | Deployment + documentation + video demo |
