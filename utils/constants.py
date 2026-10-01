"""
NeuroStage Constants
====================
Stage labels, clinical descriptions, triage rules, and neuroanatomical
interpretation mappings for the Alzheimer's Disease staging system.
"""

# ---------------------------------------------------------------------------
# Stage definitions (Clinical Dementia Rating — CDR)
# ---------------------------------------------------------------------------

STAGE_LABELS = {
    0: "Non-Demented",
    1: "Very Mild Dementia",
    2: "Mild Dementia",
    3: "Moderate Dementia",
}

STAGE_SHORT_LABELS = {
    0: "NonDem",
    1: "vMild",
    2: "Mild",
    3: "Moderate",
}

STAGE_COLORS = {
    0: "#22c55e",   # green
    1: "#eab308",   # amber
    2: "#f97316",   # orange
    3: "#ef4444",   # red
}

STAGE_EMOJIS = {
    0: "✅",
    1: "⚠️",
    2: "🔶",
    3: "🔴",
}

# ---------------------------------------------------------------------------
# Clinical descriptions (plain-language for caregivers + clinical for MDs)
# ---------------------------------------------------------------------------

STAGE_DESCRIPTIONS = {
    0: {
        "clinical": (
            "No cognitive impairment detected. Brain structures appear within "
            "normal limits with preserved hippocampal volume and normal "
            "ventricular size."
        ),
        "plain": (
            "The brain scan looks healthy — no signs of dementia-related "
            "changes were found. This is a normal result."
        ),
    },
    1: {
        "clinical": (
            "Subtle cognitive decline consistent with very mild dementia "
            "(CDR 0.5). Mild hippocampal atrophy and slight ventricular "
            "enlargement may be present."
        ),
        "plain": (
            "Very early signs of change were detected. Memory may be "
            "slightly affected, but daily activities are still independent. "
            "This is a stage worth monitoring closely."
        ),
    },
    2: {
        "clinical": (
            "Noticeable cognitive deficits consistent with mild dementia "
            "(CDR 1). Clear hippocampal and cortical atrophy with enlarged "
            "ventricles visible on imaging."
        ),
        "plain": (
            "The scan shows noticeable brain changes. Memory and thinking "
            "abilities may be affected, and some daily tasks may need help. "
            "A specialist visit is recommended."
        ),
    },
    3: {
        "clinical": (
            "Significant impairment consistent with moderate dementia "
            "(CDR 2). Severe generalized atrophy with markedly enlarged "
            "ventricles and widened sulci."
        ),
        "plain": (
            "The scan shows significant brain changes. Daily activities "
            "likely require substantial assistance. An urgent specialist "
            "review and comprehensive care plan are recommended."
        ),
    },
}

# ---------------------------------------------------------------------------
# Triage / recommended actions
# ---------------------------------------------------------------------------

TRIAGE_RULES = {
    0: {
        "level": "success",
        "action": "Routine annual check",
        "icon": "✅",
        "message": "No signs of dementia — routine follow-up recommended.",
    },
    1: {
        "level": "warning",
        "action": "Baseline established; 6–12 month follow-up",
        "icon": "⚠️",
        "message": (
            "Very mild changes detected — establish a baseline and schedule "
            "a 6–12 month follow-up scan."
        ),
    },
    2: {
        "level": "warning",
        "action": "Refer to specialist; begin intervention",
        "icon": "🔶",
        "message": (
            "Mild dementia signs present — referral to a neurologist and "
            "early intervention planning are advised."
        ),
    },
    3: {
        "level": "error",
        "action": "URGENT: Full specialist review + care plan needed",
        "icon": "🚨",
        "message": (
            "Moderate dementia indicators detected — URGENT specialist "
            "review and comprehensive care plan are required immediately."
        ),
    },
}

# ---------------------------------------------------------------------------
# Neuroanatomical interpretation (maps Grad-CAM regions to clinical meaning)
# ---------------------------------------------------------------------------

ANATOMY_GUIDE = {
    "hippocampus": {
        "stages": [1, 2],
        "location": "Medial temporal lobe",
        "meaning": (
            "The hippocampus is the brain's memory center. Atrophy here is "
            "one of the earliest signs of Alzheimer's, affecting the ability "
            "to form new memories."
        ),
    },
    "ventricles": {
        "stages": [1, 2, 3],
        "location": "Central brain cavities",
        "meaning": (
            "Enlarged ventricles (hydrocephalus ex vacuo) indicate surrounding "
            "brain tissue has been lost. Progressive enlargement correlates "
            "with disease severity."
        ),
    },
    "cortex": {
        "stages": [2, 3],
        "location": "Parietal and temporal cortical ribbon",
        "meaning": (
            "Cortical thinning and widened sulci reflect loss of neurons in "
            "higher-thinking areas, affecting language, reasoning, and "
            "spatial awareness."
        ),
    },
    "global": {
        "stages": [3],
        "location": "Diffuse (entire brain)",
        "meaning": (
            "Widespread, diffuse brain atrophy is characteristic of moderate "
            "to severe Alzheimer's, indicating extensive neuronal loss across "
            "multiple brain regions."
        ),
    },
}

# ---------------------------------------------------------------------------
# Directory-to-label mapping (matches the dataset folder names)
# ---------------------------------------------------------------------------

DIR_TO_LABEL = {
    "Non Demented": 0,
    "Very mild Dementia": 1,
    "Mild Dementia": 2,
    "Moderate Dementia": 3,
}

LABEL_TO_DIR = {v: k for k, v in DIR_TO_LABEL.items()}

# ---------------------------------------------------------------------------
# ImageNet normalization (used since EfficientNet is ImageNet-pretrained)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Quality gate thresholds
# ---------------------------------------------------------------------------

QUALITY_GATE = {
    "min_size_pixels": 64,
    "max_file_size_mb": 10,
    "min_aspect_ratio": 0.4,
    "max_aspect_ratio": 2.5,
    "min_mean_intensity": 10,    # reject near-black images
    "max_mean_intensity": 245,   # reject near-white images
}
