"""
NeuroStage Model Architecture
==============================
EfficientNet-B0 backbone with a custom classification head for 4-stage
Alzheimer's Disease staging from axial brain MRI slices.

Architecture:
    Input  → 3 × 224 × 224 (grayscale replicated to RGB)
    Backbone → EfficientNet-B0 (ImageNet pretrained, ~5.3M params)
    GAP    → 1280-d feature vector
    Head   → Dropout(0.4) → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→4)
    Output → 4 logits (stage 0–3)

Total: ~5.7M parameters
"""

import torch
import torch.nn as nn
import timm


class AlzheimerStagingModel(nn.Module):
    """
    EfficientNet-B0 based classifier for Alzheimer's Disease staging.

    Uses transfer learning from ImageNet — the texture/edge priors transfer
    well to medical imaging. The squeeze-excite attention in EfficientNet
    aligns well with detecting localized atrophy patterns (ventricles,
    hippocampus).
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 4,
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout_1: float = 0.4,
        dropout_2: float = 0.2,
    ):
        super().__init__()

        # Load pretrained backbone, remove its native classifier
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # remove classification head
            global_pool="avg",      # global average pooling
        )

        # Get feature dimension from the backbone
        self.feature_dim = self.backbone.num_features  # 1280 for efficientnet_b0

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_1),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_2),
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialize head weights with Kaiming/He initialization
        self._init_head()

    def _init_head(self):
        """Initialize classification head weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224)

        Returns:
            Logits tensor of shape (batch, 4)
        """
        features = self.backbone(x)        # (batch, 1280)
        logits = self.classifier(features)  # (batch, 4)
        return logits

    def get_backbone_params(self):
        """Get backbone parameters (for differential learning rate)."""
        return self.backbone.parameters()

    def get_head_params(self):
        """Get classification head parameters."""
        return self.classifier.parameters()

    def freeze_backbone(self):
        """Freeze all backbone parameters (for initial fine-tuning phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_model(config: dict) -> AlzheimerStagingModel:
    """
    Factory function to build the model from a config dictionary.

    Args:
        config: Dictionary with keys from config.yaml's 'model' section.

    Returns:
        AlzheimerStagingModel instance
    """
    model_cfg = config.get("model", config)
    head_cfg = model_cfg.get("head", {})

    return AlzheimerStagingModel(
        backbone_name=model_cfg.get("backbone", "efficientnet_b0"),
        num_classes=config.get("data", {}).get("num_classes", 4),
        pretrained=model_cfg.get("pretrained", True),
        hidden_dim=head_cfg.get("hidden_dim", 256),
        dropout_1=head_cfg.get("dropout_1", 0.4),
        dropout_2=head_cfg.get("dropout_2", 0.2),
    )


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": f"{total / 1e6:.2f}M",
        "trainable_millions": f"{trainable / 1e6:.2f}M",
    }
