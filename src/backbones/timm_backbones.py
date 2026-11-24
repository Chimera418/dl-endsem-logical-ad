"""Phase B Swappable Backbone Extractors for PUAD (ResNet-18, DenseNet-121, EfficientNet-B0, ConvNeXt-Tiny).

Every backbone is exposed through a uniform stride-indexed multi-scale interface at 256x256:
    C2 (stride  4): 64x64
    C3 (stride  8): 32x32
    C4 (stride 16): 16x16
    C5 (stride 32):  8x8

Teacher (frozen, ImageNet-pretrained):
    Taps {C3, C4, C5}, bilinearly aligns to 32x32, concatenates along channels,
    and applies parameter-free channel adaptive average pooling to `out_channels=384`
    (faithful to EfficientAD's PretrainedFeatureExtractor, with zero random teacher weights).

Student (trainable, randomly initialized per Uninformed Students / EfficientAD):
    Uses the same backbone family to extract multi-scale maps, with clean extension hooks for:
      - Phase C: Attention modules inserted on tapped feature stages
      - Phase D: Multi-scale feature fusion (FPN / BiFPN) over {C2, C3, C4, C5}
      - Phase E: Global contextual reasoning (Transformer / Mamba) on the fused spatial grid
    Followed by a 1x1 projection head outputting (B, 2*384, 32, 32) -> (s_st, s_ae).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_BACKBONES: Dict[str, str] = {
    "resnet18": "resnet18",
    "densenet121": "densenet121",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
}


def resolve_stride_indices(timm_name: str, target_strides: Sequence[int]) -> Tuple[int, ...]:
    """Map desired stride reductions (e.g. (8, 16, 32)) to exact timm feature_info indices."""
    probe = timm.create_model(timm_name, pretrained=False, features_only=True)
    reductions = probe.feature_info.reduction()
    indices: List[int] = []
    for s in target_strides:
        matches = [i for i, r in enumerate(reductions) if r == s]
        if not matches:
            raise ValueError(
                f"Backbone '{timm_name}' has reductions {reductions}; stride {s} not found."
            )
        # If multiple stages share a stride, take the deepest (richest) one.
        indices.append(matches[-1])
    return tuple(indices)


class TimmTeacherExtractor(nn.Module):
    """Frozen ImageNet-pretrained multi-scale teacher outputting (B, out_channels, 32, 32)."""

    def __init__(
        self,
        backbone_name: str,
        out_channels: int = 384,
        tap_strides: Sequence[int] = (8, 16, 32),
        target_grid: int = 32,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone {backbone_name!r}; choose from {list(SUPPORTED_BACKBONES)}")
        timm_name = SUPPORTED_BACKBONES[backbone_name]
        out_indices = resolve_stride_indices(timm_name, tap_strides)

        self.backbone_name = backbone_name
        self.out_channels = out_channels
        self.target_grid = target_grid
        self.tap_strides = tuple(tap_strides)
        self.encoder = timm.create_model(
            timm_name, pretrained=pretrained, features_only=True, out_indices=out_indices
        )
        self.stage_channels: List[int] = list(self.encoder.feature_info.channels())
        self.total_concat_channels = sum(self.stage_channels)
        if self.total_concat_channels < out_channels:
            raise ValueError(
                f"Teacher concat channels ({self.total_concat_channels}) < out_channels ({out_channels}) "
                f"for {backbone_name} with strides {tap_strides}."
            )

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        aligned = [
            F.interpolate(f, size=(self.target_grid, self.target_grid), mode="bilinear", align_corners=False)
            if f.shape[-1] != self.target_grid or f.shape[-2] != self.target_grid
            else f
            for f in feats
        ]
        cat = torch.cat(aligned, dim=1)  # (B, C_cat, H, W)
        b, c, h, w = cat.shape
        pooled = F.adaptive_avg_pool1d(cat.reshape(b, c, h * w).transpose(1, 2), self.out_channels)
        return pooled.transpose(1, 2).reshape(b, self.out_channels, h, w)


class TimmStudentBackbone(nn.Module):
    """Trainable swappable backbone student outputting (B, 2*out_channels, 32, 32).

    Designed with explicit modular hooks for:
      - `stage_attentions`: Optional nn.ModuleList applied to each tapped stage (Phase C)
      - `fusion_neck`: Optional FPN / BiFPN module over {C2, C3, C4, C5} (Phase D)
      - `context_module`: Optional Transformer / Mamba global context block (Phase E)
    """

    def __init__(
        self,
        backbone_name: str,
        out_channels: int = 384,
        tap_strides: Sequence[int] = (8, 16, 32),
        target_grid: int = 32,
        stage_attentions: Optional[nn.ModuleList] = None,
        fusion_neck: Optional[nn.Module] = None,
        context_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone {backbone_name!r}; choose from {list(SUPPORTED_BACKBONES)}")
        timm_name = SUPPORTED_BACKBONES[backbone_name]
        out_indices = resolve_stride_indices(timm_name, tap_strides)

        self.backbone_name = backbone_name
        self.out_channels = out_channels
        self.target_grid = target_grid
        self.tap_strides = tuple(tap_strides)

        # Randomly initialized student (Uninformed Students / EfficientAD principle)
        self.encoder = timm.create_model(
            timm_name, pretrained=False, features_only=True, out_indices=out_indices
        )
        self.stage_channels: List[int] = list(self.encoder.feature_info.channels())
        self.stage_attentions = stage_attentions
        self.fusion_neck = fusion_neck
        self.context_module = context_module

        if fusion_neck is not None:
            fused_channels = int(fusion_neck.out_channels)
        else:
            fused_channels = sum(self.stage_channels)

        # Lightweight 1x1 projection to unified 384-d intermediate + dual 768-d (s_st, s_ae) head
        self.proj_in = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(out_channels, out_channels * 2, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = list(self.encoder(x))

        if self.stage_attentions is not None:
            feats = [attn(f) for attn, f in zip(self.stage_attentions, feats)]

        if self.fusion_neck is not None:
            fused = self.fusion_neck(feats, target_size=self.target_grid)
        else:
            aligned = [
                F.interpolate(f, size=(self.target_grid, self.target_grid), mode="bilinear", align_corners=False)
                if f.shape[-1] != self.target_grid or f.shape[-2] != self.target_grid
                else f
                for f in feats
            ]
            fused = torch.cat(aligned, dim=1)

        h = self.proj_in(fused)
        if self.context_module is not None:
            h = self.context_module(h)
        return self.head(h)
