"""Phase D — Multi-Scale Feature Pyramid Fusion (FPN and BiFPN) for EnhancedPDN_S.

Design principles informed by Phase B & C findings:
  1. Zero BatchNorm layers inside the fusion neck (BatchNorm inside PDN-S Student
     destabilized hard-mining MSE + ImageNet OOD penalty in E02 & E09).
  2. True 4-level multi-scale pyramid constructed from PDN-S intermediate taps:
       - P2: (B, fpn_c, 126, 126) from s1 (128ch, stride 2) — fine local detail
       - P3: (B, fpn_c,  59,  59) from s2 (61x61) + s3 (59x59) — native PDN-S output grid
       - P4: (B, fpn_c,  30,  30) via stride-2 conv on P3 — mid-range object relationships
       - P5: (B, fpn_c,  15,  15) via stride-2 conv on P4 — macro scene layout
  3. Zero-initialized learnable residual gate `gamma` on the fused 59x59 projection:
       out = s3 + gamma * proj(fused_p2_p3_p4_p5)
     so training starts smoothly at the exact PDN-S representation with stable gradients.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DepthwiseSeparableConv(nn.Module):
    """BN-free depthwise separable 3x3 conv + SiLU (used in BiFPN fusion nodes)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pw(self.dw(x)))


class _FastNormalizedAdd(nn.Module):
    """EfficientDet fast normalized fusion: sum_i (ReLU(w_i) / (sum_j ReLU(w_j) + eps)) * x_i."""

    def __init__(self, num_inputs: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = eps

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        out = w[0] * inputs[0]
        for i in range(1, len(inputs)):
            out = out + w[i] * inputs[i]
        return out


class FPNFusion(nn.Module):
    """E16: Standard unidirectional top-down Feature Pyramid Network (Lin et al., CVPR 2017).

    Inputs: [s1 (B, 128, 126, 126), s2 (B, 256, 61, 61), s3 (B, 256, 59, 59)]
    Output: (B, 256, 59, 59) fused feature map with residual connection to s3.
    """

    def __init__(self, in_channels: List[int] = (128, 256, 256), fpn_channels: int = 128, out_channels: int = 256) -> None:
        super().__init__()
        c1, c2, c3 = in_channels
        # Lateral 1x1 projections into unified pyramid width
        self.lat2 = nn.Conv2d(c1, fpn_channels, kernel_size=1)
        self.lat3_s2 = nn.Conv2d(c2, fpn_channels, kernel_size=1)
        self.lat3_s3 = nn.Conv2d(c3, fpn_channels, kernel_size=1)

        # Coarser semantic pyramid levels (P4: 30x30, P5: 15x15)
        self.down4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1)
        self.down5 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1)

        # Top-down 3x3 smoothing convolutions (no BN)
        self.smooth5 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)

        # Multi-scale aggregation onto 59x59 + residual projection to 256ch
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(fpn_channels * 4, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        s1, s2, s3 = feats
        h3, w3 = s3.shape[-2:]

        # Build 4-level input pyramid
        c2 = self.lat2(s1)                                                                # (B, C, 126, 126)
        s2_aligned = F.interpolate(s2, size=(h3, w3), mode="bilinear", align_corners=False)
        c3 = self.lat3_s3(s3) + self.lat3_s2(s2_aligned)                                  # (B, C,  59,  59)
        c4 = F.relu(self.down4(c3))                                                       # (B, C,  30,  30)
        c5 = F.relu(self.down5(c4))                                                       # (B, C,  15,  15)

        # Unidirectional top-down pathway: P5 -> P4 -> P3 -> P2
        p5 = F.relu(self.smooth5(c5))
        p4 = F.relu(self.smooth4(c4 + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)))
        p3 = F.relu(self.smooth3(c3 + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)))
        p2 = F.relu(self.smooth2(c2 + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)))

        # Align all 4 refined scales onto the native 59x59 grid
        p2_a = F.interpolate(p2, size=(h3, w3), mode="bilinear", align_corners=False)
        p4_a = F.interpolate(p4, size=(h3, w3), mode="bilinear", align_corners=False)
        p5_a = F.interpolate(p5, size=(h3, w3), mode="bilinear", align_corners=False)

        fused = torch.cat([p2_a, p3, p4_a, p5_a], dim=1)
        return s3 + self.gamma * self.fuse_proj(fused)


class BiFPNFusion(nn.Module):
    """E17: Weighted Bi-Directional Feature Pyramid Network (Tan et al., CVPR 2020 / EfficientDet).

    Performs fast-normalized weighted top-down AND bottom-up cross-scale fusion with
    same-level skip connections over the exact same 4-level pyramid [P2, P3, P4, P5] as E16_FPN.
    """

    def __init__(self, in_channels: List[int] = (128, 256, 256), fpn_channels: int = 128, out_channels: int = 256) -> None:
        super().__init__()
        c1, c2, c3 = in_channels
        self.lat2 = nn.Conv2d(c1, fpn_channels, kernel_size=1)
        self.lat3_s2 = nn.Conv2d(c2, fpn_channels, kernel_size=1)
        self.lat3_s3 = nn.Conv2d(c3, fpn_channels, kernel_size=1)

        self.down4 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1)
        self.down5 = nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, stride=2, padding=1)

        # Top-down weighted fusion nodes (P5 -> P4_td -> P3_td -> P2_out)
        self.w_p4_td = _FastNormalizedAdd(2)
        self.conv_p4_td = _DepthwiseSeparableConv(fpn_channels)
        self.w_p3_td = _FastNormalizedAdd(2)
        self.conv_p3_td = _DepthwiseSeparableConv(fpn_channels)
        self.w_p2_out = _FastNormalizedAdd(2)
        self.conv_p2_out = _DepthwiseSeparableConv(fpn_channels)

        # Bottom-up weighted fusion nodes with same-level skip connections (P2_out -> P3_out -> P4_out -> P5_out)
        self.w_p3_out = _FastNormalizedAdd(3)
        self.conv_p3_out = _DepthwiseSeparableConv(fpn_channels)
        self.w_p4_out = _FastNormalizedAdd(3)
        self.conv_p4_out = _DepthwiseSeparableConv(fpn_channels)
        self.w_p5_out = _FastNormalizedAdd(2)
        self.conv_p5_out = _DepthwiseSeparableConv(fpn_channels)

        # Weighted 4-scale aggregation onto 59x59 + residual projection to 256ch
        self.w_agg = _FastNormalizedAdd(4)
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(fpn_channels, out_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        s1, s2, s3 = feats
        h3, w3 = s3.shape[-2:]

        # Build 4-level input pyramid
        p2_in = self.lat2(s1)                                                             # (B, C, 126, 126)
        s2_aligned = F.interpolate(s2, size=(h3, w3), mode="bilinear", align_corners=False)
        p3_in = self.lat3_s3(s3) + self.lat3_s2(s2_aligned)                               # (B, C,  59,  59)
        p4_in = F.silu(self.down4(p3_in))                                                 # (B, C,  30,  30)
        p5_in = F.silu(self.down5(p4_in))                                                 # (B, C,  15,  15)

        # Top-down pass
        p4_td = self.conv_p4_td(self.w_p4_td([
            p4_in,
            F.interpolate(p5_in, size=p4_in.shape[-2:], mode="bilinear", align_corners=False),
        ]))
        p3_td = self.conv_p3_td(self.w_p3_td([
            p3_in,
            F.interpolate(p4_td, size=p3_in.shape[-2:], mode="bilinear", align_corners=False),
        ]))
        p2_out = self.conv_p2_out(self.w_p2_out([
            p2_in,
            F.interpolate(p3_td, size=p2_in.shape[-2:], mode="bilinear", align_corners=False),
        ]))

        # Bottom-up pass with same-level skips
        p3_out = self.conv_p3_out(self.w_p3_out([
            p3_in,
            p3_td,
            F.interpolate(p2_out, size=p3_in.shape[-2:], mode="bilinear", align_corners=False),
        ]))
        p4_out = self.conv_p4_out(self.w_p4_out([
            p4_in,
            p4_td,
            F.interpolate(p3_out, size=p4_in.shape[-2:], mode="bilinear", align_corners=False),
        ]))
        p5_out = self.conv_p5_out(self.w_p5_out([
            p5_in,
            F.interpolate(p4_out, size=p5_in.shape[-2:], mode="bilinear", align_corners=False),
        ]))

        # Weighted aggregation onto 59x59 grid
        agg = self.w_agg([
            F.interpolate(p2_out, size=(h3, w3), mode="bilinear", align_corners=False),
            p3_out,
            F.interpolate(p4_out, size=(h3, w3), mode="bilinear", align_corners=False),
            F.interpolate(p5_out, size=(h3, w3), mode="bilinear", align_corners=False),
        ])
        return s3 + self.gamma * self.fuse_proj(agg)


FUSION_REGISTRY = {
    "fpn": FPNFusion,
    "bifpn": BiFPNFusion,
}


def build_fusion(name: str, fpn_channels: int = 128, out_channels: int = 256) -> nn.Module:
    key = name.strip().lower()
    if key not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion {name!r}. Available: {sorted(FUSION_REGISTRY)}")
    return FUSION_REGISTRY[key](fpn_channels=fpn_channels, out_channels=out_channels)
