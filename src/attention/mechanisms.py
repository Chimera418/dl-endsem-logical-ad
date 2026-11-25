"""Phase C: Complete Suite of 10 Attention Mechanisms for Logical Anomaly Detection.

Every module preserves tensor shape: (B, C, H, W) -> (B, C, H, W).
Insertion point in winning PDN-S Student: right after conv3+ReLU at (B, 256, 59, 59).

Mechanisms implemented:
  1. SE          - Squeeze-and-Excitation (Hu et al., CVPR 2018)
  2. ECA         - Efficient Channel Attention (Wang et al., CVPR 2020)
  3. CBAM        - Convolutional Block Attention Module (Woo et al., ECCV 2018)
  4. CoordAtt    - Coordinate Attention (Hou et al., CVPR 2021)
  5. Triplet     - Triplet Attention (Misra et al., WACV 2021)
  6. SimAM       - Parameter-Free 3D Energy Attention (Yang et al., ICML 2021)
  7. GCT         - Gated Channel Transformation (Yang et al., CVPR 2020)
  8. BAM         - Bottleneck Attention Module (Park et al., BMVC 2018)
  9. NonLocal    - Embedded Gaussian Non-Local Self-Attention (Wang et al., CVPR 2018)
 10. CrissCross  - Recurrent Criss-Cross Attention, R=2 (Huang et al., ICCV 2019)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# 1. Squeeze-and-Excitation (SE)
# --------------------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


# --------------------------------------------------------------------------------------
# 2. Efficient Channel Attention (ECA)
# --------------------------------------------------------------------------------------
class ECABlock(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = torch.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y


# --------------------------------------------------------------------------------------
# 3. Convolutional Block Attention Module (CBAM)
# --------------------------------------------------------------------------------------
class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sequential Channel Attention (dual Avg + Max)
        c_avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        c_max = self.mlp(F.adaptive_max_pool2d(x, 1))
        x = x * torch.sigmoid(c_avg + c_max)

        # Sequential Spatial Attention (7x7 over channel Avg + Max)
        s_avg = torch.mean(x, dim=1, keepdim=True)
        s_max = torch.max(x, dim=1, keepdim=True)[0]
        s_attn = torch.sigmoid(self.spatial_conv(torch.cat([s_avg, s_max], dim=1)))
        return x * s_attn


# --------------------------------------------------------------------------------------
# 4. Coordinate Attention (CoordAtt)
# --------------------------------------------------------------------------------------
class CoordAttBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.conv1 = nn.Conv2d(channels, mid, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))                  # (B, C, H, 1)
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)  # (B, C, W, 1)

        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


# --------------------------------------------------------------------------------------
# 5. Triplet Attention
# --------------------------------------------------------------------------------------
class _ZPoolGate(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([torch.max(x, dim=1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(z))


class TripletAttentionBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        self.branch_cw = _ZPoolGate(kernel_size)
        self.branch_ch = _ZPoolGate(kernel_size)
        self.branch_hw = _ZPoolGate(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1: C x W interaction (permute H to channel dim)
        x_cw = self.branch_cw(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        # Branch 2: C x H interaction (permute W to channel dim)
        x_ch = self.branch_ch(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        # Branch 3: H x W spatial interaction
        x_hw = self.branch_hw(x)
        return (x_cw + x_ch + x_hw) / 3.0


# --------------------------------------------------------------------------------------
# 6. SimAM (Parameter-Free 3D Energy Attention)
# --------------------------------------------------------------------------------------
class SimAMBlock(nn.Module):
    def __init__(self, channels: int = 256, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w - 1
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / max(n, 1)
        inv_energy = x_minus_mu_sq / (4.0 * (v + self.e_lambda)) + 0.5
        return x * torch.sigmoid(inv_energy)


# --------------------------------------------------------------------------------------
# 7. Gated Channel Transformation (GCT)
# --------------------------------------------------------------------------------------
class GCTBlock(nn.Module):
    def __init__(self, channels: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        gate = 1.0 + torch.tanh(embedding * norm + self.beta)
        return x * gate


# --------------------------------------------------------------------------------------
# 8. Bottleneck Attention Module (BAM)
# --------------------------------------------------------------------------------------
class BAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, dilation: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = torch.sigmoid(self.channel_att(x) + self.spatial_att(x))
        return x * (1.0 + att)


# --------------------------------------------------------------------------------------
# 9. Non-Local Self-Attention (Wang et al., CVPR 2018 — Embedded Gaussian w/ 2x subsampling)
# --------------------------------------------------------------------------------------
class NonLocalBlock(nn.Module):
    def __init__(self, channels: int, sub_sample: bool = True) -> None:
        super().__init__()
        self.inter_channels = max(channels // 2, 16)
        self.theta = nn.Conv2d(channels, self.inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(channels, self.inter_channels, kernel_size=1, bias=False)
        self.g = nn.Conv2d(channels, self.inter_channels, kernel_size=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if sub_sample else nn.Identity()
        self.w_out = nn.Conv2d(self.inter_channels, channels, kernel_size=1, bias=True)
        # Zero-init residual projection per paper so training starts cleanly at identity
        nn.init.zeros_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.theta(x).view(b, self.inter_channels, -1).transpose(1, 2)           # (B, HW, C')
        k = self.pool(self.phi(x)).view(b, self.inter_channels, -1)                 # (B, C', HW/4)
        v = self.pool(self.g(x)).view(b, self.inter_channels, -1).transpose(1, 2)   # (B, HW/4, C')

        attn = torch.matmul(q, k) * (self.inter_channels ** -0.5)
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, self.inter_channels, h, w)
        return x + self.w_out(y)


# --------------------------------------------------------------------------------------
# 10. Criss-Cross Attention (Huang et al., ICCV 2019 — Recurrent R=2)
# --------------------------------------------------------------------------------------
class _CrissCrossOnce(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        inter = max(channels // 8, 16)
        self.query_conv = nn.Conv2d(channels, inter, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(channels, inter, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query_conv(x)  # (B, C', H, W)
        k = self.key_conv(x)
        v = self.value_conv(x)

        # Horizontal + vertical strip affinities
        q_h = q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)  # (B*W, H, C')
        q_w = q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)  # (B*H, W, C')
        k_h = k.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)                   # (B*W, C', H)
        k_w = k.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)                   # (B*H, C', W)
        v_h = v.permute(0, 3, 1, 2).contiguous().view(b * w, c, h)                    # (B*W, C, H)
        v_w = v.permute(0, 2, 1, 3).contiguous().view(b * h, c, w)                    # (B*H, C, W)

        scale = q.shape[1] ** -0.5
        e_h = torch.bmm(q_h, k_h).view(b, w, h, h).permute(0, 2, 1, 3) * scale        # (B, H, W, H)
        # Mask diagonal in e_h so center pixel isn't double-counted between H and W
        diag_mask = torch.eye(h, device=x.device, dtype=torch.bool).view(1, h, 1, h)
        e_h = e_h.masked_fill(diag_mask, float("-inf"))
        e_w = torch.bmm(q_w, k_w).view(b, h, w, w) * scale                            # (B, H, W, W)

        concate = F.softmax(torch.cat([e_h, e_w], dim=-1), dim=-1)                    # (B, H, W, H+W)
        att_h = concate[:, :, :, :h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
        att_w = concate[:, :, :, h:].contiguous().view(b * h, w, w)

        out_h = torch.bmm(v_h, att_h.permute(0, 2, 1)).view(b, w, c, h).permute(0, 2, 3, 1)
        out_w = torch.bmm(v_w, att_w.permute(0, 2, 1)).view(b, h, c, w).permute(0, 2, 1, 3)
        return self.gamma * (out_h + out_w) + x


class CrissCrossBlock(nn.Module):
    """Recurrent Criss-Cross Attention (R=2 passes -> full global HxW receptive field)."""

    def __init__(self, channels: int, recurrence: int = 2) -> None:
        super().__init__()
        self.recurrence = recurrence
        self.cca = _CrissCrossOnce(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.recurrence):
            x = self.cca(x)
        return x


ATTENTION_REGISTRY = {
    "se": SEBlock,
    "eca": ECABlock,
    "cbam": CBAMBlock,
    "coordatt": CoordAttBlock,
    "triplet": TripletAttentionBlock,
    "simam": SimAMBlock,
    "gct": GCTBlock,
    "bam": BAMBlock,
    "nonlocal": NonLocalBlock,
    "crisscross": CrissCrossBlock,
}


def build_attention(name: str, channels: int = 256) -> nn.Module:
    key = name.lower()
    if key not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention {name!r}; expected one of {list(ATTENTION_REGISTRY)}")
    return ATTENTION_REGISTRY[key](channels)
