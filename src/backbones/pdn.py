"""Patch Description Networks (PDN-S, PDN-M) reproduced verbatim from Assignment 2 / LeapMind PUAD."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PDN_S(nn.Module):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * padding)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * padding)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * padding)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * padding)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * padding)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        return self.conv4(x)


class EnhancedPDN_S(PDN_S):
    """Phase C-E modular extension of the winning PDN-S Student backbone.

    Preserves the exact PDN-S conv1..conv4 architecture and adds optional plug-in hooks:
      - `attention`: inserted after conv3+ReLU at (B, 256, 59, 59) -> (B, 256, 59, 59)
      - `fusion`:    optional multi-scale FPN / BiFPN over {s1(126x126), s2(61x61), s3(59x59)}
      - `context`:   optional global Transformer / Mamba block at (B, 256, 59, 59)
    """

    def __init__(
        self,
        out_channels: int,
        padding: bool = False,
        attention: nn.Module | None = None,
        fusion: nn.Module | None = None,
        context: nn.Module | None = None,
    ) -> None:
        super().__init__(out_channels=out_channels, padding=padding)
        self.attention = attention
        self.fusion = fusion
        self.context = context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.avgpool1(F.relu(self.conv1(x)))  # (B, 128, 126, 126)
        s2 = self.avgpool2(F.relu(self.conv2(s1))) # (B, 256,  61,  61)
        s3 = F.relu(self.conv3(s2))                # (B, 256,  59,  59)

        h = self.fusion([s1, s2, s3]) if self.fusion is not None else s3
        if self.attention is not None:
            h = self.attention(h)
        if self.context is not None:
            h = self.context(h)
        return self.conv4(h)


class PDN_M(nn.Module):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * padding)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * padding)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * padding)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * padding)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * padding)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * padding)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * padding)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.conv6(x)


def build_pdn(size: str, out_channels: int, padding: bool = False) -> nn.Module:
    if size == "s":
        return PDN_S(out_channels=out_channels, padding=padding)
    if size == "m":
        return PDN_M(out_channels=out_channels, padding=padding)
    raise ValueError(f"pdn_size must be 's' or 'm', got {size!r}")


def load_pdn_teacher_state(checkpoint_path: str, pdn_size: str = "s", out_channels: int = 384) -> Dict[str, torch.Tensor]:
    """Load either a pre-remapped teacher_pdn_s.pt or raw sequential nelson1425 checkpoint."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "conv1.weight" in raw:
        return raw

    names = {
        "s": ["conv1", "conv2", "conv3", "conv4"],
        "m": ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6"],
    }[pdn_size]
    idx_keys = sorted({k.split(".")[0] for k in raw}, key=int)
    if len(idx_keys) != len(names):
        raise ValueError(f"Checkpoint layer count {len(idx_keys)} != expected {len(names)} for PDN_{pdn_size.upper()}")
    mapping = dict(zip(idx_keys, names))
    return {k.replace(k.split(".")[0], mapping[k.split(".")[0]], 1): v for k, v in raw.items()}
