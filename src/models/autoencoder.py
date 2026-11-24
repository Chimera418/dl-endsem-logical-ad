"""EfficientAD / PUAD Global AutoEncoder (1x1 spatial bottleneck, reproduced verbatim from Assignment 2)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Compresses (B, 3, 256, 256) -> (B, 64, 1, 1) global contextual code."""

    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        return self.enconv6(x)


class Decoder(nn.Module):
    """Reconstructs teacher feature grid (56x56 for original PDN-S, or configurable target_size)."""

    def __init__(
        self,
        out_channels: int,
        img_size: int = 256,
        padding: bool = False,
        dropout_p: float = 0.2,
        target_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.last_upsample = target_size if target_size is not None else (
            img_size // 4 if padding else img_size // 4 - 8
        )
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)
        self.dropout4 = nn.Dropout(p=dropout_p)
        self.dropout5 = nn.Dropout(p=dropout_p)
        self.dropout6 = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.img_size // 64 - 1, mode="bilinear")
        x = self.dropout1(F.relu(self.deconv1(x)))
        x = F.interpolate(x, size=self.img_size // 32, mode="bilinear")
        x = self.dropout2(F.relu(self.deconv2(x)))
        x = F.interpolate(x, size=self.img_size // 16 - 1, mode="bilinear")
        x = self.dropout3(F.relu(self.deconv3(x)))
        x = F.interpolate(x, size=self.img_size // 8, mode="bilinear")
        x = self.dropout4(F.relu(self.deconv4(x)))
        x = F.interpolate(x, size=self.img_size // 4 - 1, mode="bilinear")
        x = self.dropout5(F.relu(self.deconv5(x)))
        x = F.interpolate(x, size=self.img_size // 2 - 1, mode="bilinear")
        x = self.dropout6(F.relu(self.deconv6(x)))
        x = F.interpolate(x, size=self.last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        return self.deconv8(x)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 384,
        img_size: int = 256,
        padding: bool = False,
        dropout_p: float = 0.2,
        target_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(
            out_channels=out_channels,
            img_size=img_size,
            padding=padding,
            dropout_p=dropout_p,
            target_size=target_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
