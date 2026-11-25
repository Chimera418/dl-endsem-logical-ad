"""Unified PUAD model system container & factory (supports E01 Original PUAD and future Phase B-E models)."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from src.attention.mechanisms import build_attention
from src.backbones.pdn import EnhancedPDN_S, build_pdn, load_pdn_teacher_state
from src.backbones.timm_backbones import TimmStudentBackbone, TimmTeacherExtractor
from src.models.autoencoder import AutoEncoder


class PUADSystem(nn.Module):
    """Wraps (teacher, student, autoencoder) behind a uniform interface for training, scoring, and profiling."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        autoencoder: nn.Module,
        out_channels: int = 384,
        img_size: int = 256,
        feature_grid_size: int = 56,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder
        self.out_channels = out_channels
        self.img_size = img_size
        self.feature_grid_size = feature_grid_size

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def trainable_parameters(self):
        return list(self.student.parameters()) + list(self.autoencoder.parameters())

    def forward_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            t_out = self.teacher(x)
        s_out = self.student(x)
        ae_out = self.autoencoder(x)
        return t_out, s_out, ae_out


def build_puad_system(cfg: Dict[str, Any], device: str = "cuda", load_teacher_weights: bool = True) -> PUADSystem:
    """Instantiate a PUADSystem from experiment config."""
    model_cfg = cfg["model"]
    mtype = model_cfg["type"]
    out_channels = int(model_cfg.get("out_channels", cfg.get("out_channels", 384)))
    img_size = int(cfg.get("img_size", 256))
    dropout_p = float(model_cfg.get("dropout_p", cfg["training"].get("dropout_p", 0.2)))

    if mtype == "original_puad":
        pdn_size = model_cfg.get("pdn_size", "s")
        padding = bool(model_cfg.get("padding", False))
        teacher = build_pdn(pdn_size, out_channels, padding=padding)
        if load_teacher_weights:
            teacher_path = cfg["paths"]["teacher_pdn_s"]
            if not os.path.exists(teacher_path):
                raise FileNotFoundError(
                    f"Shared PDN-S teacher checkpoint not found: {teacher_path}. "
                    "Set paths.teacher_pdn_s or ENDSEM_TEACHER_PATH."
                )
            state = load_pdn_teacher_state(teacher_path, pdn_size=pdn_size, out_channels=out_channels)
            teacher.load_state_dict(state, strict=True)

        student = build_pdn(pdn_size, out_channels * 2, padding=padding)
        autoencoder = AutoEncoder(
            out_channels=out_channels,
            img_size=img_size,
            padding=padding,
            dropout_p=dropout_p,
        )
        grid_size = img_size // 4 if padding else img_size // 4 - 8
        return PUADSystem(
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            out_channels=out_channels,
            img_size=img_size,
            feature_grid_size=grid_size,
        ).to(device)

    if mtype == "enhanced_puad":
        pdn_size = model_cfg.get("pdn_size", "s")
        padding = bool(model_cfg.get("padding", False))
        teacher = build_pdn(pdn_size, out_channels, padding=padding)
        if load_teacher_weights:
            teacher_path = cfg["paths"]["teacher_pdn_s"]
            if not os.path.exists(teacher_path):
                raise FileNotFoundError(
                    f"Shared PDN-S teacher checkpoint not found: {teacher_path}. "
                    "Set paths.teacher_pdn_s or ENDSEM_TEACHER_PATH."
                )
            state = load_pdn_teacher_state(teacher_path, pdn_size=pdn_size, out_channels=out_channels)
            teacher.load_state_dict(state, strict=True)

        attn_name = model_cfg.get("attention")
        attn_mod = build_attention(attn_name, channels=256) if attn_name else None

        student = EnhancedPDN_S(
            out_channels=out_channels * 2,
            padding=padding,
            attention=attn_mod,
        )
        autoencoder = AutoEncoder(
            out_channels=out_channels,
            img_size=img_size,
            padding=padding,
            dropout_p=dropout_p,
        )
        grid_size = img_size // 4 if padding else img_size // 4 - 8
        return PUADSystem(
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            out_channels=out_channels,
            img_size=img_size,
            feature_grid_size=grid_size,
        ).to(device)

    if mtype == "backbone_puad":
        backbone_name = model_cfg["backbone"]
        tap_strides = tuple(model_cfg.get("tap_strides", [8, 16, 32]))
        target_grid = int(model_cfg.get("target_grid", 32))

        teacher = TimmTeacherExtractor(
            backbone_name=backbone_name,
            out_channels=out_channels,
            tap_strides=tap_strides,
            target_grid=target_grid,
            pretrained=load_teacher_weights,
        )
        student = TimmStudentBackbone(
            backbone_name=backbone_name,
            out_channels=out_channels,
            tap_strides=tap_strides,
            target_grid=target_grid,
        )
        autoencoder = AutoEncoder(
            out_channels=out_channels,
            img_size=img_size,
            padding=False,
            dropout_p=dropout_p,
            target_size=target_grid,
        )
        return PUADSystem(
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            out_channels=out_channels,
            img_size=img_size,
            feature_grid_size=target_grid,
        ).to(device)

    raise NotImplementedError(
        f"Model type {mtype!r} is not yet registered."
    )
