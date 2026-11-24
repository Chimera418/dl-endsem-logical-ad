"""Stage 2 PUAD scoring: EfficientAD anomaly map + Ledoit-Wolf Mahalanobis + validation z-score fusion."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from src.data.loco import (
    CATEGORIES,
    CombinedNormalImages,
    CombinedTestImages,
    build_combined_logical_test_index,
    build_imagenet_normalization,
)
from src.evaluation.metrics import evaluate_logical_scores
from src.models.puad_system import PUADSystem


class EfficientADMapScorer:
    """Algorithm 2 of EfficientAD / PUAD: quantile-scaled map score + student feature map."""

    def __init__(
        self,
        system: PUADSystem,
        teacher_mu: torch.Tensor,
        teacher_sigma: torch.Tensor,
        quantiles: Dict[str, float],
        device: str = "cuda",
    ) -> None:
        self.system = system
        self.teacher = system.teacher.eval()
        self.student = system.student.eval()
        self.autoencoder = system.autoencoder.eval()
        self.normalize = Normalize(teacher_mu.tolist(), teacher_sigma.tolist())
        self.q = quantiles
        self.out_channels = system.out_channels
        self.img_size = system.img_size
        self.device = device

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> Tuple[float, torch.Tensor]:
        img = img.to(self.device)
        t_out = self.normalize(self.teacher(img).float())
        s_out = self.student(img).float()
        ae_out = self.autoencoder(img).float()

        s_st = s_out[:, : self.out_channels]
        s_ae = s_out[:, self.out_channels :]

        s_map = torch.mean((t_out - s_st) ** 2, dim=1, keepdim=True)
        a_map = torch.mean((ae_out - s_ae) ** 2, dim=1, keepdim=True)
        s_map = F.interpolate(s_map, size=(self.img_size, self.img_size), mode="bilinear").squeeze()
        a_map = F.interpolate(a_map, size=(self.img_size, self.img_size), mode="bilinear").squeeze()

        s_denom = max(self.q["q_b_student"] - self.q["q_a_student"], 1e-12)
        a_denom = max(self.q["q_b_autoencoder"] - self.q["q_a_autoencoder"], 1e-12)
        s_map = 0.1 * (s_map - self.q["q_a_student"]) / s_denom
        a_map = 0.1 * (a_map - self.q["q_a_autoencoder"]) / a_denom

        anomaly_map = 0.5 * s_map + 0.5 * a_map
        return float(torch.max(anomaly_map).item()), s_st


class MahalanobisHead:
    """PUAD unpicturable branch: Ledoit-Wolf shrunk covariance + pseudoinverse Mahalanobis distance."""

    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray) -> None:
        self.mean = features.mean(axis=0)
        cov = LedoitWolf().fit(features).covariance_
        self.cov_inv = np.linalg.pinv(cov)

    def score(self, feature: np.ndarray) -> float:
        assert self.mean is not None and self.cov_inv is not None
        centered = feature - self.mean
        return float(np.sqrt(max(0.0, float(centered @ self.cov_inv @ centered))))


def gap(features: torch.Tensor) -> np.ndarray:
    """Global Average Pooling over spatial dimensions (1, C, H, W) -> (C,)."""
    return torch.mean(features, dim=(0, 2, 3)).detach().cpu().numpy()


def fit_and_evaluate_puad(
    system: PUADSystem,
    teacher_mu: torch.Tensor,
    teacher_sigma: torch.Tensor,
    quantiles: Dict[str, float],
    dataset_root: str | Path,
    img_size: int,
    device: str,
    scores_csv_path: str | Path,
    reference_puad: Optional[Dict[str, Any]] = None,
    categories: List[str] = CATEGORIES,
) -> Tuple[Dict[str, Any], EfficientADMapScorer, MahalanobisHead]:
    """Fit PUAD Mahalanobis on train/good, calibrate z-stats on val/good, and score test/ once."""
    scorer = EfficientADMapScorer(system, teacher_mu, teacher_sigma, quantiles, device=device)
    normalize = build_imagenet_normalization()

    t0 = time.time()
    # 1. Fit Gaussian density strictly on pooled train/good (streamed to save RAM)
    print("  [Stage 2] Extracting pooled train/good features for Mahalanobis fit...", flush=True)
    train_ds = CombinedNormalImages(
        dataset_root, "train", img_size, normalize, augment=False, categories=categories, cache_in_ram=False
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=2)
    head = MahalanobisHead()
    train_features = np.stack([gap(scorer(img)[1]) for img in train_loader])
    head.fit(train_features)

    # 2. Calibrate z-score statistics strictly on pooled validation/good
    print("  [Stage 2] Calibrating z-score statistics on pooled validation/good...", flush=True)
    val_ds = CombinedNormalImages(
        dataset_root, "validation", img_size, normalize, augment=False, categories=categories, cache_in_ram=False
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    val_ea, val_maha = [], []
    for img in val_loader:
        ea_s, feats = scorer(img)
        val_ea.append(ea_s)
        val_maha.append(head.score(gap(feats)))

    ea_mean, ea_std = float(np.mean(val_ea)), float(max(np.std(val_ea), 1e-12))
    mh_mean, mh_std = float(np.mean(val_maha)), float(max(np.std(val_maha), 1e-12))
    fit_wall_s = time.time() - t0

    # 3. Single pass over logical-only test set (structural excluded by tripwire)
    print("  [Stage 2] Running single-pass evaluation on logical-only test set (1,136 images)...", flush=True)
    test_items = build_combined_logical_test_index(dataset_root, categories=categories)
    test_ds = CombinedTestImages(test_items, img_size, normalize)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    rows, puad_scores, ea_scores, maha_scores, labels, types, cats = [], [], [], [], [], [], []
    for (img, label, anomaly_type, cat), item in zip(test_loader, test_items):
        ea_s, feats = scorer(img)
        maha_s = head.score(gap(feats))
        puad_s = (ea_s - ea_mean) / ea_std + (maha_s - mh_mean) / mh_std

        label_i = int(label.item())
        type_s = anomaly_type[0] if isinstance(anomaly_type, (list, tuple)) else str(anomaly_type)
        cat_s = cat[0] if isinstance(cat, (list, tuple)) else str(cat)

        puad_scores.append(puad_s)
        ea_scores.append(ea_s)
        maha_scores.append(maha_s)
        labels.append(label_i)
        types.append(type_s)
        cats.append(cat_s)
        rows.append(
            {
                "unique_name": item.unique_name,
                "category": cat_s,
                "label": label_i,
                "anomaly_type": type_s,
                "score": puad_s,
                "efficientad_score": ea_s,
                "mahalanobis": maha_s,
            }
        )

    csv_p = Path(scores_csv_path)
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_p, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = evaluate_logical_scores(
        scores=puad_scores,
        labels=labels,
        anomaly_types=types,
        categories=cats,
        reference_puad=reference_puad,
        sub_scores={"efficientad_only": ea_scores, "mahalanobis_only": maha_scores},
    )
    metrics["calibration"] = {
        "efficientad_mean": ea_mean,
        "efficientad_std": ea_std,
        "mahalanobis_mean": mh_mean,
        "mahalanobis_std": mh_std,
        "n_fit_images": int(len(train_features)),
        "feature_dim": int(train_features.shape[1]),
        "n_calibration_images": int(len(val_ea)),
        "fit_split": "ALL categories train/good (pooled)",
        "calibration_split": "ALL categories validation/good (pooled)",
        "puad_stage_wall_clock_s": round(fit_wall_s, 3),
    }
    return metrics, scorer, head
