"""Stage 1 Unified Trainer for PUAD-family models.

Strict protocol guarantees:
  - Trains exclusively on pooled `train/good` (+ optional ImageNet L_penalty).
  - Evaluates validation loss strictly on pooled `validation/good` (`L_hard + L_ae + L_stae`).
  - Best checkpoint selection & early stopping use ONLY `val_loss` (test/ is never touched).
  - Supports BF16 / FP16 / FP32 mixed precision cleanly (stats & quantiles kept in FP32).
  - Saves resumable checkpoints every epoch (with optional Google Drive sync) for Colab Pro safety.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from src.common.env import (
    CsvLogger,
    EarlyStopping,
    capture_environment,
    cycle,
    resolve_amp_dtype,
    rng_state,
    save_config,
    save_json,
)
from src.common.profiling import count_parameters
from src.data.loco import (
    CATEGORIES,
    CombinedNormalImages,
    PenaltyImages,
    build_imagenet_normalization,
)
from src.models.puad_system import PUADSystem

LOG_FIELDS = [
    "epoch", "train_loss", "val_loss", "learning_rate",
    "train_loss_total", "train_loss_hard", "train_loss_penalty",
    "train_loss_ae", "train_loss_stae",
    "val_loss_hard", "val_loss_ae", "val_loss_stae",
    "iterations", "epoch_time_s", "is_best", "patience_counter",
]


@contextmanager
def preserve_bn_stats(module: torch.nn.Module):
    """Prevent out-of-distribution ImageNet L_penalty / augmented passes from corrupting LOCO BatchNorm running stats."""
    saved = []
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
            saved.append((
                m,
                m.running_mean.clone() if m.running_mean is not None else None,
                m.running_var.clone() if m.running_var is not None else None,
                m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None,
            ))
    try:
        yield
    finally:
        for m, rm, rv, nbt in saved:
            if rm is not None and m.running_mean is not None:
                m.running_mean.copy_(rm)
            if rv is not None and m.running_var is not None:
                m.running_var.copy_(rv)
            if nbt is not None and m.num_batches_tracked is not None:
                m.num_batches_tracked.copy_(nbt)


def hard_mining_loss(diff: torch.Tensor, quantile: float) -> torch.Tensor:
    """EfficientAD L_hard computed in FP32 for numerical stability with AMP."""
    diff_f32 = diff.float()
    threshold = torch.quantile(diff_f32, q=quantile)
    return torch.mean(diff_f32[diff_f32 >= threshold])


@torch.no_grad()
def streaming_teacher_stats(loader: DataLoader, teacher: torch.nn.Module, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Streaming per-channel (mu, sigma) over pooled train/good (zero VRAM blowup)."""
    teacher.eval()
    total = None
    total_sq = None
    n = 0
    for img in loader:
        out = teacher(img.to(device, non_blocking=True)).float()
        s = out.sum(dim=(0, 2, 3), dtype=torch.float64)
        sq = (out.double() ** 2).sum(dim=(0, 2, 3))
        total = s if total is None else total + s
        total_sq = sq if total_sq is None else total_sq + sq
        n += out.shape[0] * out.shape[2] * out.shape[3]
    mean = (total / n).float()
    # Floor at 1e-2 to prevent near-dead ReLU channels in pretrained ResNet/DenseNet/EfficientNet
    # from exploding standardized outliers under q=0.999 hard mining (no effect on PDN-S).
    std = (total_sq / n - (total / n) ** 2).clamp_min(1e-4).sqrt().clamp_min(1e-2).float()
    return mean, std


@torch.no_grad()
def evaluate_validation_loss(
    loader: DataLoader,
    system: PUADSystem,
    normalize_teacher: Normalize,
    quantile: float,
    device: str,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, float]:
    """Compute L_hard + L_ae + L_stae on pooled validation/good (eval mode, no augmentation)."""
    system.student.eval()
    system.autoencoder.eval()
    out_c = system.out_channels
    totals = {"val_loss_hard": 0.0, "val_loss_ae": 0.0, "val_loss_stae": 0.0}
    n = 0

    for img in loader:
        img = img.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            t_out = normalize_teacher(system.teacher(img).float())
            s_out = system.student(img).float()
            ae_out = system.autoencoder(img).float()

        totals["val_loss_hard"] += hard_mining_loss((t_out - s_out[:, :out_c]) ** 2, quantile).item()
        totals["val_loss_ae"] += torch.mean((t_out - ae_out) ** 2).item()
        totals["val_loss_stae"] += torch.mean((ae_out - s_out[:, out_c:]) ** 2).item()
        n += 1

    system.student.train()
    system.autoencoder.train()
    return {k: v / max(1, n) for k, v in totals.items()}


@torch.no_grad()
def compute_validation_quantiles(
    loader: DataLoader,
    system: PUADSystem,
    normalize_teacher: Normalize,
    q_a: float,
    q_b: float,
    device: str,
) -> Dict[str, float]:
    """Compute q_a / q_b anomaly-map calibration quantiles strictly on validation/good."""
    system.student.eval()
    system.autoencoder.eval()
    out_c = system.out_channels
    img_size = system.img_size
    s_maps, a_maps = [], []

    for img in loader:
        img = img.to(device, non_blocking=True)
        t_out = normalize_teacher(system.teacher(img).float())
        s_out = system.student(img).float()
        ae_out = system.autoencoder(img).float()

        s_map = torch.mean((t_out - s_out[:, :out_c]) ** 2, dim=1, keepdim=True)
        a_map = torch.mean((ae_out - s_out[:, out_c:]) ** 2, dim=1, keepdim=True)
        s_maps.append(F.interpolate(s_map, size=(img_size, img_size), mode="bilinear").squeeze(1).cpu())
        a_maps.append(F.interpolate(a_map, size=(img_size, img_size), mode="bilinear").squeeze(1).cpu())

    s_all = torch.cat(s_maps, dim=0).numpy().ravel()
    a_all = torch.cat(a_maps, dim=0).numpy().ravel()
    return {
        "q_a_student": float(np.quantile(s_all, q_a)),
        "q_b_student": float(np.quantile(s_all, q_b)),
        "q_a_autoencoder": float(np.quantile(a_all, q_a)),
        "q_b_autoencoder": float(np.quantile(a_all, q_b)),
    }


def _sync_file(src: Path, sync_dir: Optional[Path]) -> None:
    if sync_dir is None or not src.exists():
        return
    sync_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, sync_dir / src.name)


def train_stage1_puad(
    system: PUADSystem,
    cfg: Dict[str, Any],
    out_dir: Path,
    ckpt_dir: Path,
    device: str = "cuda",
    force: bool = False,
    smoke: bool = False,
    drive_sync_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train Stage 1 (Student + AutoEncoder) on pooled MVTec LOCO train/good."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = ckpt_dir / "best_checkpoint.pth"
    resume_ckpt_path = ckpt_dir / "train_resume.pth"
    summary_path = out_dir / "training_summary.json"

    # Pull prior checkpoints/summaries from Google Drive if resuming on a fresh Colab VM
    if drive_sync_dir is not None and drive_sync_dir.is_dir() and not force:
        for fname, dst in (
            ("best_checkpoint.pth", best_ckpt_path),
            ("train_resume.pth", resume_ckpt_path),
            ("training_summary.json", summary_path),
            ("loss_history.csv", out_dir / "loss_history.csv"),
        ):
            src = drive_sync_dir / fname
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                print(f"  [drive-restore] Restored {fname} from {drive_sync_dir}", flush=True)

    if summary_path.exists() and best_ckpt_path.exists() and not force:
        import json
        print(f"[skip] Training already complete: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    if force:
        for stale in (best_ckpt_path, resume_ckpt_path):
            if stale.exists():
                stale.unlink()

    tr = cfg["training"]
    es_cfg = tr["early_stopping"]
    img_size = int(cfg["img_size"])
    out_c = system.out_channels
    categories = cfg.get("categories", CATEGORIES)
    batch_size = int(tr["batch_size"])
    epochs = 2 if smoke else int(tr["epochs"])
    max_iters = (min(30, 1778 // max(batch_size, 1)) if smoke else tr.get("max_iterations"))
    lr_decay_epoch = int(tr.get("lr_decay_epoch", max(1, int(epochs * 0.9))))
    quantile = float(tr["hard_mining_quantile"])
    use_penalty = bool(tr.get("use_penalty", True))
    amp_dtype = resolve_amp_dtype(tr.get("mixed_precision", "bf16"), device)
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    normalize = build_imagenet_normalization()
    dataset_root = cfg["paths"]["dataset_root"]

    cache_ram = bool(tr.get("cache_in_ram", not smoke))
    print(f"Loading pooled train/good and validation/good (img_size={img_size}, cache_in_ram={cache_ram})...", flush=True)
    train_ds = CombinedNormalImages(
        dataset_root, "train", img_size, normalize, augment=bool(tr.get("augment_ae_only", True)),
        categories=categories, cache_in_ram=cache_ram
    )
    val_ds = CombinedNormalImages(
        dataset_root, "validation", img_size, normalize, augment=False,
        categories=categories, cache_in_ram=cache_ram
    )
    plain_train_ds = CombinedNormalImages(
        dataset_root, "train", img_size, normalize, augment=False,
        categories=categories, cache_in_ram=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    plain_loader = DataLoader(plain_train_ds, batch_size=max(batch_size, 8), shuffle=False, num_workers=0)

    penalty_stream = None
    if use_penalty:
        imagenet_root = cfg["paths"]["imagenet_root"]
        penalty_ds = PenaltyImages(imagenet_root, normalize, img_size)
        n_workers = 0 if smoke else int(tr.get("penalty_workers", 2))
        penalty_loader = DataLoader(
            penalty_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(n_workers > 0),
            drop_last=True,
        )
        penalty_stream = cycle(penalty_loader)
        print(f"  Penalty corpus ready: {len(penalty_ds):,} images (batch_size={batch_size})")

    print("  Computing streaming teacher channel normalization stats on pooled train/good...")
    mu, sigma = streaming_teacher_stats(plain_loader, system.teacher, device)
    normalize_teacher = Normalize(mu.tolist(), sigma.tolist())

    opt_name = str(tr.get("optimizer", "adam")).lower()
    if opt_name == "adamw":
        optimizer = AdamW(system.trainable_parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    else:
        optimizer = Adam(system.trainable_parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))

    sched_name = str(tr.get("scheduler", "step")).lower()
    scheduler = (
        CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=float(tr["lr_after_decay"]))
        if sched_name == "cosine"
        else None
    )
    stopper = EarlyStopping(patience=int(es_cfg["patience"]), min_delta=float(es_cfg["min_delta"]))

    start_epoch = 0
    iteration = 0
    prior_wall = 0.0

    # Auto-resume if a Colab session disconnected mid-training
    if resume_ckpt_path.exists() and not force:
        st = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        system.student.load_state_dict(st["student"])
        system.autoencoder.load_state_dict(st["autoencoder"])
        optimizer.load_state_dict(st["optimizer"])
        if scheduler is not None and st.get("scheduler") is not None:
            scheduler.load_state_dict(st["scheduler"])
        start_epoch = int(st["epoch"])
        iteration = int(st["iteration"])
        prior_wall = float(st.get("wall_clock_s", 0.0))
        stopper.load_state_dict(st["early_stopping"])
        for g in optimizer.param_groups:
            g["lr"] = float(st["learning_rate"])
        print(f"  [resume] Restored epoch {start_epoch} (iter {iteration:,}, best val_loss {stopper.best_value:.4f})")

    log = CsvLogger(out_dir / "loss_history.csv", LOG_FIELDS, append=(start_epoch > 0))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(
        f"  Starting training: epochs={epochs}, batch_size={batch_size}, amp={amp_dtype}, "
        f"trainable_params={count_parameters(system, True):,}, frozen_params={count_parameters(system, False) - count_parameters(system, True):,}"
    )

    t_start = time.time()
    stop_reason = "max_epochs_reached"
    last_epoch = start_epoch

    for epoch in range(start_epoch + 1, epochs + 1):
        last_epoch = epoch
        t_ep = time.time()
        if scheduler is None and epoch == lr_decay_epoch:
            for g in optimizer.param_groups:
                g["lr"] = float(tr["lr_after_decay"])

        system.student.train()
        system.autoencoder.train()
        sums = {"hard": 0.0, "penalty": 0.0, "ae": 0.0, "stae": 0.0}
        steps = 0

        for batch in train_loader:
            if max_iters is not None and iteration >= max_iters:
                break
            iteration += 1
            if isinstance(batch, (list, tuple)):
                img, img_aug = batch
            else:
                img, img_aug = batch, batch

            img = img.to(device, non_blocking=True)
            img_aug = img_aug.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                with torch.no_grad():
                    t_out = normalize_teacher(system.teacher(img).float())
                    t_out_aug = normalize_teacher(system.teacher(img_aug).float())

                s_out = system.student(img)
                loss_hard = hard_mining_loss((t_out - s_out[:, :out_c].float()) ** 2, quantile)

                if penalty_stream is not None:
                    p_img = next(penalty_stream).to(device, non_blocking=True)
                    with preserve_bn_stats(system.student):
                        loss_penalty = torch.mean(system.student(p_img)[:, :out_c].float() ** 2)
                else:
                    loss_penalty = torch.tensor(0.0, device=device)

                ae_out = system.autoencoder(img_aug)
                with preserve_bn_stats(system.student):
                    s_out_aug = system.student(img_aug)
                loss_ae = torch.mean((t_out_aug - ae_out.float()) ** 2)
                loss_stae = torch.mean((ae_out.float() - s_out_aug[:, out_c:].float()) ** 2)

                loss_total = loss_hard + loss_penalty + loss_ae + loss_stae

            if use_scaler:
                scaler.scale(loss_total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                optimizer.step()

            sums["hard"] += float(loss_hard.item())
            sums["penalty"] += float(loss_penalty.item())
            sums["ae"] += float(loss_ae.item())
            sums["stae"] += float(loss_stae.item())
            steps += 1

        if steps == 0:
            break

        means = {k: v / steps for k, v in sums.items()}
        train_loss = means["hard"] + means["ae"] + means["stae"]
        train_total = train_loss + means["penalty"]

        val = evaluate_validation_loss(val_loader, system, normalize_teacher, quantile, device, amp_dtype)
        val_loss = val["val_loss_hard"] + val["val_loss_ae"] + val["val_loss_stae"]

        is_best = stopper.step(epoch, val_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()
        ep_s = time.time() - t_ep

        log.log(
            epoch=epoch,
            train_loss=f"{train_loss:.6f}",
            val_loss=f"{val_loss:.6f}",
            learning_rate=lr_now,
            train_loss_total=f"{train_total:.6f}",
            train_loss_hard=f"{means['hard']:.6f}",
            train_loss_penalty=f"{means['penalty']:.6f}",
            train_loss_ae=f"{means['ae']:.6f}",
            train_loss_stae=f"{means['stae']:.6f}",
            val_loss_hard=f"{val['val_loss_hard']:.6f}",
            val_loss_ae=f"{val['val_loss_ae']:.6f}",
            val_loss_stae=f"{val['val_loss_stae']:.6f}",
            iterations=iteration,
            epoch_time_s=f"{ep_s:.1f}",
            is_best=int(is_best),
            patience_counter=stopper.counter,
        )

        tag = " <-- BEST" if is_best else f" (patience {stopper.counter}/{stopper.patience})"
        print(
            f"  Epoch {epoch:>3}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"lr={lr_now:.1e} | {ep_s:.1f}s{tag}",
            flush=True,
        )

        if is_best:
            torch.save(
                {
                    "model_state_dict": {
                        "teacher": system.teacher.state_dict(),
                        "student": system.student.state_dict(),
                        "autoencoder": system.autoencoder.state_dict(),
                    },
                    "epoch": epoch,
                    "iteration": iteration,
                    "best_val_loss": val_loss,
                    "train_loss_at_best_epoch": train_loss,
                    "learning_rate": lr_now,
                    "dropout_p": float(tr.get("dropout_p", 0.2)),
                    "teacher_mu": mu.cpu(),
                    "teacher_sigma": sigma.cpu(),
                    "config": cfg,
                    "rng_state": rng_state(),
                },
                best_ckpt_path,
            )
            _sync_file(best_ckpt_path, drive_sync_dir)

        if bool(tr.get("save_resume_every_epoch", True)):
            torch.save(
                {
                    "student": system.student.state_dict(),
                    "autoencoder": system.autoencoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "epoch": epoch,
                    "iteration": iteration,
                    "wall_clock_s": prior_wall + (time.time() - t_start),
                    "early_stopping": stopper.state_dict(),
                },
                resume_ckpt_path,
            )
            _sync_file(resume_ckpt_path, drive_sync_dir)

        if stopper.should_stop:
            stop_reason = "early_stopping"
            print(f"  Early stopping triggered at epoch {epoch}.")
            break
        if max_iters is not None and iteration >= max_iters:
            stop_reason = "max_iterations_reached"
            break

    log.close()
    wall_s = prior_wall + (time.time() - t_start)
    peak_train_mib = (
        round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
        if device == "cuda"
        else None
    )

    # Restore strictly minimum-val-loss checkpoint & calibrate quantiles on validation/good
    best = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    system.student.load_state_dict(best["model_state_dict"]["student"])
    system.autoencoder.load_state_dict(best["model_state_dict"]["autoencoder"])
    q_cfg = cfg["evaluation"]["quantiles"]
    quantiles = compute_validation_quantiles(
        val_loader, system, normalize_teacher, float(q_cfg["q_a"]), float(q_cfg["q_b"]), device
    )
    best["quantiles"] = quantiles
    torch.save(best, best_ckpt_path)
    _sync_file(best_ckpt_path, drive_sync_dir)

    summary = {
        "exp_id": cfg["exp_id"],
        "smoke_test": bool(smoke),
        "batch_size": batch_size,
        "mixed_precision": str(amp_dtype) if amp_dtype is not None else "fp32",
        "epochs_configured": epochs,
        "epochs_run": last_epoch,
        "total_iterations_run": iteration,
        "stop_reason": stop_reason,
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["best_val_loss"]),
        "train_loss_at_best_epoch": float(best["train_loss_at_best_epoch"]),
        "wall_clock_s": round(wall_s, 2),
        "peak_training_gpu_memory_mib": peak_train_mib,
        "quantiles": quantiles,
        "early_stopping": stopper.summary(),
        "environment": capture_environment(str(out_dir)),
    }
    save_json(summary, summary_path)
    save_config(cfg, out_dir / "config_used.yaml")
    _sync_file(summary_path, drive_sync_dir)
    _sync_file(out_dir / "loss_history.csv", drive_sync_dir)

    if resume_ckpt_path.exists():
        resume_ckpt_path.unlink()
    return summary
