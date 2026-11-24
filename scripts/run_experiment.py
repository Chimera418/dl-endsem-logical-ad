"""Unified experiment runner for Endsem Logical Anomaly Detection (E01..E19).

Modes:
  - verify_a2 : Load the exact untouched Assignment 2 Combined PUAD checkpoint, verify exact
                87.56% Mean / 83.30% Pooled parity, and profile params/GFLOPs/latency/VRAM.
  - train     : Train Stage 1 + fit/calibrate Stage 2 PUAD + single-pass test eval + profiling.
  - eval      : Evaluate an existing trained checkpoint in checkpoints/<exp_id>/best_checkpoint.pth.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.env import (
    capture_environment,
    load_experiment_config,
    save_json,
    set_seed,
)
from src.common.profiling import profile_puad_system
from src.data.loco import audit_dataset
from src.evaluation.metrics import format_result_summary
from src.models.puad_system import build_puad_system
from src.models.trainer import train_stage1_puad
from src.scoring.puad_scorer import fit_and_evaluate_puad, gap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Path to experiment YAML (e.g. configs/E01_puad.yaml)")
    ap.add_argument("--mode", choices=["verify_a2", "train", "eval"], default="train")
    ap.add_argument("--smoke", action="store_true", help="Fast 2-epoch/30-iter smoke test for debugging")
    ap.add_argument("--force", action="store_true", help="Overwrite existing results/checkpoints")
    ap.add_argument("--seed", type=int, default=None, help="Override random seed")
    ap.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs")
    ap.add_argument("--mixed-precision", choices=["bf16", "fp16", "fp32"], default=None)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--imagenet-root", default=None)
    ap.add_argument("--teacher-path", default=None)
    ap.add_argument("--drive-sync-dir", default=None, help="Optional Google Drive folder to persist artifacts")
    args = ap.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    tr_ov = {}
    if args.batch_size is not None:
        tr_ov["batch_size"] = args.batch_size
    if args.epochs is not None:
        tr_ov["epochs"] = args.epochs
    if args.mixed_precision is not None:
        tr_ov["mixed_precision"] = args.mixed_precision
    if tr_ov:
        overrides["training"] = tr_ov
    p_ov = {}
    if args.dataset_root:
        p_ov["dataset_root"] = args.dataset_root
    if args.imagenet_root:
        p_ov["imagenet_root"] = args.imagenet_root
    if args.teacher_path:
        p_ov["teacher_pdn_s"] = args.teacher_path
    if p_ov:
        overrides["paths"] = p_ov

    cfg = load_experiment_config(args.config, overrides=overrides)
    exp_id = cfg["exp_id"]
    run_tag = f"{exp_id}_verify_a2" if args.mode == "verify_a2" else (f"{exp_id}_smoke" if args.smoke else exp_id)
    if args.seed is not None and args.seed != 42:
        run_tag = f"{run_tag}_seed{args.seed}"

    out_root = Path(cfg["paths"]["output_root"])
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    ckpt_root = Path(cfg["paths"]["checkpoint_root"])
    if not ckpt_root.is_absolute():
        ckpt_root = ROOT / ckpt_root

    out_dir = out_root / run_tag
    ckpt_dir = ckpt_root / run_tag
    drive_sync = Path(args.drive_sync_dir) / run_tag if args.drive_sync_dir else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(int(cfg["seed"]), bool(cfg.get("deterministic", True)))

    print("=" * 78)
    print(f"RUN: {run_tag} | mode={args.mode} | device={device} | seed={cfg['seed']}")
    print("=" * 78)

    ds_audit = audit_dataset(cfg["paths"]["dataset_root"], categories=cfg["categories"])
    p_tot = ds_audit["pooled"]
    print(
        f"Dataset audit OK: train={p_tot['n_train_normal']} | val={p_tot['n_validation_normal']} | "
        f"test_good={p_tot['n_test_good']} | test_logical={p_tot['n_test_logical']} | "
        f"structural_excluded={p_tot['n_test_structural_excluded']}"
    )

    load_teacher = args.mode != "verify_a2"
    system = build_puad_system(cfg, device=device, load_teacher_weights=load_teacher)

    training_summary = None
    if args.mode == "verify_a2":
        ckpt_path = Path(cfg["paths"]["a2_combined_puad_ckpt"])
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Assignment 2 Combined PUAD checkpoint not found: {ckpt_path}")
        print(f"Loading untouched Assignment 2 checkpoint: {ckpt_path}")
    elif args.mode == "train":
        training_summary = train_stage1_puad(
            system=system,
            cfg=cfg,
            out_dir=out_dir,
            ckpt_dir=ckpt_dir,
            device=device,
            force=args.force,
            smoke=args.smoke,
            drive_sync_dir=drive_sync,
        )
        ckpt_path = ckpt_dir / "best_checkpoint.pth"
    else:
        ckpt_path = ckpt_dir / "best_checkpoint.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for eval mode: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    system.teacher.load_state_dict(ckpt["model_state_dict"]["teacher"])
    system.student.load_state_dict(ckpt["model_state_dict"]["student"])
    system.autoencoder.load_state_dict(ckpt["model_state_dict"]["autoencoder"])
    system.eval()

    print(
        f"Loaded checkpoint: epoch={ckpt.get('epoch')} | best_val_loss={ckpt.get('best_val_loss', float('nan')):.4f}"
    )

    metrics, scorer, head = fit_and_evaluate_puad(
        system=system,
        teacher_mu=ckpt["teacher_mu"],
        teacher_sigma=ckpt["teacher_sigma"],
        quantiles=ckpt["quantiles"],
        dataset_root=cfg["paths"]["dataset_root"],
        img_size=int(cfg["img_size"]),
        device=device,
        scores_csv_path=out_dir / "test_scores.csv",
        reference_puad=cfg.get("reference_puad"),
        categories=cfg["categories"],
    )

    # Full end-to-end single-image inference profiling (EfficientAD + PUAD Mahalanobis)
    def _infer_once(x: torch.Tensor):
        ea_s, feats = scorer(x)
        maha_s = head.score(gap(feats))
        return ea_s, maha_s

    eval_cfg = cfg["evaluation"]
    efficiency = profile_puad_system(
        system=system,
        forward_fn=_infer_once,
        img_size=int(cfg["img_size"]),
        device=device,
        warmup=int(eval_cfg.get("latency_warmup", 20)),
        runs=int(eval_cfg.get("latency_runs", 100)),
    )
    if training_summary and training_summary.get("peak_training_gpu_memory_mib") is not None:
        efficiency["peak_training_gpu_memory_mib"] = training_summary["peak_training_gpu_memory_mib"]

    payload = {
        "exp_id": exp_id,
        "run_tag": run_tag,
        "mode": args.mode,
        "smoke_test": bool(args.smoke),
        "description": cfg.get("description", ""),
        "seed": int(cfg["seed"]),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_val_loss": ckpt.get("best_val_loss"),
        "metrics": metrics,
        "efficiency": efficiency,
        "training_summary": training_summary,
        "dataset_audit": ds_audit,
        "environment": capture_environment(str(out_dir)),
    }

    result_path = out_dir / "test_results.json"
    save_json(payload, result_path)
    if drive_sync is not None:
        import shutil
        drive_sync.mkdir(parents=True, exist_ok=True)
        for fname in ("test_results.json", "test_scores.csv"):
            src_f = out_dir / fname
            if src_f.exists():
                shutil.copy2(src_f, drive_sync / fname)

    print("\n" + format_result_summary(run_tag, payload))
    print(f"\nSaved -> {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
