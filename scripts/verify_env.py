"""Section 3 Hardware, Software, Disk, and Dataset Verification Utility.

Run at the beginning of every local or Google Colab Pro session before training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.env import capture_environment, load_experiment_config
from src.data.loco import audit_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "E01_puad.yaml"))
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--imagenet-root", default=None)
    ap.add_argument("--teacher-path", default=None)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    dataset_root = args.dataset_root or cfg["paths"]["dataset_root"]
    imagenet_root = args.imagenet_root or cfg["paths"]["imagenet_root"]
    teacher_path = args.teacher_path or cfg["paths"]["teacher_pdn_s"]

    env = capture_environment(str(ROOT))
    print("=" * 78)
    print("SECTION 3 — ENVIRONMENT & DATASET VERIFICATION AUDIT")
    print("=" * 78)
    print(f"1. CUDA Available    : {env['cuda_available']}")
    if env["gpu"]:
        g = env["gpu"]
        print(f"2. GPU Model         : {g['name']} (CC {g['capability']}, BF16={g['bf16_supported']})")
        print(f"3. GPU Memory        : {g['total_memory_gib']} GiB ({g['total_memory_mib']} MiB)")
    else:
        print("2. GPU Model         : NONE (CPU only!)")
        print("3. GPU Memory        : N/A")
    print(f"4. PyTorch / CUDA    : PyTorch {env['torch_version']} | CUDA {env['torch_cuda_version']} | cuDNN {env['cudnn_version']}")
    print(f"   Python / Packages : Python {env['python_version']} | timm={env['packages'].get('timm')} | thop={env['packages'].get('thop')} | mamba_ssm={env['packages'].get('mamba_ssm')}")
    ram = env["system_ram"]
    print(f"5. System RAM        : {ram['available_gb']} GB free / {ram['total_gb']} GB total ({ram['used_percent']}% used)")
    d = env["disk"]
    print(f"6. Disk Space        : {d['free_gb']} GB free / {d['total_gb']} GB total ({d['path']})")

    print("-" * 78)
    print("7. Dataset & Weights Accessibility:")
    ds_audit = None
    if os.path.isdir(dataset_root):
        ds_audit = audit_dataset(dataset_root)
        p = ds_audit["pooled"]
        print(f"   MVTec LOCO AD     : OK ({dataset_root})")
        print(
            f"     train={p['n_train_normal']} | val={p['n_validation_normal']} | "
            f"test_good={p['n_test_good']} | test_logical={p['n_test_logical']} | "
            f"test_structural={p['n_test_structural_excluded']} (EXCLUDED)"
        )
        if ds_audit["warnings"]:
            print(f"     WARNINGS: {ds_audit['warnings']}")
    else:
        print(f"   MVTec LOCO AD     : MISSING ({dataset_root})")

    if os.path.isdir(imagenet_root):
        n_im = sum(1 for p in Path(imagenet_root).rglob("*") if p.is_file() and p.suffix in (".png", ".jpg", ".jpeg", ".JPEG"))
        print(f"   ImageNet Penalty  : OK ({n_im:,} images at {imagenet_root})")
    else:
        print(f"   ImageNet Penalty  : MISSING ({imagenet_root})")

    print(f"   PDN-S Teacher     : {'OK' if os.path.isfile(teacher_path) else 'MISSING'} ({teacher_path})")
    print("=" * 78)

    if args.save_json:
        out = {"environment": env, "dataset_audit": ds_audit}
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"Saved audit JSON -> {args.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
