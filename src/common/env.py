"""Environment verification, reproducibility, config loading, logging, and early stopping."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs used across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True


def rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def get_system_ram_gb() -> Dict[str, Optional[float]]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / (1024 ** 3), 2),
            "available_gb": round(vm.available / (1024 ** 3), 2),
            "used_percent": float(vm.percent),
        }
    except Exception:
        return {"total_gb": None, "available_gb": None, "used_percent": None}


def capture_environment(check_path: str = ".") -> Dict[str, Any]:
    """Capture full hardware, software, RAM, and disk environment (Section 3 audit)."""
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "total_memory_gib": round(props.total_memory / (1024 ** 3), 2),
            "total_memory_mib": round(props.total_memory / (1024 ** 2)),
            "capability": f"{props.major}.{props.minor}",
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        }

    driver = None
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
    except Exception:
        pass

    disk = shutil.disk_usage(os.path.abspath(check_path))
    optional_pkgs = {}
    for pkg in ("torchvision", "timm", "thop", "einops", "sklearn", "mamba_ssm"):
        try:
            m = __import__(pkg)
            optional_pkgs[pkg] = str(getattr(m, "__version__", "installed"))
        except Exception:
            optional_pkgs[pkg] = None

    return {
        "python_version": sys.version.split()[0],
        "python_executable": str(sys.executable),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu,
        "nvidia_driver": str(driver) if driver else None,
        "system_ram": get_system_ram_gb(),
        "disk": {
            "path": os.path.abspath(check_path),
            "free_gb": round(disk.free / (1024 ** 3), 2),
            "total_gb": round(disk.total / (1024 ** 3), 2),
        },
        "numpy_version": str(np.__version__),
        "packages": optional_pkgs,
    }


def has_native_bf16() -> bool:
    """True only on Ampere+ GPUs (CC >= 8.0) with hardware BF16 Tensor Cores (not Turing CC 7.5 emulation)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8 and bool(torch.cuda.is_bf16_supported())


def resolve_amp_dtype(requested: str, device: str) -> Optional[torch.dtype]:
    """Select safe AMP dtype based on actual hardware capability (never emulate BF16 on Turing T4)."""
    req = (requested or "fp32").lower()
    if device != "cuda" or req == "fp32":
        return None
    if req == "bf16":
        if has_native_bf16():
            return torch.bfloat16
        props = torch.cuda.get_device_properties(0)
        print(
            f"[info] {props.name} (CC {props.major}.{props.minor}) lacks native BF16 Tensor Cores; "
            "using FP16 with GradScaler instead."
        )
        return torch.float16
    if req == "fp16":
        return torch.float16
    raise ValueError(f"Unknown mixed_precision setting: {requested!r}")


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_experiment_config(
    exp_config_path: str | Path,
    base_config_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load base.yaml + experiment yaml + environment variable overrides + CLI overrides."""
    exp_path = Path(exp_config_path).resolve()
    if base_config_path is None:
        base_config_path = exp_path.parent / "base.yaml"
    cfg = deep_merge(load_yaml(base_config_path), load_yaml(exp_path))

    # Environment variable path overrides for Colab Pro / Drive seamlessly
    env_map = {
        "ENDSEM_LOCO_ROOT": "dataset_root",
        "ENDSEM_IMAGENET_ROOT": "imagenet_root",
        "ENDSEM_TEACHER_PATH": "teacher_pdn_s",
        "ENDSEM_A2_CKPT": "a2_combined_puad_ckpt",
        "ENDSEM_OUTPUT_ROOT": "output_root",
        "ENDSEM_CHECKPOINT_ROOT": "checkpoint_root",
    }
    paths = cfg.setdefault("paths", {})
    for env_key, cfg_key in env_map.items():
        val = os.environ.get(env_key)
        if val:
            paths[cfg_key] = val

    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)


class CsvLogger:
    """Flushing CSV logger so Colab disconnects never lose epoch history."""

    def __init__(self, path: str | Path, fieldnames: Iterable[str], append: bool = False) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        resuming = append and self.path.exists() and self.path.stat().st_size > 0
        self._fh = open(self.path, "a" if resuming else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if not resuming:
            self._writer.writeheader()
        self._fh.flush()

    def log(self, **row: Any) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 0.0

    best_value: float = float("inf")
    best_epoch: int = -1
    counter: int = 0
    stopped_epoch: Optional[int] = None
    first_worsening_epoch: Optional[int] = None
    history: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []

    def step(self, epoch: int, value: float) -> bool:
        self.history.append(float(value))
        if value < self.best_value - self.min_delta:
            self.best_value = float(value)
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        if self.first_worsening_epoch is None:
            self.first_worsening_epoch = epoch
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
        return False

    @property
    def should_stop(self) -> bool:
        return self.stopped_epoch is not None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "stopped_epoch": self.stopped_epoch,
            "first_worsening_epoch": self.first_worsening_epoch,
            "history": list(self.history),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best_value = state["best_value"]
        self.best_epoch = state["best_epoch"]
        self.counter = state["counter"]
        self.stopped_epoch = state["stopped_epoch"]
        self.first_worsening_epoch = state["first_worsening_epoch"]
        self.history = list(state["history"])

    def summary(self) -> Dict[str, Any]:
        return {
            "minimum_validation_loss": self.best_value if self.best_epoch > 0 else None,
            "epoch_of_minimum": self.best_epoch if self.best_epoch > 0 else None,
            "epoch_validation_loss_began_worsening": self.first_worsening_epoch,
            "early_stopping_epoch": self.stopped_epoch,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "epochs_run": len(self.history),
        }


def cycle(iterable):
    while True:
        for item in iterable:
            yield item
