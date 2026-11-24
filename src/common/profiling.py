"""Model efficiency profiling: parameter count, GFLOPs, inference latency, and peak GPU VRAM."""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def measure_gflops(module: nn.Module, input_shape=(1, 3, 256, 256), device: str = "cpu") -> Optional[float]:
    """Compute forward-pass GFLOPs using thop (on a fresh copy so hooks never pollute the model)."""
    try:
        from thop import profile
        model_copy = copy.deepcopy(module).to(device).eval()
        dummy = torch.randn(*input_shape, device=device)
        with torch.no_grad():
            macs, _ = profile(model_copy, inputs=(dummy,), verbose=False)
        del model_copy, dummy
        if device == "cuda":
            torch.cuda.empty_cache()
        # Standard convention: 1 MAC = 2 FLOPs, or report MACs*2 in GFLOPs
        return round((macs * 2.0) / 1e9, 4)
    except Exception:
        return None


@torch.no_grad()
def benchmark_inference(
    forward_fn: Callable[[torch.Tensor], Any],
    input_shape=(1, 3, 256, 256),
    device: str = "cuda",
    warmup: int = 20,
    runs: int = 100,
) -> Dict[str, Any]:
    """Measure single-image inference latency (ms), throughput (FPS), and peak inference VRAM (MiB)."""
    dummy = torch.randn(*input_shape, device=device)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for _ in range(warmup):
        _ = forward_fn(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = forward_fn(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    peak_mib = (
        round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
        if device == "cuda"
        else None
    )
    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    return {
        "inference_latency_ms_mean": round(mean_ms, 3),
        "inference_latency_ms_std": round(std_ms, 3),
        "throughput_fps": round(1000.0 / max(mean_ms, 1e-6), 2),
        "peak_inference_gpu_memory_mib": peak_mib,
        "warmup_runs": warmup,
        "timed_runs": runs,
    }


def profile_puad_system(
    system: nn.Module,
    forward_fn: Callable[[torch.Tensor], Any],
    img_size: int = 256,
    device: str = "cuda",
    warmup: int = 20,
    runs: int = 100,
) -> Dict[str, Any]:
    """Full parameter, FLOPs, latency, and peak VRAM report for any PUAD-family model."""
    trainable = count_parameters(system, trainable_only=True)
    total = count_parameters(system, trainable_only=False)
    frozen = total - trainable

    per_component = {}
    for name in ("teacher", "student", "autoencoder"):
        sub = getattr(system, name, None)
        if isinstance(sub, nn.Module):
            per_component[name] = {
                "total_params": count_parameters(sub, trainable_only=False),
                "trainable_params": count_parameters(sub, trainable_only=True),
                "gflops": measure_gflops(sub, (1, 3, img_size, img_size), device="cpu"),
            }

    total_gflops = None
    vals = [c["gflops"] for c in per_component.values() if c["gflops"] is not None]
    if len(vals) == len(per_component) and vals:
        total_gflops = round(sum(vals), 4)

    latency = benchmark_inference(
        forward_fn,
        input_shape=(1, 3, img_size, img_size),
        device=device,
        warmup=warmup,
        runs=runs,
    )

    return {
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "total_parameters": total,
        "total_parameters_m": round(total / 1e6, 3),
        "trainable_parameters_m": round(trainable / 1e6, 3),
        "total_gflops": total_gflops,
        "components": per_component,
        **latency,
    }
