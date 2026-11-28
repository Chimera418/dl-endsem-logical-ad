"""Strict logical-only evaluation metrics: Per-Category, Combined Mean, and Pooled Raw AUROC."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.loco import CATEGORIES, CATEGORY_LABELS, EVALUATED_TYPES, GOOD, LOGICAL, STRUCTURAL


def strict_logical_auroc(
    scores: Sequence[float],
    labels: Sequence[int],
    anomaly_types: Sequence[str],
) -> float:
    """Compute I-AUROC strictly on test/good vs test/logical_anomalies with tripwire checks."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    t = np.asarray(anomaly_types)

    if (t == STRUCTURAL).any():
        raise AssertionError("TRIPWIRE: Structural anomalies reached strict_logical_auroc!")
    if not np.isin(t, EVALUATED_TYPES).all():
        raise ValueError(f"Unexpected anomaly types outside {EVALUATED_TYPES}: {np.unique(t)}")
    if not np.isfinite(s).all():
        raise ValueError(f"Non-finite scores detected: {(~np.isfinite(s)).sum()} / {len(s)}")
    if not ((t == GOOD).any() and (t == LOGICAL).any()):
        raise ValueError("Both good and logical_anomalies samples are required to compute AUROC.")

    return float(roc_auc_score(y, s))


def evaluate_logical_scores(
    scores: Sequence[float],
    labels: Sequence[int],
    anomaly_types: Sequence[str],
    categories: Sequence[str],
    reference_puad: Optional[Dict[str, Any]] = None,
    sub_scores: Optional[Dict[str, Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Compute full Section 10 evaluation report: per-category, Combined Mean, Pooled Raw, and deltas."""
    pooled_auroc = strict_logical_auroc(scores, labels, anomaly_types)

    present_cats = set(categories)
    eval_cats = [c for c in CATEGORIES if c in present_cats]
    if not eval_cats:
        raise ValueError("No valid MVTec LOCO categories found in evaluation inputs.")

    per_cat: Dict[str, float] = {}
    for cat in eval_cats:
        idx = [i for i, c in enumerate(categories) if c == cat]
        c_s = [scores[i] for i in idx]
        c_y = [labels[i] for i in idx]
        c_t = [anomaly_types[i] for i in idx]
        per_cat[cat] = strict_logical_auroc(c_s, c_y, c_t)

    combined_mean = float(np.mean(list(per_cat.values())))

    n_good = sum(1 for t in anomaly_types if t == GOOD)
    n_logical = sum(1 for t in anomaly_types if t == LOGICAL)
    n_structural = sum(1 for t in anomaly_types if t == STRUCTURAL)

    out: Dict[str, Any] = {
        "primary_anomaly_type": LOGICAL,
        "n_images_scored": len(scores),
        "n_test_good": n_good,
        "n_test_logical": n_logical,
        "n_structural_in_final_auroc": n_structural,
        "per_category_auroc": per_cat,
        "combined_mean_auroc": combined_mean,
        "pooled_auroc": pooled_auroc,
    }

    if reference_puad:
        ref_mean = float(reference_puad["combined_mean_auroc"])
        ref_pooled = float(reference_puad["pooled_auroc"])
        out["improvement_over_puad"] = {
            "delta_combined_mean_pp": round((combined_mean - ref_mean) * 100.0, 4),
            "delta_pooled_pp": round((pooled_auroc - ref_pooled) * 100.0, 4),
            "reference_combined_mean_auroc": ref_mean,
            "reference_pooled_auroc": ref_pooled,
        }

    if sub_scores:
        diagnostics = {}
        for name, sub_s in sub_scores.items():
            sub_pooled = strict_logical_auroc(sub_s, labels, anomaly_types)
            sub_cats = {}
            for cat in eval_cats:
                idx = [i for i, c in enumerate(categories) if c == cat]
                sub_cats[cat] = strict_logical_auroc(
                    [sub_s[i] for i in idx],
                    [labels[i] for i in idx],
                    [anomaly_types[i] for i in idx],
                )
            diagnostics[name] = {
                "combined_mean_auroc": float(np.mean(list(sub_cats.values()))),
                "pooled_auroc": sub_pooled,
                "per_category_auroc": sub_cats,
            }
        out["sub_branch_diagnostics"] = diagnostics

    return out


def format_result_summary(exp_id: str, results: Dict[str, Any]) -> str:
    """Render a concise terminal & markdown summary of Section 10 required metrics."""
    m = results["metrics"]
    p = results.get("efficiency", {})
    lines = [
        f"=== {exp_id} ({results.get('mode', 'unknown')}) ===",
        "Per-Category Logical I-AUROC:",
    ]
    for cat in CATEGORIES:
        if cat in m["per_category_auroc"]:
            lines.append(f"  {CATEGORY_LABELS[cat]:<22}: {m['per_category_auroc'][cat]*100:6.2f}%")
    lines.append(f"  {'Combined Mean AUROC':<22}: {m['combined_mean_auroc']*100:6.2f}%  (Primary)")
    lines.append(f"  {'Pooled Raw AUROC':<22}: {m['pooled_auroc']*100:6.2f}%  (Secondary)")
    if "improvement_over_puad" in m:
        imp = m["improvement_over_puad"]
        lines.append(
            f"  Delta vs A2 PUAD      : Mean {imp['delta_combined_mean_pp']:+.2f} pp | "
            f"Pooled {imp['delta_pooled_pp']:+.2f} pp"
        )
    if p:
        lines.append(
            f"Efficiency: Params={p.get('total_parameters_m', 'N/A')}M "
            f"(Trainable={p.get('trainable_parameters_m', 'N/A')}M) | "
            f"GFLOPs={p.get('total_gflops', 'N/A')} | "
            f"Latency={p.get('inference_latency_ms_mean', 'N/A')} ms/img "
            f"({p.get('throughput_fps', 'N/A')} FPS) | "
            f"Peak Inference VRAM={p.get('peak_inference_gpu_memory_mib', 'N/A')} MiB"
        )
    return "\n".join(lines)
