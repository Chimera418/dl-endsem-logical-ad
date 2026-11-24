"""Aggregate Section 10 required metrics across all completed Endsem experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]


def collect_rows(search_dirs: List[Path]) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for d in search_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("test_results.json")):
            with open(p, "r", encoding="utf-8") as fh:
                j = json.load(fh)
            if j.get("smoke_test"):
                continue
            tag = j.get("run_tag", p.parent.name)
            if tag in seen:
                continue
            seen.add(tag)
            m = j["metrics"]
            eff = j.get("efficiency", {})
            imp = m.get("improvement_over_puad", {})
            gpu = (j.get("environment") or {}).get("gpu") or {}
            rows.append(
                {
                    "run_tag": tag,
                    "exp_id": j.get("exp_id", tag),
                    "breakfast_box": round(m["per_category_auroc"]["breakfast_box"] * 100, 2),
                    "juice_bottle": round(m["per_category_auroc"]["juice_bottle"] * 100, 2),
                    "pushpins": round(m["per_category_auroc"]["pushpins"] * 100, 2),
                    "screw_bag": round(m["per_category_auroc"]["screw_bag"] * 100, 2),
                    "splicing_connectors": round(m["per_category_auroc"]["splicing_connectors"] * 100, 2),
                    "combined_mean_auroc": round(m["combined_mean_auroc"] * 100, 2),
                    "pooled_auroc": round(m["pooled_auroc"] * 100, 2),
                    "delta_mean_pp": round(imp.get("delta_combined_mean_pp", 0.0), 2),
                    "delta_pooled_pp": round(imp.get("delta_pooled_pp", 0.0), 2),
                    "total_params_m": eff.get("total_parameters_m"),
                    "trainable_params_m": eff.get("trainable_parameters_m"),
                    "gflops": eff.get("total_gflops"),
                    "latency_ms": eff.get("inference_latency_ms_mean"),
                    "peak_infer_vram_mib": eff.get("peak_inference_gpu_memory_mib"),
                    "peak_train_vram_mib": eff.get("peak_training_gpu_memory_mib"),
                    "best_epoch": j.get("checkpoint_epoch"),
                    "best_val_loss": round(j.get("checkpoint_best_val_loss", 0.0), 4)
                    if j.get("checkpoint_best_val_loss") is not None
                    else None,
                    "gpu": gpu.get("name", "N/A"),
                }
            )
    return rows


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    header = (
        "| Run | BB | JB | PP | SB | SC | **Comb Mean** | **Pooled** | Δ Mean | Δ Pooled | "
        "Params (M) | GFLOPs | Lat (ms) | Peak VRAM (MiB) | GPU |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | "
        ":---: | :---: | :---: | :---: | :--- |"
    )
    lines = [header]
    for r in rows:
        vram = r["peak_train_vram_mib"] or r["peak_infer_vram_mib"]
        lines.append(
            f"| {r['run_tag']} | {r['breakfast_box']:.2f}% | {r['juice_bottle']:.2f}% | "
            f"{r['pushpins']:.2f}% | {r['screw_bag']:.2f}% | {r['splicing_connectors']:.2f}% | "
            f"**{r['combined_mean_auroc']:.2f}%** | **{r['pooled_auroc']:.2f}%** | "
            f"{r['delta_mean_pp']:+.2f} pp | {r['delta_pooled_pp']:+.2f} pp | "
            f"{r['total_params_m']} | {r['gflops']} | {r['latency_ms']} | {vram} | {r['gpu']} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", default=["experiments"])
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    rows = collect_rows([Path(d) for d in args.dirs])
    if not rows:
        print("No completed non-smoke test_results.json found.")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = to_markdown(rows)
    print(md)

    (out_dir / "master_comparison.md").write_text(md + "\n", encoding="utf-8")
    with open(out_dir / "master_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {out_dir / 'master_comparison.md'} and {out_dir / 'master_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
