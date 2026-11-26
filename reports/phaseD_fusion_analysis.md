# Phase D — Multi-Scale Feature Fusion Investigation (`E16_FPN`, `E17_BiFPN`)

**Project:** 22AI304 / 22AIE304 End-Semester Deep Learning Capstone (Team 11)  
**Baseline:** `E01_PUAD_200ep` (Attention-free, Fusion-free `PDN-S`, `89.64%` Combined Mean / `84.07%` Pooled Raw)  
**Protocol:** Per Section 7, both `E16_FPN` and `E17_BiFPN` were evaluated on **attention-free `PDN-S`** to isolate the standalone contribution of multi-scale pyramid fusion before integrating the Phase C winning attention (`E15_CrissCross`) in Final Integration.

---

## 1. Architectural Design

Both fusion necks build the **exact same 4-level BN-free feature pyramid** over `EnhancedPDN_S` (`fpn_channels=128`, zero BatchNorm, zero-initialized learnable residual gate `gamma` onto `s3` at `59×59`):
- `P2` (`126×126`, stride 2) from `s1` (`128ch`) — fine spatial detail
- `P3` (`59×59`, stride 4) from `s2 + s3` (`256ch`) — native `PDN-S` resolution
- `P4` (`30×30`, stride 8) via stride-2 conv — mid-range part grouping
- `P5` (`15×15`, stride 16) via stride-2 conv — macro scene layout

- **`E16_FPN`** (Lin et al., 2017): Unidirectional top-down pathway (`P5 → P4 → P3 → P2`) with `3×3` smoothing convs and 4-scale concatenation onto `59×59`.
- **`E17_BiFPN`** (Tan et al., 2020): Fast-normalized weighted top-down (`P5 → P2`) + bottom-up (`P2 → P5`) pathways with same-level skip connections, depthwise-separable `3×3` convs, and weighted 4-scale aggregation.

---

## 2. Phase D Measured Results (Solo NVIDIA L4, Native BF16, 200 Epochs, `patience=10`)

| Model | Best Ep (`val_loss`) | BB | JB | PP | SB | SC | **Combined Mean** | **Pooled Raw** | Δ vs E01_200 (Mean / Pooled) | Δ vs A2 (Mean / Pooled) | Params | GFLOPs | L4 Solo Latency (FPS) | Peak Infer VRAM | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `E01_PUAD_200ep` (No Fusion) | 183 (`1.0517`) | `90.53%` | `96.87%` | `98.02%` | `68.94%` | `93.85%` | **89.64%** | `84.07%` | — | `+2.08` / `+0.77 pp` | `8.058M` | `75.91G` | `10.00 ms` (`100.0`) | `131.4 MiB` | Baseline |
| **`E16_FPN`** (Top-Down) | 90 (`1.0872`) | **`91.91%`** | **`98.23%`** | **`98.77%`** | `68.47%` | `94.25%` | **`90.33%`** | **`87.84%`** | **`+0.69` / `+3.77 pp`** | **`+2.77` / `+4.54 pp`** | `9.747M` | `88.27G` | `11.59 ms` (`86.3`) | `172.8 MiB` | **RETAIN (WINNER)** |
| **`E17_BiFPN`** (Bi-Dir Weighted) | **199 (`0.8770`)** | `90.95%` | `97.96%` | `97.04%` | **`71.29%`** | **`94.29%`** | **`90.31%`** | `83.72%` | `+0.67` / `-0.35 pp` | `+2.75` / `+0.42 pp` | `9.165M` | `82.42G` | `11.83 ms` (`84.6`) | `166.6 MiB` | Runner-Up |
| *ref: `E15_CrissCross` (Phase C)* | *184 (`1.0087`)* | *`91.41%`* | *`96.99%`* | *`97.71%`* | *`71.65%`* | *`95.95%`* | ***`90.74%`*** | *`85.33%`* | *`+1.10` / `+1.26 pp`* | *`+3.18` / `+2.03 pp`* | *`8.140M`* | *`77.05G`* | *`11.09 ms` (`90.2`)* | *`140.2 MiB`* | *Phase C Winner* |

---

## 3. Core Scientific Findings

1. **Multi-scale pyramid fusion delivers a clear standalone gain (`+0.67` to `+0.69 pp` Combined Mean):** Both `FPN` (`90.33%`) and `BiFPN` (`90.31%`) surpassed the `E01_PUAD_200ep` baseline (`89.64%`) without any attention module, confirming that combining fine (`126×126`) and coarse (`30×30`, `15×15`) spatial scales strengthens logical reasoning across object scales.
2. **Why `E16_FPN` is retained over `E17_BiFPN`:**
   - **Record cross-category score calibration (`87.84%` Pooled Raw AUROC, `+3.77 pp` vs `E01_200ep`, `+4.54 pp` vs A2 PUAD, `+4.12 pp` over `BiFPN`):** `E16_FPN` achieved the highest Pooled Raw AUROC of any architecture tested so far, proving its multi-scale features align anomaly score distributions cleanly across all 5 categories.
   - **Higher Combined Mean (`90.33%` vs `90.31%`) and faster solo L4 inference (`11.59 ms` vs `11.83 ms`):** Despite slightly more parameters (`9.75M` vs `9.17M`), standard dense convolutions in `FPN` execute faster on L4 Tensor Cores than `BiFPN`'s depthwise-separable convolutions.
   - **Over-distillation diagnosis on `E17_BiFPN`:** `BiFPN` drove validation distillation loss down to `0.8770` at epoch 199 (the lowest of all runs), which helped `Screw Bag` (`71.29%`) via bidirectional coarse-to-fine feedback, but slightly over-fitted normal teacher outputs on `Pushpins` (`97.04%` vs `98.77%`) and skewed inter-category score scales (`Pooled = 83.72%`).
3. **Strong complementarity with Phase C Winner (`E15_CrissCross`):** `E16_FPN` excels on `Breakfast Box` (`91.91%`), `Juice Bottle` (`98.23%`), `Pushpins` (`98.77%`), and cross-category `Pooled` calibration (`87.84%`), while `E15_CrissCross` excels on `Screw Bag` (`71.65%`) and `Splicing Connectors` (`95.95%`). Combining them in Final Integration is strongly motivated by measured evidence.
