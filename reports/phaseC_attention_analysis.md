# Phase C — Systematic Attention Mechanism Investigation (`E06–E15`)

**Project:** 22AI304 / 22AIE304 End-Semester Deep Learning Capstone (Team 11)  
**Baseline:** `E01_PUAD_200ep` (Winning Phase B backbone `PDN-S`, unified 5-category model, 200 epochs, `AdamW lr=3e-4 → 1e-5` cosine, `patience=10`, native `BF16`, `seed=42`)  
**Published Reference:** Assignment 2 Combined PUAD (`87.56%` Combined Mean / `83.30%` Pooled Raw)  
**Strict Evaluation Protocol:** Image-level Logical AUROC on `test/good` (`575`) vs. `test/logical_anomalies` (`561`) = `1,136` images; `test/structural_anomalies` (`432`) strictly excluded; validation-only `val_loss` checkpointing + single-pass test evaluation.

---

## 1. Experimental Setup & Controlled Placement

All 10 attention mechanisms were inserted at the **exact same location** inside `EnhancedPDN_S` (Student branch only; frozen `PDN-S` Teacher and `AutoEncoder` held strictly identical):
- **Insertion point:** Directly after `conv3 + ReLU` on the `(B, 256, 59, 59)` spatial feature tensor before `conv4` (`(B, 768, 56, 56)`).
- **Training protocol:** Identical across all 10 runs (`batch_size=8`, `epochs=200`, `AdamW`, `lr=3e-4 → 1e-5` cosine, `weight_decay=1e-4`, `patience=10`, appearance-only `RandomAugment` on AE input, `10,000`-image ImageNet penalty corpus, native `BF16`).
- **Hardware:** `E06–E10` trained on Colab Pro **NVIDIA A100-SXM4-40GB**; `E11–E15` trained on Colab Pro **NVIDIA L4 (22.5 GB)** (`CC ≥ 8.0`, native `BF16`). *(Note: Latency/FPS for `E07`, `E09`, `E12`, `E13`, `E14` was measured while sharing the GPU with parallel runs; `E06`, `E10`, `E11`, `E15` evaluated solo.)*

---

## 2. Full Phase C Leaderboard (`E06–E15` vs. `E01_PUAD_200ep` & A2 PUAD)

Ranked by **Primary Metric (Combined Mean AUROC)**:

| Rank | Exp | Attention Family | Best Ep (`val_loss`) | BB | JB | PP | SB | SC | **Combined Mean** | **Pooled Raw** | Δ Mean vs E01_200 | Δ Mean vs A2 | Params | GFLOPs | Peak Infer VRAM | Decision |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **1** | **`E15_CrissCross`** | **Global Axial (`R=2`)** | **184 (`1.0087`)** | `91.41%` | `96.99%` | `97.71%` | `71.65%` | **`95.95%`** | **90.74%** | `85.33%` | **`+1.10 pp`** | **`+3.18 pp`** | `8.140M` (`+82k`) | `77.05G` | `167.0 MiB` | **RETAIN (WINNER)** |
| **2** | **`E14_NonLocal`** | **Global Pairwise 2D** | 133 (`1.0904`) | **`92.30%`** | `96.24%` | `96.57%` | **`72.24%`** | `92.80%` | **90.03%** | **`86.11%`** | **`+0.39 pp`** | **`+2.47 pp`** | `8.189M` (`+131k`) | `76.82G` | `169.3 MiB` | Runner-Up |
| **3** | **`E12_GCT`** | Gated Channel Norm | 161 (`1.0718`) | `91.08%` | `95.95%` | `98.45%` | `70.26%` | `94.14%` | **89.98%** | `84.04%` | **`+0.34 pp`** | **`+2.41 pp`** | `8.059M` (`+768`) | `75.91G` | `152.3 MiB` | Strong #3 |
| *ref* | *`E01_PUAD_200ep`* | *Baseline (None)* | *183 (`1.0517`)* | *`90.53%`* | *`96.87%`* | *`98.02%`* | *`68.94%`* | *`93.85%`* | ***`89.64%`*** | *`84.07%`* | *—* | *`+2.08 pp`* | *`8.058M`* | *`75.91G`* | *`171.8 MiB`* | *Baseline* |
| 4 | `E07_ECA` | Local 1D Channel | 184 (`1.1126`) | `89.31%` | `95.71%` | `98.92%` | `70.43%` | `93.79%` | `89.63%` | `85.58%` | `-0.01 pp` | `+2.07 pp` | `8.058M` (`+5`) | `75.91G` | `150.7 MiB` | Reject (tie) |
| 5 | `E13_BAM` | Parallel Ch+Dilated Sp | 188 (`1.1268`) | `90.09%` | `96.52%` | `98.45%` | `68.06%` | `95.01%` | `89.63%` | `85.47%` | `-0.01 pp` | `+2.07 pp` | `8.075M` (`+17k`) | `75.97G` | `150.9 MiB` | Reject (tie) |
| 6 | `E11_SimAM` | Param-Free 3D Energy | 198 (`1.0348`) | `90.26%` | `96.46%` | `97.78%` | `67.66%` | `94.90%` | `89.41%` | `84.63%` | `-0.23 pp` | `+1.85 pp` | `8.058M` (`+0`) | `75.91G` | `131.4 MiB` | Reject |
| 7 | `E10_Triplet` | Cross-Dim Permute | 192 (`1.2352`) | `89.63%` | `96.82%` | `96.47%` | `68.74%` | `94.25%` | `89.18%` | `84.66%` | `-0.46 pp` | `+1.62 pp` | `8.058M` (`+294`) | `75.91G` | `153.7 MiB` | Reject |
| 8 | `E08_CBAM` | Seq. Ch + 7×7 Spatial | 200 (`1.0576`) | `85.91%` | `98.20%` | **`99.10%`** | `63.35%` | `93.45%` | `88.00%` | `85.43%` | `-1.64 pp` | `+0.44 pp` | `8.066M` (`+8.3k`) | `75.91G` | `153.0 MiB` | Reject |
| 9 | `E06_SE` | Global Channel MLP | 198 (`1.0705`) | `87.21%` | **`98.32%`** | `94.23%` | `65.47%` | `93.61%` | `87.77%` | `82.70%` | `-1.87 pp` | `+0.21 pp` | `8.066M` (`+8.5k`) | `75.91G` | `150.8 MiB` | Reject |
| 10 | `E09_CoordAtt` | Axis Pool + BN | 22 (`2.9503`) | `90.30%` | `95.74%` | `79.83%` | `69.31%` | `85.07%` | `84.05%` | `81.08%` | `-5.59 pp` | `-3.51 pp` | `8.071M` (`+12.8k`) | `75.91G` | `150.7 MiB` | Reject (BN spike) |
| *ref* | *A2 PUAD* | *Published Reference* | *70k iters* | *`90.99%`* | *`97.58%`* | *`86.73%`* | *`67.87%`* | *`94.64%`* | *`87.56%`* | *`83.30%`* | *`-2.08 pp`* | *`0.00 pp`* | *`8.058M`* | *`75.91G`* | *`171.8 MiB`* | *Published Ref* |

---

## 3. Core Scientific Findings (Section 12 A–J)

### A. Why Global Long-Range Spatial Attention Won Decisively (`E15_CrissCross` #1, `E14_NonLocal` #2)
1. **The bottleneck of `PDN-S` is its finite local receptive field (`33×33` px):** Logical anomalies in MVTec LOCO AD require comparing distant objects across the entire `256×256` scene — counting/pairing screws of different lengths in `Screw Bag`, verifying cable-to-connector correspondence across opposite sides of `Splicing Connectors`, and checking relative food-compartment composition in `Breakfast Box`.
2. **Direct proof in per-category deltas:**
   - `E15_CrissCross` (`R=2` recurrent row+column attention = full-image `59×59` receptive field) lifted **Screw Bag by `+2.71 pp`** (`68.94% → 71.65%`), **Splicing Connectors by `+2.10 pp`** (`93.85% → 95.95%`), and **Breakfast Box by `+0.88 pp`** (`90.53% → 91.41%`), while achieving the **lowest validation loss of any model (`1.0087`)**.
   - `E14_NonLocal` (full `3481×3481` pairwise self-attention) achieved the **highest Screw Bag (`72.24%`, `+3.30 pp`)**, **highest Breakfast Box (`92.30%`, `+1.77 pp`)**, and **highest Pooled Raw AUROC (`86.11%`, `+2.04 pp`)**, confirming the exact same mechanistic hypothesis!
3. **Why `CrissCross` beat `NonLocal` overall (`90.74%` vs `90.03%`):**
   - `NonLocal` attends densely over `3,481×3,481` pairs in one hop (`12.1M` attention weights per image), which slightly overfits on `1,778` normal training images (early-stopped at ep `133`, `val=1.0904`) and loses `3.15 pp` on `Splicing Connectors` (`92.80%` vs `95.95%`).
   - `CrissCross` factorizes global attention into two structured horizontal+vertical passes (`O(HW(H+W))` instead of `O((HW)^2)`), acting as a **strong geometric inductive bias** for aligned industrial scenes (connectors, trays, compartments) — converging deeper (`val=1.0087`, ep `184`) with **38% fewer extra parameters** (`+82k` vs `+131k`) and real-time solo speed (**`11.26 ms/img` = `88.8 FPS`** on L4).

### B. Why Local / Bottleneck Channel Attentions Underperformed (`SE`, `CBAM`, `CoordAtt`)
- **`E06_SE` (`87.77%`, `-1.87 pp`) & `E08_CBAM` (`88.00%`, `-1.64 pp`):** Both use global average pooling → bottleneck FC layers (`256 → 16 → 256`). While this boosted color/fill-dominated **Juice Bottle** (`98.32%` / `98.20%`) and **Pushpins** (`99.10%` on CBAM), it collapsed spatial context on **Screw Bag** (`65.47%` / `63.35%`) and **Breakfast Box** (`87.21%` / `85.91%`).
- **`E09_CoordAtt` (`84.05%`, `-5.59 pp`):** The *only* Phase C module containing `BatchNorm2d` inside its attention path suffered a sudden `val_loss` explosion at epoch 29 (`2.95 → 14.64`, early-stopped at ep 32), reinforcing our Phase B finding that **BatchNorm inside the PDN-S Student destabilizes hard-mining MSE + OOD penalty optimization**.
- **`E12_GCT` (`89.98%`, `+0.34 pp`)** was the only pure-channel mechanism to beat baseline (`+768` params, zero extra GFLOPs) because its L2-normalized channel gating uses **zero BatchNorm and zero bottleneck compression**.

### C. Efficiency & Selection Decision
- **Selected Winner to RETAIN:** **`E15_CrissCross`**
  - **Combined Mean AUROC:** **`90.74%`** (`+1.10 pp` vs `E01_PUAD_200ep`, **`+3.18 pp` vs A2 PUAD**)
  - **Pooled Raw AUROC:** **`85.33%`** (`+1.26 pp` vs `E01_PUAD_200ep`, **`+2.03 pp` vs A2 PUAD**)
  - **Overhead:** `+81,921` params (`+1.0%`), `+1.14 GFLOPs` (`+1.5%`), **`11.26 ms/img` (`88.8 FPS` solo on L4)**, `167.0 MiB` inference VRAM.
- **Next Phase (Phase D — Multi-Scale Fusion):** Test **`E16_FPN`** and **`E17_BiFPN`** on attention-free `PDN-S` first (isolating multi-scale fusion's standalone contribution before integrating `CrissCross` in Final Integration).
