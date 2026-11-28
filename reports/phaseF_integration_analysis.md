# Phase F — Final Progressive Integration & Ablation (`E20`, `E21`, `E22`)

**Project:** 22AI304 / 22AIE304 End-Semester Deep Learning Capstone (Team 11)  
**Baseline:** `E01_PUAD_200ep` (`PDN-S`, `89.64%` Combined Mean / `84.07%` Pooled Raw) | **A2 Reference:** `87.56%` Mean / `83.30%` Pooled  
**Protocol:** Progressive stacking of proven phase winners — **Phase B (`PDN-S`)**, **Phase C (`E15_CrissCross`)**, **Phase D (`E16_FPN`)**, and **Phase E co-winners (`E18_Transformer` vs. `E19_Mamba`)** — under the identical controlled protocol (200 epochs `AdamW` cosine `3e-4 → 1e-5`, `patience=10`, native `BF16` on solo NVIDIA L4, `seed=42`, validation-only checkpointing, strict logical-only `1,136`-image test evaluation).

---

## 1. Architectural Integration Pipeline

Inside `EnhancedPDN_S` (BN-free student branch), each retained component operates at a distinct, complementary spatial hierarchy:
1. **Backbone (`Phase B`, `PDN-S`):** Preserves exact `33×33` local receptive fields and teacher-student geometric parity.
2. **Multi-Scale Fusion (`Phase D`, `E16_FPN`):** 4-level top-down pyramid (`P2` `126×126`, `P3` `59×59`, `P4` `30×30`, `P5` `15×15`, `128ch`, zero-init residual gate) fused onto `59×59`.
3. **Spatial Attention (`Phase C`, `E15_CrissCross`):** Two recurrent passes of axial row-and-column affinity attention over the `59×59` feature map.
4. **Global Context (`Phase E`, `E18_Transformer` vs. `E19_Mamba`):** Downsampled `16×16 = 256` token grid with 2D learnable positional embeddings and zero-init residual gate — comparing **all-pairs 2-layer Transformer self-attention (`E21`)** vs. **4-way directional 2D Mamba SSM (`E22`)**.

---

## 2. Phase F Measured Results (Solo NVIDIA L4, Native BF16)

| Model | Stacked Components | Best Ep (`val_loss`) | BB | JB | PP | SB | SC | **Combined Mean** | **Pooled Raw** | Δ vs E01_200 (Mean / Pooled) | Δ vs A2 (Mean / Pooled) | Params | GFLOPs | L4 Solo Latency (FPS) | Peak Train / Infer VRAM | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `E01_PUAD_200ep` | `PDN-S` (Baseline) | 183 (`1.0517`) | `90.53%` | `96.87%` | `98.02%` | `68.94%` | `93.85%` | **89.64%** | `84.07%` | — | `+2.08` / `+0.77 pp` | `8.058M` | `75.91G` | `10.00 ms` (`100.0`) | `1463 / 131 MiB` | Baseline |
| *ref: `E15_CrissCross`* | *`PDN-S + CC`* | *184 (`1.0087`)* | *`91.41%`* | *`96.99%`* | *`97.71%`* | ***`71.65%`*** | ***`95.95%`*** | *`90.74%`* | *`85.33%`* | *`+1.10` / `+1.26 pp`* | *`+3.18` / `+2.03 pp`* | ***`8.140M`*** | ***`77.05G`*** | ***`11.09 ms` (`90.2`)*** | *`— / 140 MiB`* | *Lightweight Pareto* |
| *ref: `E16_FPN`* | *`PDN-S + FPN`* | *90 (`1.0872`)* | *`91.91%`* | ***`98.23%`*** | *`98.77%`* | *`68.47%`* | *`94.25%`* | *`90.33%`* | ***`87.84%`*** | *`+0.69` / `+3.77 pp`* | *`+2.77` / `+4.54 pp`* | *`9.747M`* | *`88.27G`* | *`11.59 ms` (`86.3`)* | *`1952 / 173 MiB`* | *Pooled Record* |
| *ref: `E18_Transformer`* | *`PDN-S + Trans`* | *175 (`0.8213`)* | *`91.19%`* | *`97.48%`* | *`98.07%`* | *`69.73%`* | *`94.82%`* | *`90.26%`* | *`85.40%`* | *`+0.62` / `+1.33 pp`* | *`+2.70` / `+2.10 pp`* | *`10.358M`* | *`81.61G`* | *`10.80 ms` (`92.6`)* | *`1708 / 194 MiB`* | *Phase E Winner* |
| *ref: `E19_Mamba`* | *`PDN-S + Mamba`* | *200 (`0.8806`)* | *`91.87%`* | *`97.58%`* | *`96.62%`* | *`69.44%`* | *`95.71%`* | *`90.25%`* | *`87.33%`* | *`+0.61` / `+3.26 pp`* | *`+2.68` / `+4.03 pp`* | *`9.736M`* | *`81.29G`* | *`12.72 ms` (`78.6`)* | *`3388 / 190 MiB`* | *Phase E Co-Winner* |
| `E20_FPN_CrissCross` | `PDN-S + FPN + CC` | 173 (`0.8446`) | `89.26%` | `97.80%` | `98.87%` | `68.36%` | `94.34%` | `89.73%` | `85.33%` | `+0.09` / `+1.26 pp` | `+2.17` / `+2.03 pp` | `9.829M` | `89.41G` | `12.75 ms` (`78.4`) | `2625 / 192 MiB` | Stage 1 Integration |
| **`E21_FPN_CrissCross_Transformer`** | **`PDN-S + FPN + CC + Trans`** | **192 (`0.7677`)** | **`91.93%`** | `97.70%` | **`99.22%`** | `70.28%` | `95.26%` | **`90.88%`** | **`87.36%`** | **`+1.24` / `+3.29 pp`** | **`+3.32` / `+4.05 pp`** | `12.129M` | `95.12G` | **`13.45 ms`** (`74.3`) | **`2910 / 222 MiB`** | **FINAL CHAMPION** |
| `E22_FPN_CrissCross_Mamba` | `PDN-S + FPN + CC + Mamba` | 197 (`0.8117`) | `89.96%` | `98.11%` | `98.48%` | `64.46%` | `95.36%` | `89.27%` | `86.43%` | `-0.37` / `+2.36 pp` | `+1.71` / `+3.12 pp` | `11.507M` | `94.80G` | `15.61 ms` (`64.1`) | `4596 / 209 MiB` | Stage 2b Reject |

---

## 3. Core Scientific Findings

1. **`E21_FPN_CrissCross_Transformer` achieves the all-time best overall performance (`90.88%` Combined Mean / `87.36%` Pooled Raw):**
   - Surpasses verified Assignment 2 PUAD (`87.56% / 83.30%`) by **`+3.32 pp` Combined Mean** and **`+4.05 pp` Pooled Raw**, and beats the controlled 200-epoch baseline `E01_PUAD_200ep` (`89.64% / 84.07%`) on **all 5 categories simultaneously** (`BB +1.40`, `JB +0.83`, `PP +1.20`, `SB +1.34`, `SC +1.41`).
   - Achieves the **all-time highest Pushpins AUROC (`99.22%`)** and **all-time lowest validation loss (`0.7677`)** across all 22 runs, while maintaining **real-time `74.3 FPS` (`13.45 ms/img`)** on NVIDIA L4.
2. **Why Stage 1 (`E20_FPN_CrissCross`) dipped before Stage 2a (`E21`) surged (`+1.15 pp` Mean / `+2.03 pp` Pooled):**
   - Stacking `FPN` + `CrissCross` alone (`E20`) yielded `89.73%` Mean / `85.33%` Pooled (`BB` dipped to `89.26%`, `SB` to `68.36%`). Injecting dense multi-scale `P2/P3` features directly into axial row/column attention without a global token bottleneck caused local axial over-correlation.
   - Adding the `16×16` **2D Transformer** (`E21`) resolved this completely via direction-agnostic all-pairs global token mixing, lifting `Breakfast Box` by `+2.67 pp`, `Screw Bag` by `+1.92 pp`, `Pushpins` to `99.22%`, and `Pooled Raw` by `+2.03 pp`.
3. **Why `E21` (`Transformer`) decisively defeated `E22` (`Mamba`) in the full stack (`+1.61 pp` Mean, `+0.93 pp` Pooled):**
   - Although `E18_Transformer` (`90.26%`) and `E19_Mamba` (`90.25%`) were virtually tied in Phase E isolation, stacking `Mamba` on top of `FPN + CrissCross` (`E22`) dropped Combined Mean to **`89.27%`** (`Screw Bag` collapsed to `64.46%`, `-5.82 pp` vs `E21`) while running slower (`15.61 ms` vs `13.45 ms`) and consuming `58%` more training VRAM (`4.60 GiB` vs `2.91 GiB`).
   - **Mechanistic cause:** `CrissCross` already applies axial row/column inductive bias at `59×59`; layering 4-way directional 1D `Mamba` scans directly afterward creates redundant axial bias and exponential state decay on unaligned object layouts (`Screw Bag`). In contrast, `Transformer` all-pairs self-attention is permutation-invariant and cleanly complements axial `CrissCross`.
4. **Pareto Deployment Frontier:**
   - **Maximum Accuracy & Cross-Category Calibration:** **`E21_FPN_CrissCross_Transformer`** (`90.88%` Mean / `87.36%` Pooled, `12.13M` params, `13.45 ms` / `74.3 FPS`).
   - **Ultra-Lightweight Edge Deployment:** **`E15_CrissCross`** (`90.74%` Mean / `85.33%` Pooled, `8.14M` params [`-33%`], `11.09 ms` / `90.2 FPS`).
5. **Next Step (Individual 5-Category Benchmark):**
   - Evaluate **`E21_FPN_CrissCross_Transformer`** in the zero-training-pooling Individual 5-Category setting (plus concatenated `1,136`-image pooled test evaluation) via `notebooks/06_colab_individual_winner.ipynb` for direct comparison against published per-category benchmarks (`PUAD 91.58%`, `SALAD 91.52%`, `LogiCo 87.86%`, `CSAD 86.54%`, `GCAD 86.52%`, `ComAD 76.03%`).
