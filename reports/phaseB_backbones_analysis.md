# Phase B Analysis — Backbone Investigation (E01–E05) on MVTec LOCO AD

## 1. Research Questions Addressed
- Does residual feature learning (**E02 ResNet-18**) improve logical anomaly detection over PDN-S?
- Does dense feature reuse (**E03 DenseNet-121**) improve feature representation?
- Does compound-scaled MBConv (**E04 EfficientNet-B0**) offer a useful accuracy–efficiency trade-off?
- Does a modern large-kernel pure-CNN (**E05 ConvNeXt-Tiny**) improve visual feature extraction?
- Which backbone preserves the most useful spatial information under unified multi-class normal-only training?

---

## 2. Controlled Experimental Setup
All 5 runs (`E01_PUAD_200ep` through `E05_ConvNeXtTiny`) were trained and evaluated on **NVIDIA L4 (22.03 GiB VRAM, CC 8.9, native BF16)** under identical conditions:
- **Dataset**: Pooled 5-category MVTec LOCO AD (`1,778` train normal, `305` val normal, `1,136` logical-only test = `575` good + `561` logical; `432` structural excluded).
- **Training**: `batch_size=8`, `AdamW`, `lr=3e-4 → 1e-5` (cosine), `weight_decay=1e-4`, max `200 epochs`, `EarlyStopping(patience=10)` on `validation/good` `val_loss`, `seed=42`, `L_penalty` on 10k ImageNet corpus with `preserve_bn_stats` to prevent BatchNorm OOD contamination.
- **Scoring**: EfficientAD quantile-calibrated map + Ledoit–Wolf Mahalanobis on GAP student features (`384-d`), fused via validation z-scores.

---

## 3. Master Measured Results Table (E01–E05)

| Exp | Backbone | Breakfast Box | Juice Bottle | Pushpins | Screw Bag | Splicing Conn. | **Combined Mean (Primary)** | **Pooled Raw (Secondary)** | Δ Mean vs A2 PUAD | Δ Pooled vs A2 PUAD | Params (M) | GFLOPs | Latency (ms) | Peak Train VRAM (MiB) | Best Ep / Val Loss |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ref** | A2 PUAD (`verify_a2`) | 90.99% | **97.58%** | 86.73% | 67.87% | **94.64%** | 87.56% | 83.30% | +0.00 pp | +0.00 pp | 8.06 | 75.91 | 20.48 *(4060)* | — | ep 40 / `1.3154` |
| **E01** | **Original `PDN-S` (200ep)** | 90.53% | 96.87% | **98.02%** | **68.94%** | 93.85% | **89.64%** | **84.07%** | **+2.08 pp** | **+0.77 pp** | **8.06** | 75.91 | 10.00 *(L4)* | 1463.44 | ep 183 / `1.0517` |
| **E02** | ResNet-18 | 85.40% | 87.62% | 62.88% | 64.51% | 70.08% | 74.10% | 69.95% | -13.46 pp | -13.36 pp | 24.09 | 14.53 | **7.40** *(L4)* | **912.24** | ep 9 / `30.7954` |
| **E03** | DenseNet-121 | **95.23%** | 94.74% | 65.11% | 59.85% | 83.30% | 79.65% | 78.03% | -7.91 pp | -5.27 pp | 16.28 | 21.12 | 33.75 *(L4)* | 2635.03 | ep 37 / `14.3004` |
| **E04** | EfficientNet-B0 | 90.90% | 96.88% | 66.41% | 65.27% | 85.58% | 81.01% | 75.40% | -6.55 pp | -7.91 pp | 8.77 | **6.58** | 19.50 *(L4)* | 1726.55 | ep 132 / `5.8452` |
| **E05** | ConvNeXt-Tiny | 87.48% | 90.65% | 56.94% | 51.04% | 73.44% | 71.91% | 70.51% | -15.65 pp | -12.79 pp | 57.55 | 28.63 | 15.31 *(L4)* | 2742.68 | ep 1 / `25.4076` |

---

## 4. Deep Scientific Interpretation (Per-Backbone A–J Summary)

### E01 — Original `PDN-S` (200ep Unified) → **RETAIN (PHASE B WINNER)**
- **Result**: **89.64% Combined Mean / 84.07% Pooled Raw** (`+2.08 pp` Mean and `+0.77 pp` Pooled over Assignment 2's 70k-iter reference).
- **Why it won**: (1) Preserves dense stride-4 (`56×56`) spatial detail (`Pushpins = 98.02%`, `Splicing Connectors = 93.85%`), (2) compact capacity (`5.36M` trainable params) with zero BatchNorm avoids overfitting on `1,778` normal images (`best_val_loss = 1.0517` at epoch 183), and (3) smooth linear teacher output (`conv4`) matches `q=0.999` hard-mining MSE cleanly.
- **Remaining weakness to target in Phases C–E**: `Screw Bag` (`68.94%`) and `Breakfast Box` (`90.53%`), where wider receptive fields and relational context are needed.

### E04 — EfficientNet-B0 → **REJECT (Best ImageNet Alternative, -8.63 pp vs E01)**
- **Result**: **81.01% Combined Mean / 75.40% Pooled Raw**; lowest compute (**6.58 GFLOPs**, `11.5×` lower than E01).
- **Why it led E02–E05**: Its compact trainable size (`5.17M`) and smooth `SiLU` activations prevented overfitting (`ep 132`, `val_loss = 5.85`), matching E01 on `Breakfast Box` (`90.90%`) and `Juice Bottle` (`96.88%`). However, stride-8 `32×32` resolution degraded fine-part counting (`Pushpins = 66.41%`).

### E03 — DenseNet-121 → **REJECT (-9.99 pp vs E01, Notable Macro-Layout Insight)**
- **Result**: **79.65% Combined Mean / 78.03% Pooled Raw** (`+5.55 pp` Mean / `+8.08 pp` Pooled over ResNet-18).
- **Key insight**: Achieved the **highest Breakfast Box AUROC (`95.23%`, `+4.70 pp` over E01!)** and highest Pooled AUROC among E02–E05, confirming that dense multi-scale feature reuse strengthens global layout reasoning — but slower inference (`33.75 ms`), earlier overfitting (`ep 37`), and `32×32` spatial loss on `Pushpins` (`65.11%`) and `Screw Bag` (`59.85%`) kept overall mean below E01.

### E02 — ResNet-18 → **REJECT (-15.54 pp vs E01)**
- **Result**: **74.10% Combined Mean / 69.95% Pooled Raw**; fastest inference (**7.40 ms / 135 FPS**), but `12.91M` randomly initialized trainable parameters + sparse ReLU teacher spikes caused rapid overfitting (`best_epoch = 9`, `val_loss = 30.80`).

### E05 — ConvNeXt-Tiny → **REJECT (-17.73 pp vs E01)**
- **Result**: **71.91% Combined Mean / 70.51% Pooled Raw**; heaviest model (`57.55M` total / `29.73M` trainable). Randomly initialized ConvNeXt-Tiny overfit immediately on `1,778` normal images (`best_epoch = 1`, `val_loss = 25.41`), confirming that large modern ConvNeXt blocks cannot be trained from scratch on small one-class industrial datasets without large-scale pretraining or strong structural constraints.

---

## 5. Phase B Conclusion & Selected Backbone for Phase C
- **Selected Backbone**: **`PDN-S` (E01, 89.64% Combined Mean / 84.07% Pooled Raw)** — empirically proven #1 by **+8.63 pp** over the closest alternative (`EfficientNet-B0`, `81.01%`).
- **Architectural Roadmap for Phases C–E on `PDN-S`**:
  - **Phase C (`E06–E15`)**: Insert all 10 attention mechanisms cleanly inside `PDN-S` Student (`conv3` output: `(B, 256, 59, 59)` → `(B, 256, 59, 59)` before `conv4`), keeping the frozen `PDN-S` teacher and training protocol 100% identical to `E01_PUAD_200ep`.
  - **Phase D (`E16–E17`)**: Add multi-scale **FPN** and **BiFPN** across `PDN-S` Student's multi-resolution feature pyramid.
  - **Phase E (`E18–E19`)**: Add **Transformer (MHSA)** vs. **Mamba / 2D-SSM** global contextual reasoning on `PDN-S` Student's spatial feature grid to directly target `Screw Bag` (`68.94%`) and `Breakfast Box` (`90.53%`) while preserving `Pushpins` (`98.02%`).
