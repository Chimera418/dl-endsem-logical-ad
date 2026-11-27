# Phase E — Global Context Modelling Investigation (`E18_Transformer` vs. `E19_Mamba`)

**Project:** 22AI304 / 22AIE304 End-Semester Deep Learning Capstone (Team 11)  
**Baseline:** `E01_PUAD_200ep` (`PDN-S`, `89.64%` Combined Mean / `84.07%` Pooled Raw)  
**Protocol:** Controlled head-to-head comparison of **Vision Transformer Multi-Head Self-Attention (`E18_Transformer`, `O(N^2 D)`)** vs. **2D 4-Way Cross-Scan Selective State-Space Model (`E19_Mamba`, `O(N D d_state)`)** on the **exact same `16×16 = 256` spatial token grid** (BN-free, 2D learnable positional embeddings, zero-initialized residual gate, 200 epochs `AdamW` cosine, `patience=10`, native `BF16` on solo NVIDIA L4).

---

## 1. Phase E Measured Results (Solo NVIDIA L4, Native BF16)

| Model | Theoretical Complexity | Best Ep (`val_loss`) | BB | JB | PP | SB | SC | **Combined Mean** | **Pooled Raw** | Δ vs E01_200 (Mean / Pooled) | Δ vs A2 (Mean / Pooled) | Params | GFLOPs | L4 Solo Latency (FPS) | Peak Train / Infer VRAM | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `E01_PUAD_200ep` | Local CNN | 183 (`1.0517`) | `90.53%` | `96.87%` | `98.02%` | `68.94%` | `93.85%` | **89.64%** | `84.07%` | — | `+2.08` / `+0.77 pp` | `8.058M` | `75.91G` | `10.00 ms` (`100.0`) | `1463 / 131 MiB` | Baseline |
| **`E18_Transformer`** | **`O(N^2 D)`** | **175 (`0.8213`)** | `91.19%` | `97.48%` | **`98.07%`** | **`69.73%`** | `94.82%` | **`90.26%`** | `85.40%` | **`+0.62` / `+1.33 pp`** | **`+2.70` / `+2.10 pp`** | `10.358M` | `81.61G` | **`10.80 ms`** (`92.6`) | **`1708 / 194 MiB`** | **RETAIN (WINNER)** |
| **`E19_Mamba`** | **`O(N D d_s)`** | 200 (`0.8806`) | **`91.87%`** | **`97.58%`** | `96.62%` | `69.44%` | **`95.71%`** | **`90.25%`** | **`87.33%`** | `+0.61` / **`+3.26 pp`** | `+2.68` / **`+4.03 pp`** | **`9.736M`** | **`81.29G`** | `12.72 ms` (`78.6`) | `3388 / 190 MiB` | Runner-Up (Linear SSM) |

---

## 2. Core Scientific Findings

1. **Both global context paradigms beat the local-CNN baseline by `+0.61–0.62 pp` Combined Mean:**
   - `E18_Transformer` (`90.26%`) and `E19_Mamba` (`90.25%`) are virtually tied (`0.01 pp` apart) on Primary Combined Mean AUROC, proving that **global token-level context over `16×16` spatial patches** consistently improves logical anomaly detection regardless of whether global mixing is implemented via quadratic self-attention or linear selective state-space scans.
2. **Complementary Per-Category & Calibration Profiles:**
   - **`E18_Transformer`** improved **all 5 categories simultaneously** over `E01_PUAD_200ep` (`BB +0.66`, `JB +0.61`, `PP +0.05`, `SB +0.79`, `SC +0.97`) with zero regressions, achieving the lowest validation distillation loss (`0.8213`).
   - **`E19_Mamba`** excelled on macro-layout & linear-path categories (**Breakfast Box `91.87%`**, **Splicing Connectors `95.71%`**) and achieved an outstanding **`87.33%` Pooled Raw AUROC** (`+3.26 pp` vs `E01_200ep`, `+1.93 pp` over `Transformer`, `#2` all-time Pooled behind only `E16_FPN`'s `87.84%`), while trading `-1.40 pp` on small-object `Pushpins` (`96.62%`).
3. **Practical Efficiency Trade-Off at `N = 256` Tokens (NVIDIA L4, Native BF16):**
   - Although `Mamba` has strictly superior asymptotic scaling (`O(N)` vs `O(N^2)`) and `-622k` fewer parameters (`9.74M` vs `10.36M`), at `N = 256` tokens PyTorch's fused **Flash SDPA** makes `E18_Transformer` **18% faster at inference (`10.80 ms` vs `12.72 ms`)** and **2× more VRAM-efficient during training (`1.71 GiB` vs `3.39 GiB`)** than a pure-PyTorch 8-step Hillis-Steele parallel scan (since `mamba_ssm` CUDA kernels are unavailable on Python 3.13 + CUDA 13.0).
4. **Selection Decision:**
   - Because `E18_Transformer` (`90.26%` Mean, `10.80 ms`, `1.71 GiB` train VRAM) and `E19_Mamba` (`90.25%` Mean, **`87.33%` Pooled [`+1.93 pp`]**, 3/5 category wins, `-622k` fewer params, `O(N)` scaling) finished in a **`0.01 pp` statistical dead heat**, **both** are advanced to Final Integration for a direct stacked showdown!
   - **Next Step (Final Integration & Ablation):** Test **`E20_FPN_CrissCross`** (`PDN-S + E16_FPN + E15_CrissCross`), **`E21_FPN_CrissCross_Transformer`** (`+ E18_Transformer`), and **`E22_FPN_CrissCross_Mamba`** (`+ E19_Mamba`) to lock in the single final proposed architecture.
