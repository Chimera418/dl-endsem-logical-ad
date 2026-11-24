# Experiment E01 Analysis — Original PUAD Reference Architecture

## A. Research Hypothesis
Original PUAD (EfficientAD PDN-S Teacher/Student + 1×1 Global AutoEncoder + Ledoit–Wolf Mahalanobis OOD head with validation z-score fusion) serves as our verified reference architecture (**87.56%** Combined Mean AUROC, **83.30%** Pooled Raw AUROC). Running both an untouched Assignment 2 checkpoint verification (`E01_verify_a2`) and a fresh Colab Pro unified-protocol training run (`E01_colab_bs8`) verifies exact numerical reproducibility, captures missing efficiency metrics (GFLOPs, latency, VRAM), and tests sensitivity to batch-size scaling before Phase B backbone experiments.

## B. Architectural Explanation
- **Frozen Teacher**: `PDN_S(out_channels=384, padding=False)` (`2.694M` frozen params, `30.63 GFLOPs`), receptive field `33×33`, output `(B, 384, 56, 56)`.
- **Trainable Student**: `PDN_S(out_channels=768, padding=False)` (`4.267M` params, `40.49 GFLOPs`), split into `s_st` (`384` ch) and `s_ae` (`384` ch).
- **Trainable AutoEncoder**: 6-conv encoder → `(B, 64, 1, 1)` global bottleneck → 8-conv bilinear decoder with `Dropout(0.2)` (`1.096M` params, `4.79 GFLOPs`), output `(B, 384, 56, 56)`.
- **Stage 2 Scoring**: Global Average Pooling on `s_st` → `384-d` vector → `LedoitWolf` covariance + `pinv` Mahalanobis distance fitted on pooled `train/good` (`1,778` imgs) → validation z-score sum with EfficientAD quantile-normalized anomaly map (`305` `validation/good` imgs).

## C. Implementation & D. Training Configuration
| Variant | Hardware | Batch Size | Optimizer & LR | Steps / Epochs | Precision | Best Val Loss |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **E01_orig (`verify_a2`)** | RTX 4060 (8 GB) | 1 | Adam `1e-4 → 1e-5` (iter 66.5k) | 70,000 iters (40 ep) | FP32 | **1.3154** (ep 40) |
| **E01_colab (`bs8_lr1e4`)** | Colab Tesla T4 (14.6 GB, CC 7.5) | 8 | Adam `1e-4 → 1e-5` (ep 45) | 11,100 iters (50 ep) | FP16 + GradScaler | **2.1314** (ep 50) |

*Note on Colab Hardware Adaptation (Section 3)*: A100 was temporarily unavailable; Colab assigned a Tesla T4 (`CC 7.5`, `51 GB RAM`). Our environment detector caught that T4 lacks native BF16 Tensor Cores and automatically ran native FP16 with `GradScaler`.

## E–G. Actual Experimental Results (Measured — Logical Anomalies Only, 0 Structural Included)

| Metric | E01_orig (`verify_a2`) | E01_colab (`bs8_lr1e4`) | EfficientAD Map Only (`verify_a2`) | Mahalanobis Only (`verify_a2`) |
| :--- | :---: | :---: | :---: | :---: |
| Breakfast Box AUROC | **90.99%** | 90.27% | 77.19% | 94.63% |
| Juice Bottle AUROC | **97.58%** | 95.16% | 94.26% | 96.69% |
| Pushpins AUROC | **86.73%** | 78.64% | 84.39% | 72.96% |
| Screw Bag AUROC | **67.87%** | 67.24% | 59.78% | 71.75% |
| Splicing Connectors AUROC | **94.64%** | 88.41% | 90.80% | 82.61% |
| **Combined Mean AUROC (Primary)** | **87.56%** (`+0.00 pp`) | 83.94% (`-3.62 pp`) | 81.28% | 83.73% |
| **Pooled Raw AUROC (Secondary)** | **83.30%** (`+0.00 pp`) | 80.07% (`-3.23 pp`) | 80.97% | 76.23% |

## H. Computational Overhead
- **Parameters**: `8.058M` total (`5.364M` trainable, `2.694M` frozen)
- **Compute**: `75.91 GFLOPs` (very high FLOPs relative to parameter count because `PDN_S` applies `4×4` convs over `253×253` and `123×123` spatial grids)
- **Inference Latency**: `20.48 ms/img` (`48.83 FPS`) on RTX 4060; `20.22 ms/img` (`49.46 FPS`) on Tesla T4
- **Peak Inference GPU Memory**: `171.78 MiB` (RTX 4060) / `151.65 MiB` (Tesla T4)

## I. Interpretation of Results & Protocol Lesson Learned
1. **Exact Baseline Parity Confirmed**: `E01_verify_a2` reproduced Assignment 2 to **+0.00 pp** on both Combined Mean (`87.56%`) and Pooled Raw (`83.30%`), proving the new Endsem codebase introduces zero evaluation drift.
2. **Why `E01_colab` (`bs=8, lr=1e-4, 50 ep`) scored `83.94%` (`-3.62 pp`)**:
   - Raising `batch_size` from `1` → `8` while holding `lr = 1e-4` cut total Adam gradient steps by **6.3×** (`70,000 → 11,100`).
   - Validation loss was still steadily decreasing at Epoch 50 (`2.1314` vs. `1.3154` in A2), confirming **under-training** rather than architectural degradation.
   - **Actionable Protocol Adjustment for Phase B–E**: Under `batch_size=8`, apply Adam's square-root batch-size learning-rate scaling ($\sqrt{8} \times 10^{-4} \approx 3 \times 10^{-4}$) with cosine annealing down to `1e-5` over `80 epochs` (`patience=15`), and document this explicitly as a controlled protocol update applied uniformly across all Phase B–E experiments.
3. **Key Weakness to Target in Phase B**:
   - `screw_bag` (`67.87%`) is the clearest bottleneck (EfficientAD map alone gets only `59.78%`), driven by `PDN_S`'s `33×33` local receptive field.
   - Modern hierarchical backbones (`ResNet-18`, `DenseNet-121`, `EfficientNet-B0`, `ConvNeXt-Tiny`) expand the receptive field while cutting GFLOPs by ~4–10× relative to `PDN_S`.

## J. Decision
- **Retain** `E01_verify_a2` (**87.56% Combined Mean / 83.30% Pooled Raw**) as the official reference anchor (`Improvement over PUAD = 0.00 pp`).
- **Proceed to Phase B** (`E02_ResNet18`, `E03_DenseNet121`, `E04_EfficientNetB0`, `E05_ConvNeXtTiny`) with scaled `lr=3e-4` (cosine schedule, 80 epochs, `batch_size=8`, `patience=15`).
