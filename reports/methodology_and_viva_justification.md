# Experimental Methodology, Candidate Selection & Viva Defense Reference

**Project:** 22AI304 / 22AIE304 End-Semester Deep Learning Capstone (Team 11: Aneesh Sagar Reddy, Nikil Raagav, Sousheel T)  
**Benchmark:** MVTec LOCO AD — Strictly Unsupervised / Normal-Only **Logical Anomaly Detection** (`test/good` = `575` vs. `test/logical_anomalies` = `561` → `1,136` images; `test/structural_anomalies` = `432` strictly excluded).

---

## 1. Why This Exact Sequential Phase Order (`A → B → C → D → E → Integration`)?

Our experimental protocol follows **coarse-to-fine, largest-impact-first sequential coordinate-descent** (the same controlled design methodology used in *ConvNeXt*, *EfficientNet*, and *YOLOv4*):

1. **Phase A — Baseline Verification & Protocol Calibration (`E01`):**
   - First verify exact `+0.00 pp` parity with our untouched Assignment 2 Combined PUAD checkpoint (`87.56%` Combined Mean / `83.30%` Pooled Raw).
   - Establish a single controlled 200-epoch training schedule (`E01_PUAD_200ep`: `89.64%` Mean / `84.07%` Pooled) so every downstream `+ / -` delta is strictly attributable to architectural changes alone.
2. **Phase B — Backbone Investigation (`E02–E05`):**
   - Investigated **first** because the backbone dictates spatial output resolution (`56×56` stride-4 vs. `32×32` stride-8), normalization regime (BN-free vs. BatchNorm), and parameter capacity — producing the largest accuracy swings (`9–18 pp`). Attention, fusion, and global context modules cannot be fairly evaluated before locking the backbone.
3. **Phase C — Attention Investigation (`E06–E15`):**
   - Holding the winning backbone (`PDN-S`) fixed, evaluate **10 mechanisms spanning the complete attention taxonomy** at the exact same `(B, 256, 59, 59)` spatial feature map to diagnose *which inductive bias* logical anomalies actually require.
4. **Phase D — Multi-Scale Feature Fusion (`E16–E17`):**
   - Evaluate `FPN` vs. `BiFPN` on **attention-free `PDN-S`** first (controlled isolation) so multi-scale pyramid fusion's standalone contribution (`+0.69 pp` Mean, `+3.77 pp` Pooled) is proven independently before stacking with attention.
5. **Phase E — Global Context Modelling (`E18–E19`):**
   - Evaluate **Transformer (MHSA)** vs. **Mamba (2D Cross-Scan Selective SSM)** individually on the exact same `16×16 = 256` spatial token grid to directly compare quadratic pairwise self-attention (`O(N^2 D)`) against linear state-space propagation (`O(N D d_state)`).
6. **Final Integration & Progressive Ablation (`E20–E21`):**
   - Stack **only empirically winning components** progressively (`E20: PDN-S + E16_FPN + E15_CrissCross`, then `E21: E20 + Phase E Winner`) so every retained component is verified both in isolation (`E01–E19`) and in combination (`E20–E21`). Any module that degrades AUROC when stacked is rejected.

---

## 2. Why Each Phase's Specific Candidates Were Chosen

### Phase B — Backbones (`PDN-S` vs. `E02–E05`)
Each candidate represents a **distinct CNN architectural family** (avoiding redundant within-family variants such as ResNet-18/34/50) sized appropriately for `1,778` normal training images (very high-capacity models like ResNet-152 or ViT-Large overfit immediately, as confirmed by `ConvNeXt-Tiny` early-stopping at epoch 1):

| Exp | Backbone | Architectural Family / Inductive Bias | Why Included | Measured Outcome |
|---|---|---|---|---|
| `E01` | **`PDN-S`** | Shallow, BN-free patch-descriptor CNN (stride-4 `56×56`) | Winning A2 PUAD reference backbone | **`89.64%` Mean / `84.07%` Pooled (RETAIN)** |
| `E02` | `ResNet-18` | Classic residual skip-connection CNN (He et al., 2016) | Universal AD benchmark backbone (PatchCore, RD4AD, STFPM) | `74.10%` / `69.95%` (Reject) |
| `E03` | `DenseNet-121` | Dense cross-layer feature reuse via concatenation (Huang et al., 2017) | Tests whether early+deep feature reuse aids layout (`#1` on BB: `95.23%`) | `79.65%` / `78.03%` (Reject) |
| `E04` | `EfficientNet-B0` | Compound-scaled inverted bottleneck + SE (Tan & Le, 2019) | Tests extreme FLOP efficiency (`6.58 GFLOPs`) | `81.01%` / `75.40%` (Reject) |
| `E05` | `ConvNeXt-Tiny` | Modern large-kernel (`7×7` DW) + LayerNorm CNN (Liu et al., 2022) | Tests ViT-inspired large kernels without BatchNorm | `71.91%` / `70.51%` (Reject) |

### Phase C — 10 Attention Mechanisms (`E06–E15`)
Selected to cover **all 5 major taxonomic families of visual attention**, spanning `+0` parameters to full global self-attention:

| Taxonomic Family | Experiments | Mechanism / Hypothesis Tested | Measured Outcome |
|---|---|---|---|
| **1. Pure Channel Recalibration** | `E06_SE`, `E07_ECA`, `E12_GCT` | FC bottleneck (`SE`) vs. local 1D conv without reduction (`ECA`, `+5` params) vs. L2-norm gated channel competition (`GCT`, no FC/BN) | `GCT` `89.98%`, `ECA` `89.63%`, `SE` `87.77%` |
| **2. Joint Channel + Local Spatial** | `E08_CBAM`, `E13_BAM` | Sequential channel → `7×7` spatial (`CBAM`) vs. parallel channel + dilated (`d=4`) spatial (`BAM`) | `BAM` `89.62%`, `CBAM` `88.00%` |
| **3. Coordinate / Cross-Dimension** | `E09_CoordAtt`, `E10_Triplet` | 1D H/W directional pooling with BN (`CoordAtt`) vs. 3-branch `(C,H)/(C,W)/(H,W)` rotation (`Triplet`, `+294` params) | `Triplet` `89.18%`, `CoordAtt` `84.05%` (BN spike) |
| **4. Parameter-Free 3D Energy** | `E11_SimAM` | Closed-form neuroscience-inspired 3D neuron energy (`+0` params — zero overfitting risk) | `SimAM` `89.41%` (lowest early `val_loss=1.035`) |
| **5. Global Long-Range Spatial** | `E14_NonLocal`, `E15_CrissCross` | Full `O((HW)^2)` pairwise attention (`NonLocal`) vs. factorized `O(HW(H+W))` recurrent (`R=2`) axial attention (`CrissCross`) | **`CrissCross` `90.74%` (#1 RETAIN)**, `NonLocal` `90.03%` (#2) |

### Phase D — Multi-Scale Feature Fusion (`E16–E17`)
Represents the two canonical **feature pyramid fusion** paradigms over the exact same BN-free 4-level pyramid (`P2: 126×126, P3: 59×59, P4: 30×30, P5: 15×15`):
- **`E16_FPN`** (Lin et al., 2017): Unidirectional **top-down** semantic propagation with equal-weight addition → **`90.33%` Mean / `87.84%` All-Time Record Pooled (RETAIN)**.
- **`E17_BiFPN`** (Tan et al., 2020): **Bi-directional** (top-down + bottom-up) with **fast-normalized learnable fusion weights** and same-level skips → `90.31%` Mean / `83.72%` Pooled (Runner-Up; over-distilled to `val=0.877`).

### Phase E — Global Context Modelling (`E18–E19`)
Direct head-to-head comparison of the **two dominant modern global sequence-modelling families** on the exact same `16×16 = 256` spatial token grid:
- **`E18_Transformer`** (Vaswani et al., 2017 / ViT): 2× Pre-LN Multi-Head Self-Attention (Flash SDPA, `O(N^2 D)` quadratic all-pairs interaction) → **`90.68%` Mean / `86.13%` Pooled** (improved all 5 categories simultaneously, `11.08 ms` on L4).
- **`E19_Mamba`** (Gu & Dao, 2023 / VMamba, 2024): 2× Pre-LN 2D 4-Way Cross-Scan Selective SSM (`O(N D d_state)` linear complexity via exact 8-step Hillis-Steele parallel associative scan) → *in progress*.

---

## 3. Key Training & Evaluation Protocol Justifications (Viva FAQ)

### Q1. Why 200 Epochs (`AdamW`, `lr=3e-4 → 1e-5` cosine, `patience=10`) Instead of 50 Epochs?
- Measured directly on `E01_PUAD`: at `batch_size=8` over `1,778` images (`222` steps/epoch), 50 epochs (`11.1k` steps) severely under-trained (`val_loss=2.1314`, **`83.94%` Mean**).
- Scaling to 200 epochs (`44.4k` steps, closer to original EfficientAD/PUAD's `70k` steps) with `AdamW` cosine decay and `patience=10` early stopping reached `val_loss=1.0517` and **`89.64%` Mean (`+5.70 pp`)**, while `patience=10` automatically halted over-fitting models early (`E02_ResNet18` at ep 19, `E05_ConvNeXt` at ep 11).

### Q2. Why `10,000` Stratified ImageNet Penalty Images Instead of All `50,000`?
- Our deterministic (`seed=42`) `10,000`-image subset still spans **all 1,000 ImageNet classes** (~10 OOD images/class), providing full natural-image OOD diversity for `L_penalty = mean(S(x_ood)^2)`.
- Empirically, `E01_PUAD_200ep` with the 10k subset outperformed Assignment 2's 50k-penalty PUAD by `+2.08 pp` (`89.64%` vs. `87.56%`).
- Cuts Colab storage/transfer by 20× (`288.8 MB` vs. `~6.5 GB`, 8s SSD extraction) and is held **100% identical** across every experiment (`E01–E21`).

### Q3. Why Combined-First (Unified 5-Category) and Individual-Category Only on the Final Winner?
- Running 21 experiments × 5 individual categories would require **105 separate trainings** (~5× compute cost).
- Unified 5-category training is both **5× more compute-efficient** and a **stricter industrial deployment setting** (one model handles all 5 domains without test-time category labels).
- Running **1 Individual 5-Category Benchmark** (5 strictly separate per-category models, zero pooling) once at the end on the **single final winning architecture** enables direct apples-to-apples comparison against *both* Combined and Individual Assignment 2/3 baselines (`PUAD 91.58%`, `SALAD 91.52%`, etc.).

### Q4. What If a Losing Backbone + Losing Attention Combined Gave a Better Result?
- **Theoretically possible** (higher-order cross-factor interaction), since exhaustive grid search across `5 backbones × 11 attentions × 3 fusions × 3 contexts = 495 models` (~850+ GPU hours) is computationally intractable.
- **Empirically very unlikely here due to effect-size separation:** `PDN-S` beat the `#2` backbone (`EfficientNet-B0`, `81.01%`) by **`+8.63 pp`** and `ResNet-18` (`74.10%`) by **`+15.54 pp`** due to a hard spatial-resolution bottleneck (stride-8 `32×32` collapsed `Pushpins` from `98.0%` to `57–66%`). The largest attention gain across all 10 mechanisms was `+1.10 pp` (`CrissCross`), which cannot bridge a `9–18 pp` backbone resolution deficit.
- **Reported transparently:** Unexplored higher-order interactions (e.g., whether a multi-scale/attention neck could rescue `DenseNet-121`, which scored `#1` on `Breakfast Box` at `95.23%`) are explicitly documented under **Limitations & Future Work**.
