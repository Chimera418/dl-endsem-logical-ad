# End-Semester Deep Learning Project (22AI304) — Team 11
# Systematic Architectural Investigation for Logical Anomaly Detection on MVTec LOCO AD

Reference baseline: **PUAD (Combined-Category)** — Combined Mean AUROC **87.56%**, Pooled AUROC **83.30%** (verified from Assignment 2).

## Project Layout
```
Endsem/
├── configs/                 # Base protocol & per-experiment YAML configs (E01..E19)
├── src/
│   ├── common/              # Environment audit, seeding, profiling (params/FLOPs/latency/VRAM)
│   ├── data/                # Read-only MVTec LOCO combined loader, SplitAudit, ImageNet penalty
│   ├── backbones/           # PDN-S (E01) + ResNet-18, DenseNet-121, EfficientNet-B0, ConvNeXt-Tiny
│   ├── attention/           # Phase C: 10 attention mechanisms
│   ├── fusion/              # Phase D: FPN & BiFPN
│   ├── context/             # Phase E: Transformer & Mamba/SSM
│   ├── models/              # Unified PUAD architecture + trainer
│   ├── scoring/             # EfficientAD map scorer + Ledoit-Wolf Mahalanobis + z-score fusion
│   └── evaluation/          # Strict logical-only I-AUROC (Combined Mean + Pooled Raw)
├── scripts/                 # CLI runners, env verification, Colab data packaging helper
├── notebooks/               # Google Colab Pro notebooks (A100 + Drive persistence)
├── experiments/             # Per-experiment artifacts (E01_PUAD/, E02_ResNet18/, ...)
├── checkpoints/             # Local checkpoints (gitignored; synced to Google Drive on Colab)
├── results/                 # Consolidated comparison tables
├── logs/                    # Run logs
└── reports/                 # Analysis & final report
```

## Quick Start (Local RTX 4060 Verification)
```powershell
python scripts/verify_env.py
python scripts/run_experiment.py --config configs/E01_puad.yaml --mode verify_a2
python scripts/run_experiment.py --config configs/E01_puad.yaml --mode train --smoke
```

## Google Colab Pro (A100)
Open `notebooks/00_colab_setup_and_E01.ipynb` in Colab Pro (A100, High-RAM) to mount Google Drive, run the 7-point hardware/dataset audit, copy archives to fast local SSD (`/content/`), and execute E01.
