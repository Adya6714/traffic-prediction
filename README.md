# Traffic Demand Prediction

Spatiotemporal forecasting of normalized traffic **demand** per `(geohash, 15-minute slot)` for the GridLock 2.0 hackathon. This repository documents the full problem framing, experimentation, and the **87.90** public-leaderboard solution (legitimate, reproducible, no external data).

## Problem

Predict `demand` ∈ (0, 1] for every test row — a **forward forecast**, not a random split:

| Split | Content |
|-------|---------|
| **Train** | Day 48 (all 96 slots) + day 49 morning (slots 0–8, 00:00–02:00) |
| **Test** | Day 49 daytime (slots 9–55, 02:15–13:45) |

**Metric:** `score = max(0, 100 × R²)` on the test set.

Only two days of history exist. Any feature for a test row may use day-48 history and that row's own attributes — never future labels.

## Solution (champion)

A **four-way blend** of diverse predictors with **per-RoadType weights**:

| Predictor | Role |
|-----------|------|
| `raw_cell` | Day-48 `(geohash, slot)` lookup with hierarchical fallback |
| `model` | LightGBM on smoothed target encodings |
| `corr` | Day-over-day corrector (day-48 base + day-49 row features) |
| `corr2` | Corrector + lat/lon + CatBoost/LightGBM ensemble |

**Blend weights** (raw / model / corr / corr2):

| Road type | Weights |
|-----------|---------|
| Highway | 0.45 / 0.25 / 0.15 / 0.15 |
| Residential | 0.65 / 0.10 / 0.12 / 0.13 |
| Street & default | 0.55 / 0.15 / 0.15 / 0.15 |

**Result:** `outputs/agent_b6_hwy_model_up.csv` — leaderboard score **87.90** (87.89558).

Progression: 83.6 → 86.33 → 87.47 → 87.65 → **87.90**.

## Quick start

```bash
# 1. Environment (Python 3.11 recommended)
./scripts/setup_env.sh
source .venv/bin/activate

# 2. Data — place competition files in dataset/
#    train.csv, test.csv, sample_submission.csv

# 3. Reproduce the champion submission (~3–5 min CPU)
jupyter notebook reproduce.ipynb          # Run All cells
# or:
python -m experiments.gen_batch6          # writes outputs/agent_b6_hwy_model_up.csv
```

## Repository layout

```
├── reproduce.ipynb          # Primary reproduction notebook (verification)
├── config.yaml              # Paths, seed, data contract
├── requirements.txt
├── make_submission_bundle.sh
├── src/                     # Core pipeline modules
├── experiments/             # Batch generators + diagnostics
├── notebooks/               # EDA notebooks
├── docs/
│   ├── APPROACH.md          # Full methodology & experiment log
│   └── EXPERIMENT_LOG.md    # Assumptions tested, dead-ends, findings
├── dataset/                 # Competition CSVs (not in git — see dataset/README.md)
├── outputs/                 # Generated submissions (champion kept for reference)
└── scripts/                 # Environment setup
```

## Key findings

- **Slot-level lookup matters** for daytime (dropping slot detail → score ~69).
- **Blending beats any single model** — raw + model + corrector + corr2.
- **RoadType-specific blend weights** beat a global mix (+0.25 over uniform four-way).
- **Day-49 morning scaling does not transfer to daytime** — multiplicative day-factor hurt badly (batch 8).
- **Local nighttime folds mislead** — always validate daytime ideas on the leaderboard.

Full write-up: [docs/APPROACH.md](docs/APPROACH.md).

## Build verification bundle

For competition source submission:

```bash
bash make_submission_bundle.sh   # → submission_bundle.zip
```

## Requirements

Python 3.11 · pandas · numpy · scikit-learn · LightGBM · CatBoost · Jupyter

## License

MIT — see [LICENSE](LICENSE).
