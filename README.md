<p align="center">
  <strong>GridLock 2.0 — Traffic Demand Prediction</strong><br>
  <em>Spatiotemporal forecasting of normalized traffic demand per geohash &amp; 15-minute slot</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11"></a>
  <a href="#results"><img src="https://img.shields.io/badge/leaderboard-87.90-success" alt="Score 87.90"></a>
  <a href="#"><img src="https://img.shields.io/badge/data-competition%20only-lightgrey" alt="No external data"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#the-problem">Problem</a> ·
  <a href="#solution">Solution</a> ·
  <a href="#research-journey">Research</a> ·
  <a href="#results">Results</a> ·
  <a href="docs/APPROACH.md">Full Write-up</a> ·
  <a href="reproduce.ipynb">Notebook</a>
</p>

---

## At a glance

| | |
|:--|:--|
| **Task** | Predict `demand` ∈ (0, 1] for 41,778 test cells on day 49 daytime |
| **Metric** | `max(0, 100 × R²)` |
| **Champion file** | [`outputs/agent_b6_hwy_model_up.csv`](outputs/agent_b6_hwy_model_up.csv) |
| **Score** | **87.90** (87.89558 on public LB) |
| **Method** | Four-way ensemble · per-RoadType blend weights |
| **Reproduce** | [`reproduce.ipynb`](reproduce.ipynb) · `python -m experiments.gen_batch6` |
| **Runtime** | ~3–5 min on CPU · deterministic (`seed=42`) |

> **Integrity:** Built only from competition `train.csv`. No external datasets, no label lookups, fully reproducible from this repo.

---

## The problem

This is a **forward forecast**, not a random train/test split. You see the past; you predict the future.

```
  DAY 48              DAY 49 — morning           DAY 49 — daytime
 ┌─────────────┐      ┌──────────┐               ┌────────────────────────┐
 │ 96 slots    │      │ slots    │               │ slots 9–55             │
 │ full day    │  +   │ 0–8      │  ──predict──▶ │ 02:15 – 13:45          │
 │             │      │ 00:00–   │               │ (41,778 rows)          │
 └─────────────┘      │ 02:00    │               └────────────────────────┘
      TRAIN               TRAIN                         TEST (no labels)
```

<details>
<summary><strong>Dataset facts</strong></summary>

| | Train | Test |
|:--|--:|--:|
| **Rows** | 77,299 | 41,778 |
| **Days** | 48 + day 49 morning | Day 49 daytime only |
| **Geohashes** | 1,190 | 1,190 (10 cold-start) |
| **Granularity** | 15-min slots (0–95 per day) | Same |

**What makes it hard**
- Only **two days** of history — no long time series.
- **RoadType dominates:** Highway mean demand ≈ 0.61 vs Residential ≈ 0.057 (~10×).
- **Highways are peaky** (evening rush); residential demand is flat across the day.
- Features must be **leakage-safe:** test rows may use day-48 history + own attributes only.

</details>

---

## Solution

We do not rely on a single model. We blend **four predictors** that fail in different ways, then tune **how much of each** by road type.

### Pipeline

```
  train.csv (day 48)
        │
        ├──────────────────┬──────────────────┬──────────────────┐
        ▼                  ▼                  ▼                  ▼
   raw_cell            LightGBM           corrector            corr2
   (geohash,slot       (target            (day-48 base +      (+ lat/lon,
    lookup)             encodings)          day-49 features)     CatBoost)
        │                  │                  │                  │
        └──────────────────┴────────┬─────────┴──────────────────┘
                                    ▼
                         per-RoadType weighted blend
                                    │
                                    ▼
                    agent_b6_hwy_model_up.csv  →  LB 87.90
```

### The four predictors

| # | Name | Idea | Alone (~LB) |
|:-:|:-----|:-----|------------:|
| 1 | **raw_cell** | Yesterday's demand at same `(geohash, slot)` + fallback chain | 79.6 |
| 2 | **model** | LightGBM on smoothed hierarchical encodings | — |
| 3 | **corr** | Correct day-48 level using day-49 row features | — |
| 4 | **corr2** | corr + decoded lat/lon + 2000-tree LGBM/CatBoost mix | — |

### Champion recipe — blend weights

Weights = **raw / model / corr / corr2**

| Road type | Weights | Why |
|:----------|:--------|:----|
| **Highway** | `0.45` / `0.25` / `0.15` / `0.15` | Peaky diurnal curve → trust the model more |
| **Residential** | `0.65` / `0.10` / `0.12` / `0.13` | Flat demand → trust the lookup more |
| **Street & default** | `0.55` / `0.15` / `0.15` / `0.15` | Balanced four-way mix |

<details>
<summary><strong>Feature engineering (summary)</strong></summary>

All features are leakage-safe (out-of-fold encodings on day 48; test rows never see their own label).

- Per-cell lookup with fallback: `geohash → prefix-4 → RoadType×slot → global`
- Smoothed target encodings: geohash, prefix-5/4, geohash×slot, RoadType×slot
- Cyclical time: `sin/cos(2π·slot/96)`
- Decoded lat/lon from geohash
- Day-49 row attributes for the corrector (Weather, RoadType, slot, …)

→ Full detail in [`docs/APPROACH.md`](docs/APPROACH.md) §4

</details>

---

## Research journey

We did not tune one model until it worked. We ran a structured assumption loop: state a belief → encode it → test on the leaderboard (daytime is the only honest validator) → log the outcome. Every assumption is tracked in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md); the full batch-by-batch write-up is in [`docs/APPROACH.md`](docs/APPROACH.md) §6.

### How we measured

| Validator | What it tests | Trust for daytime? |
|:----------|:--------------|:-------------------|
| **Public leaderboard** | Real day-49 daytime labels | **Yes — primary** |
| Daytime folds on day 48 | Same time regime, single day | Noisy; filter only |
| Across-day fold (day 49 morning) | Nighttime day-over-day shift | **Misleading** — looked great, hurt LB |

Local `competition_score()` (`max(0, 100×R²`) lives in `src/splits.py` and was used in diagnostics — but the **87.90 champion score was never computable locally** (test labels are hidden).

### Assumptions that survived → built into the champion

| Belief | Evidence | In the final model? |
|:-------|:---------|:--------------------|
| **RoadType dominates** demand (Highway ≫ Residential) | EDA Layer 2 | Per-RoadType blend weights |
| **Slot-level detail matters** for daytime highways | LB: drop slots → ~69 | `raw_cell` keeps `(geohash, slot)` |
| **Seen geohashes are predictable** from day-48 history | Baselines, diag02 | Hierarchical lookup + encodings |
| **Model adds value as a correction**, not a replacement | Hybrid beats either alone | 0.15–0.25 model weight in blend |
| **Day-49 row features** capture day-over-day change | diag03: 49 → 83 | `corr` corrector |
| **Lat/lon + heavier trees help a little** | Batch 4 +0.18 | `corr2` in four-way blend |
| **Highways vs residential want different mixes** | Batch 6 +0.25 | Champion recipe |

### Assumptions we tested and rejected

| Belief | Result | Why it failed |
|:-------|:-------|:--------------|
| Weather / Temperature / Landmarks drive demand | REJECTED | Flat within RoadType (EDA) |
| Scale predictions up for busier day 49 | REJECTED | exp01 + batch 8: morning ≠ daytime |
| Morning momentum / damping (today-factor) | REJECTED | 83.6 → 50.6 as damping rises |
| Geohash all-day average (no slot) | REJECTED | ~69 on daytime LB |
| Smoothed per-cell lookup | REJECTED | ~69 — washes highway peaks |
| More model power alone (trees, CatBoost, depth) | PLATEAU | 87.47 → 87.65 only (+0.18) |
| Spatio-temporal neighbours (corr3) | REJECTED | +7.4 on night fold, 87.41 on LB |
| Early-slot corrector boost | REJECTED | 87.63 < champion |
| Finer per-RoadType weight grid (batch 7) | PLATEAU | All variants ≤ 87.90 |
| External Grab dataset lookup | **Not pursued** | Rules violation; scores of 100 are lookups |

### Progression in one line

```
lookup → +model blend → +corrector → +corr2 → +per-RoadType weights → plateau
 83.6      86.3           87.5         87.7            87.9
```

Gains came from **structural changes** (new predictor, blend architecture, segmentation). Tweaking the same knobs (more trees, stronger scaling, spatial features) stopped moving the needle after batch 4; batch 6 was the last structural win; batches 7–8 confirmed the ceiling.

### Why we stopped

1. **Model-power axis exhausted** — batches 4–5 added lat/lon, 2000 trees, CatBoost, spatio-temporal features; score moved 87.47 → 87.65 → stuck.
2. **One structural win left** — per-RoadType blend weights (batch 6) gained +0.25; refining them (batch 7) gained nothing.
3. **Last hypothesis failed** — scaling daytime preds by day-49 morning busy-ratio (batch 8) collapsed to 63.9 despite +22 on the morning fold.
4. **No hidden test labels** — we cannot tune further without new ideas or external data (which we did not use).

Legitimate participants report an honest ceiling around **92–93**; our **87.90** is reproducible from `train.csv` only. Pushing further would need a genuinely new structural idea (e.g. live day-49 trajectory features, rank targets, segmented models) — none of those beat the champion in our experiments.

<details>
<summary><strong>Where to read more</strong></summary>

| Document | Contents |
|:---------|:---------|
| [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) | Every assumption (A1–A25+), CONFIRMED/REJECTED/OPEN status, full LB submission table |
| [`docs/APPROACH.md`](docs/APPROACH.md) | EDA, features, models, validation challenge, all 8 batches |
| [`experiments/README.md`](experiments/README.md) | Which script runs which experiment |
| [`notebooks/`](notebooks/) | Layer 1–3 EDA that grounded early assumptions |

</details>

---

## Results

### How the score improved

```
83.6  ████████████████████░░░░░░░░░░  baseline lookup
86.3  ████████████████████████░░░░░░  + model blend        (batch 2)
87.5  █████████████████████████░░░░░  + corrector          (batch 3)
87.7  █████████████████████████░░░░░  + corr2 four-way     (batch 4)
87.9  ██████████████████████████░░░░  + per-RoadType blend (batch 6) ★ champion
```

| Batch | Key change | Score |
|:-----:|:-----------|------:|
| — | Reference lookup + light correction | 83.6 |
| 2 | `0.7·raw + 0.3·model` — restore slot-level lookup | **86.33** |
| 3 | Three-way: raw + model + corrector | **87.47** |
| 4 | Four-way uniform: + corr2 (lat/lon, CatBoost) | **87.65** |
| **6** | **Per-RoadType blend weights** | **87.90** |
| 7 | Refine weights around batch 6 | ≤ 87.90 (plateau) |
| 8 | Multiplicative day-49 morning factor | 63.9 – 83.5 (hurt) |

See [**Research journey**](#research-journey) for assumptions, dead-ends, and why we plateaued. Score tables for every submission: [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) §5.

---

## Quick start

### Prerequisites

- Python **3.11**
- Competition files → [`dataset/`](dataset/README.md) (`train.csv`, `test.csv`)

### Install

```bash
git clone git@github.com:Adya6714/traffic-prediction.git
cd traffic-prediction
./scripts/setup_env.sh
source .venv/bin/activate
```

Place `train.csv` and `test.csv` in `dataset/` (see [`dataset/README.md`](dataset/README.md)).

### Reproduce champion submission

<table>
<tr>
<td width="50%">

**Notebook** *(verification / review)*

```bash
jupyter notebook reproduce.ipynb
```

Run All cells → writes `outputs/agent_b6_hwy_model_up.csv`

</td>
<td width="50%">

**CLI** *(same output)*

```bash
python -m experiments.gen_batch6
```

~3–5 min CPU · byte-identical to submitted file

</td>
</tr>
</table>

### Optional — build source bundle for competition upload

```bash
bash make_submission_bundle.sh   # → submission_bundle.zip
```

---

## Project structure

```
traffic-prediction/
│
├── reproduce.ipynb                 ★ Verification entry point
├── config.yaml                     Paths, seed, data contract
├── requirements.txt
│
├── src/                            Core pipeline
│   ├── data.py                     Load & validate
│   ├── features.py                 Leakage-safe encodings
│   ├── model.py                    LightGBM trainer
│   ├── corrector.py                Day-over-day corrector
│   └── geohash_decode.py           Geohash → lat/lon
│
├── experiments/
│   ├── gen_batch6.py               ★ Champion generator
│   ├── gen_batch2.py               raw_cell lookup
│   ├── gen_batch4.py               corr2 trainer
│   └── diag*.py                    Leak hunts & probes
│
├── notebooks/                      EDA (layers 1–3)
├── docs/
│   ├── APPROACH.md                 Full methodology
│   └── EXPERIMENT_LOG.md           Assumptions & findings
│
├── dataset/                        Competition CSVs (gitignored)
├── outputs/
│   └── agent_b6_hwy_model_up.csv   Champion submission
│
└── scripts/setup_env.sh            .venv + Jupyter kernel
```

---

## Documentation

| Read this | For |
|:----------|:----|
| [**docs/APPROACH.md**](docs/APPROACH.md) | Complete methodology, EDA, features, all 8 batches |
| [**docs/EXPERIMENT_LOG.md**](docs/EXPERIMENT_LOG.md) | Every assumption tested — confirmed vs rejected |
| [**experiments/README.md**](experiments/README.md) | What each script does |
| [**reproduce.ipynb**](reproduce.ipynb) | Runnable end-to-end pipeline |

---

## Key lessons

1. **Frame it as forecasting** — random splits and nighttime validation folds gave false signals.
2. **Preserve slot-level detail** — highways have strong diurnal patterns; averaging slots destroyed daytime R².
3. **Blend, don't chase one model** — lookup + GBDT + correctors each fix different errors.
4. **Segment at blend time** — RoadType structure (flat vs peaky) beat more trees or more features.
5. **Morning ≠ daytime** — day-49 morning is busier, but scaling daytime predictions by that ratio hurt badly.
6. **Use the right validator** — for daytime, the public leaderboard was the only trustworthy judge.

---

## Tech stack

`Python 3.11` · `pandas` · `numpy` · `scikit-learn` · `LightGBM` · `CatBoost` · `Jupyter`

---

## License

[MIT](LICENSE) — Adya, 2026
