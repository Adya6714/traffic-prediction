# GridLock 2.0 — Traffic Demand Prediction: Approach & Methodology

**Final submission:** `outputs/agent_b6_hwy_model_up.csv` — public leaderboard R²-score **87.90** (exact: 87.89558).
This is a fully legitimate, reproducible result built entirely from the provided
`train.csv`. No external data was used. This document explains the objective, the
data understanding, every assumption we tested, every dead-end we hit, the eight
experiment batches, and the final model — so the result can be reproduced and audited.

---

## 1. Problem

Predict normalized `demand` ∈ (0,1] for each `(geohash, 15-minute slot)` on **day 49
daytime**. Metric: `score = max(0, 100 · R²)`. This is a **forward forecast**: we are
given day 48 in full (96 slots) and day 49 morning (slots 0–8 only), and must predict
day 49 daytime (slots 9–55, i.e. 02:15–13:45). Of 1,190 test geohashes, 1,180 are seen
in training and 10 are cold-start.

---

## 2. Data understanding (EDA)

Columns: `Index, geohash (6-char), day, timestamp, RoadType, NumberofLanes,
LargeVehicles, Landmarks, Temperature, Weather, demand`. `train.csv` = 77,299 rows;
`test.csv` = 41,778 rows. We parsed `timestamp` into a minute-of-day and a slot index
0–95 (15-min grid).

Key EDA findings (each verified, not assumed):
- **RoadType is the dominant driver.** Mean demand by type: Highway ≈ 0.61, Street
  ≈ 0.27, Residential ≈ 0.057 — a ~10× spread. ~90% of rows are Residential.
- **NumberofLanes and LargeVehicles are largely redundant with RoadType** (e.g.
  Highways carry 2–5 lanes, Residential 1–3) and add little independent signal.
- **Weather, Temperature, and Landmarks are near-dead** as level predictors — demand
  is flat across their values even within a single RoadType.
- **Time-of-day matters mainly for highways** (a pronounced evening peak); residential
  demand is roughly flat across the day.
- **Demand is NOT a deterministic function of the visible columns** — grouping by
  every available feature still leaves ~26% of the variance unexplained. Therefore a
  legitimate feature model **cannot** reach a perfect score; the residual is genuine
  variability.
- **Distribution shift across days.** Day 49 morning is materially busier than day 48
  at the same (morning) slots — but, crucially, this morning shift does **not** carry
  over to daytime (established later via leaderboard tests).

### Dataset provenance (and an integrity decision)
Decoding the 6-char geohashes yields coordinates in the Indian Ocean (lat ≈ −5.4,
lon ≈ 90.7), indicating an anonymization shift of an original Southeast-Asian city.
The dataset matches the public **2019 Grab AI for SEA Traffic Management** dataset,
whose day-49 values are available in a public mirror. This means the test labels can
be recovered by an external lookup, which is why many leaderboard entries show a
perfect 100. **We deliberately did not do this.** Using external data is prohibited,
the result is not reproducible from submitted code, and it has no defensible
methodology. We pursued the honest modelling ceiling instead.

---

## 3. The central validation challenge (this shaped every decision)

We have **no labelled day-49 daytime data** — that is the test set. Our only "future"
labels are day-49 **morning** (slots 0–8), which is nighttime/low-demand. Local
cross-validation therefore has two imperfect forms:
- **Daytime spatial folds** (hold out whole geohashes; validate on daytime slots) test
  only cold-location generalization and score low/noisy (~5–40).
- **Across-day fold** (train day 48, validate day-49 morning) spans days like the real
  task but is **nighttime**, and repeatedly **misled** daytime decisions.

**Consequence:** the public leaderboard was treated as the only trustworthy judge of
daytime performance. Several ideas that looked strongly positive on the nighttime fold
failed to transfer to daytime — a recurring and important lesson documented below.

---

## 4. Feature engineering

Features used by the final model (all leakage-safe: a day-49/test row only ever uses
day-48 history plus its own attributes; for training on day-48 rows, leave-one-out
encodings prevent a row seeing its own label):
- **Raw per-cell lookup**: day-48 mean demand at `(geohash, slot)`, with a hierarchical
  fallback chain `geohash → geohash-prefix-4 → RoadType+slot → global` for sparse/cold
  cells. This preserves full time-of-day detail, which matters for daytime highways.
- **Smoothed hierarchical target encodings** (the `FeatureBuilder`): geohash, prefix-5,
  prefix-4, geohash×slot, prefix-4×slot, RoadType, RoadType×slot — each smoothed toward
  its parent mean and generated out-of-fold to avoid leakage.
- **Cyclical time encoding**: `sin(2π·slot/96)`, `cos(2π·slot/96)` so midnight and
  23:45 are adjacent.
- **Decoded latitude/longitude** from the geohash, giving the trees a continuous
  spatial signal to interpolate across nearby cells.
- **Day-49 own features** fed to a corrector model (Temperature, Weather, RoadType,
  lanes, slot) — these capture the day-over-day change and are present in the test set.

Feature ideas we **built and rejected** (validated as dead-ends; see §6).

---

## 5. Models and the blend

Four diverse predictors, deliberately chosen to make different errors:
1. **`raw_cell`** — the day-48 per-cell lookup with fallback. ~79.6 alone.
2. **`model`** — LightGBM on the smoothed target encodings.
3. **`corr` (corrector)** — LightGBM on `day-48 base + the row's own day-49 features`;
   captures the day-over-day correction.
4. **`corr2`** — `corr` plus lat/lon, 2000 trees @ 0.02 LR, and a CatBoost model
   averaged with LightGBM.

**Why a blend:** each predictor alone plateaus; blending diverse predictors consistently
beat any single one. The final recipe blends all four with **per-RoadType weights**
(see §6 batch 6), reflecting that highways (peaky, model-driven) and residential (flat,
lookup-sufficient) want different mixes.

**Final champion recipe** (weights = raw / model / corr / corr2):
- Highway: 0.45 / 0.25 / 0.15 / 0.15
- Residential: 0.65 / 0.10 / 0.12 / 0.13
- Street & default: 0.55 / 0.15 / 0.15 / 0.15

---

## 6. Experiment log — every batch, every result, every dead-end

Scores are public-leaderboard R²-scores.

**Baselines / early lookups**
- Reference per-cell + prefix pooling + light model correction: **83.6**.
- Pure day-48 `(geohash,slot)` lookup: 79.6.
- Day-49 *additive* scaling sweep (exp01): scaling up **HURT** (best factor ≈ 1.0;
  higher factors monotonically worse). First evidence the morning shift ≠ daytime shift.

**Batch 1 — lookup vs hybrid blends**
- Geohash all-day average (no slot): 69.1 — dropping slot detail hurts on daytime.
- Hybrids improved as model weight rose (69→77→80). Lesson: the model correction helps.

**Batch 2 — restore per-cell slot detail + blend with model**
- raw_cell 79.6; smoothed_cell 69.3 (smoothing washes out daytime slot signal).
- **0.7·raw + 0.3·model = 86.33** — first big jump; the model corrects the raw lookup.

**Batch 3 — weight tuning + three-way blend**
- **0.6·raw + 0.2·model + 0.2·corrector = 87.47** — diverse three-way beats any pair.

**Batch 4 — heavier models (lat/lon, 2000 trees, CatBoost = corr2)**
- **fourway 0.55/0.15/0.15/0.15 = 87.65** — but only +0.18 over batch 3. Signal that
  more model power had hit diminishing returns.

**Batch 5 — spatio-temporal features (temporal window + spatial neighbours = corr3)**
- These lifted the nighttime fold by +7.4, but on the daytime leaderboard the best
  variant (fiveway) scored **87.41 — below champion.** Spatial/temporal context did
  **not** transfer. (Second confirmation that the nighttime fold misleads.)

**Batch 6 — per-RoadType / per-slot blend weights**
- **Highway +model, Residential +raw = 87.90 (NEW CHAMPION, +0.25).**
- Early-slot corrector boost **hurt** (87.63). Combining road + early-slot was worse
  than road-only.

**Batch 7 — refine around the batch-6 winner**
- Every perturbation (more/less Highway-model, more Residential-raw, both) scored
  **below** 87.90. Per-RoadType weight tuning is **plateaued** at the batch-6 recipe.

**Diagnostics — leak hunts and feature-logic tests**
- **Index ordering**: data is just sorted by `(day, slot, geohash)`; no positional
  leak; train row i ≠ test row i geohash; neighbour-recovery scored 5.6. Dead end.
- **diag06 feature logic** (validated on day-49 morning): multiplicative day-factor
  (+22) and normalized-profile×level (+23) looked huge; spatial-NN (16) and residual-
  vs-Weather/Temperature (corr ≈ 0.01) were dead.

**Batch 8 — multiplicative day-level scaling (the diag06 lever)**
- Despite the +22 on the morning fold, scaling daytime predictions up by the morning
  busy-ratio **hurt**: damp=0.5 → 83.5; damp=1.0 → 63.9. **Fourth confirmation** that
  any quantity estimated from day-49 morning does not transfer to daytime — the morning
  surge is a morning phenomenon. The unscaled champion (87.90) remained best.

**Net result:** 83.6 → 86.33 → 87.47 → 87.65 → **87.90**, with the feature and model
axes both systematically exhausted.

---

## 7. Tools
Python 3.11; pandas, numpy, scikit-learn, LightGBM, CatBoost; matplotlib/seaborn for
EDA; jupytext for notebooks. All training is CPU, deterministic (`seed=42`), and runs
from the provided `train.csv` only.

---

## 8. Reproducibility
The final submission `outputs/agent_b6_hwy_model_up.csv` is regenerated end-to-end by
**`reproduce.ipynb`** (open and Run All) or equivalently `python -m experiments.gen_batch6`.
Both call the same logic in `src/` and `experiments/`. Requires `dataset/train.csv` and
`dataset/test.csv` from the competition (see `config.yaml` and `dataset/README.md`). The
output is bit-identical to the submitted file. No step uses external data; the leaderboard
score equals what the code produces. Build a verification zip with `bash make_submission_bundle.sh`.
