# Traffic Demand Prediction — Approach

## 1. What the data actually is (and why it changes the strategy)

The dataset is a **spatiotemporal panel**, unique per `(day, timestamp, geohash)`:

- `train.csv` = day 48 (all 96 fifteen-minute slots, 00:00–23:45) + day 49 early slots (00:00–02:00, 9 slots).
- `test.csv`  = day 49, 02:15–13:45 (47 slots). It is a **strict forward forecast**, not a random split.
- Target `demand` ∈ (0, 1], heavily right-skewed (skew ≈ 3.7), max = 1.0 (normalized demand).
- 1180 / 1190 test geohashes are seen in train; 10 are cold-start.
- Missingness is low and identical across train/test: RoadType 0.78%, Temperature 3.23%, Weather 1.03%.

The decisive constraint is that there are **only two days**. A model trained purely on day 48
cannot learn day-over-day dynamics, because day 48 has no "yesterday". The standard
"LightGBM + target encoding + Optuna + blend" recipe therefore leaves the single largest
signal on the table. That signal has to be injected analytically.

## 2. Architecture (hybrid)

```
prediction = clip( base_level * today_factor , 0, 1 )

base_level   = 0.75 * yesterday_same_slot(day48)  +  0.25 * gbdt_level
               (falls back to pure gbdt_level where no yesterday value exists)

gbdt_level   = ensemble[ LightGBM, CatBoost ] predicting the typical demand of a
               (geohash, slot) from spatial geohash-prefix target encodings, the
               diurnal curve, and covariates. Trained on day 48 (full coverage).

today_factor = per-geohash multiplicative level shift = (day-49 morning mean)
               / (day-48 same-morning-slots mean), shrunk toward 1.0 by support
               count, globally damped by FACTOR_DAMP, clipped to [0.25, 4.0].
```

Rationale per component:

- **yesterday_same_slot** — seasonal-naive predictor. The strongest per-cell estimate; for
  day-49 it is literally day-48 demand at the same geohash and time-of-day.
- **gbdt_level** — covers the ~11% of test cells with no yesterday value (incl. the 10 cold
  geohashes). Its strength is spatial pooling via geohash prefixes (p3/p4/p5), so it
  generalizes to **unseen** geohashes (GroupKFold R² ≈ 0.72). It also smooths the noisy
  single yesterday observation when blended at weight 0.25.
- **today_factor** — captures "is today running hotter/colder than yesterday for this
  location", measured from the only day-49 history available (the morning). This is the
  component a 2-day GBDT cannot learn, and it is the largest lever.

## 3. Feature engineering

- **Time**: minutes-since-midnight, 15-min slot index (0–95), hour, cyclical `sin/cos` of slot.
- **Space**: geohash plus prefixes p3/p4/p5 (geohash is hierarchical, so prefixes pool
  geographically adjacent cells — essential for stable estimates and cold-start fallback).
- **Spatial × temporal target encodings** (smoothed, empirical-Bayes): geohash, p3, p4, p5,
  and p4×slot. Computed with **GroupKFold-by-geohash out-of-fold** during validation so the
  encoding must generalize spatially and never leaks a row's own target.
- **Covariates**: RoadType, NumberofLanes, LargeVehicles, Landmarks, Weather (native
  categoricals; NaN → "missing"), Temperature (passed raw; trees handle NaN).

## 4. Leakage controls (this is where hackathons are won or lost)

- Target encodings use **out-of-fold, group-by-geohash** during validation, and a smoothed
  full-fit map for the final test transform. The level model is fit on **day 48 only**, so
  per-`(geohash, slot)` aggregates never see the predicted day-49 row.
- `today_factor` for the **test** uses only the observed day-49 morning vs day-48 — it never
  touches a test target.
- For the **reported validation number**, `today_factor` is rebuilt **leave-one-slot-out**
  (`today_factor_loo`) so a held-out row's own value cannot inform its own factor. Without
  this the holdout R² is inflated.

## 5. Validation and the real risk

The only day-49 ground truth available is the 9 morning slots, so the honest forward holdout
is **nighttime**:

| configuration                                   | forward-holdout R² (nighttime, leakage-free) |
|-------------------------------------------------|----------------------------------------------|
| seasonal-naive level only                       | ~0.53                                        |
| gbdt_level only (no factor)                     | ~0.61                                        |
| hybrid base, no factor                          | ~0.61                                        |
| hybrid base × today_factor (DAMP=0.50)          | ~0.84                                        |
| hybrid base × today_factor (DAMP=0.75, default) | ~0.87                                        |
| hybrid base × today_factor (DAMP=1.00)          | ~0.86                                        |

GBDT level model, **GroupKFold-by-geohash on day 48** (cold-geohash generalization): R² ≈ 0.72.

**The real risk is NOT the public/private shakeup** (the warned-about trap). It is the
**nighttime → daytime horizon gap**: the `today_factor` is measured over 00:00–02:00 and
applied across an 11.5-hour daytime horizon it cannot observe. The morning level shift decays,
so the **live daytime score will sit below the nighttime numbers above**, and the optimal
`FACTOR_DAMP` is likely **below** the nighttime peak of 0.75.

`FACTOR_DAMP` is the one knob that genuinely warrants leaderboard calibration — it is a single,
well-motivated parameter that cannot be tuned locally, which is categorically different from
blind public-LB climbing. Recommended probe order under a tight submission budget:

1. `submission_damp50.csv` (DAMP=0.50) — balanced first probe.
2. `submission_damp75_default.csv` (DAMP=0.75) — if this beats 0.50 live, the morning signal
   persists into daytime; if not, move toward 0.25–0.50.
3. Keep `submission_damp00_nofactor.csv` (DAMP=0.0, R²≈0.61 nighttime) as a robustness floor.

## 6. Tools

Python 3.12, pandas, numpy, scikit-learn, LightGBM 4.6, CatBoost 1.2. No internet or external
data used. GPU is parametrized via `USE_GPU` (LightGBM `device='gpu'`, CatBoost
`task_type='GPU'`); the pipeline runs on CPU in a couple of minutes.

## 7. How to run

```bash
pip install pandas numpy scikit-learn lightgbm catboost
# place train.csv / test.csv / sample_submission.csv under ./dataset/
python traffic_demand_pipeline.py        # prints forward-holdout R², writes submission.csv
```

Knobs at the top of `traffic_demand_pipeline.py`: `FACTOR_DAMP`, `YS_WEIGHT`, `FACTOR_CLIP`,
`USE_GPU`, `USE_CATBOOST`. Output `submission.csv` is `41778 × 2` (`Index`, `demand`), Index
order asserted to match `test.csv`, predictions clipped to [0, 1].

## 8. Honest extensions not included (and why)

- A GBDT trained directly on day-49 morning with yesterday/factor features as inputs: only
  ~7.8k nighttime rows in one regime — high overf_t risk for daytime extrapolation. Left out
  to protect against the horizon gap rather than chase the nighttime holdout.
- XGBoost is wired-ready as a third level learner but adds little over LightGBM+CatBoost on
  the level sub-task (which drives only the fallback cells + 25% blend weight).
