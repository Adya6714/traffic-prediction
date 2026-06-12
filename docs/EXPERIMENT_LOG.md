# ASSUMPTIONS LOG

The single source of truth for what we believe about this problem, what we have
tested, and what to try next. Update this every time we learn something.

How to read the STATUS column:

- CONFIRMED = tested, it held up.
- REJECTED = tested, it was false.
- OPEN = believed but not yet tested.
- PARTIAL = some evidence, not conclusive.

---

## 0. PROBLEM FACTS (not assumptions — these are measured truths)

| #   | Fact                                                                                                                                | Evidence      |
| --- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| F1  | Test is a forward forecast: train=day48(full)+day49 morning(00:00-02:00); test=day49 daytime(02:15-13:45).                          | data profile  |
| F2  | Only TWO days of data exist.                                                                                                        | data profile  |
| F3  | Target `demand` is in (0,1], very right-skewed (skew 3.73), mean ~0.094, median ~0.048.                                             | Layer 1.1     |
| F4  | Test set = 1180/1190 geohashes already seen in train; only 10 are cold-start.                                                       | data profile  |
| F5  | A perfect score (100 = R²=1.0) is achievable -> demand is largely DETERMINED by the visible columns. This is a constructed dataset. | leaderboard   |
| F6  | Missing values: RoadType 0.78% (600 rows), Temperature 3.23% (2495), Weather 1.03% (797).                                           | Layer 1.3/1.4 |

---

## 1. WHAT DRIVES DEMAND (single-variable assumptions, from Layer 2 EDA)

| #   | Assumption                                                                                                                                                                                 | Status                  | Evidence         | If it fails / next step                                                                  |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------- | ---------------- | ---------------------------------------------------------------------------------------- |
| A1  | **RoadType is a strong driver.** Highway(0.61) >> Street(0.27) >> Residential(0.057). ~10x range.                                                                                          | CONFIRMED               | Layer 2.2        | Strongest single signal. Must be used. Handle the 600 NaN RoadType rows carefully.       |
| A2  | **NumberofLanes is a strong driver, as a CLIFF not a slope.** Lanes 1-3 ~0.08; lanes 4-5 jump to ~0.60.                                                                                    | CONFIRMED               | Layer 2.2        | Encode as "is 4-5 lanes" flag, not raw number, OR keep raw and let trees find the cliff. |
| A3  | **LargeVehicles matters moderately.** Allowed(0.13) vs Not Allowed(0.074), ~2x.                                                                                                            | CONFIRMED               | Layer 2.2        | Keep, but check in A8 whether it's just a proxy for highways.                            |
| A4  | **Landmarks does NOT matter.** Yes(0.093) vs No(0.096) — flat.                                                                                                                             | REJECTED                | Layer 2.2        | Drop it.                                                                                 |
| A5  | **Weather does NOT matter.** Confirmed dead even INSIDE each road type: Highway 0.60-0.62, Residential ~0.057 across all weathers.                                                         | REJECTED (final)        | Layer 2.2 + 3.A3 | Drop Weather entirely.                                                                   |
| A6  | **Time-of-day matters ONLY for highways.** Overall flat because 90% is Residential. Highway demand rises through the day and spikes in the evening (slots 72-88); Residential/Street flat. | CONFIRMED (conditional) | Layer 3.A4       | Build slot feature; interact slot with RoadType.                                         |
| A7  | **Temperature does NOT matter.** Flat across range; spike was noise.                                                                                                                       | REJECTED                | Layer 2.3        | Drop it.                                                                                 |

---

## 2. CROSS-VARIABLE assumptions (Layer 3 — TO TEST NEXT)

| #   | Assumption                                                                             | Status                                           | Why we think so | How to test                                                                                                                                                                                                              |
| --- | -------------------------------------------------------------------------------------- | ------------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A8  | **Highways == 4-5 lane roads == LargeVehicles allowed.** The strong variables overlap. | CONFIRMED                                        | Layer 3.A1      | Highway=2-5 lanes+Allowed; Residential=1-3 lanes; Street=1 lane+NotAllowed. Redundant — trees will untangle; don't over-engineer.                                                                                        |
| A9  | **Weak variables matter within a road type.**                                          | REJECTED (weather/temp); CONFIRMED (time-of-day) | Layer 3.A3/A4   | Only time-of-day survives, and only for highways.                                                                                                                                                                        |
| A10 | **Time-of-day matters for highways specifically.**                                     | CONFIRMED                                        | Layer 3.A4      | Highway has clear daily pattern + evening spike.                                                                                                                                                                         |
| A11 | **(RoadType, NumberofLanes,...) nearly DETERMINE demand.**                             | REJECTED                                         | Layer 3.B1      | Grouping by all visible columns only cuts spread from 0.42 to 0.26 of overall std. NOT a lookup table — irreducible variation remains. The 100-scorer found structure we have not, OR exploits per-cell history heavily. |

---

## 3. SPATIAL / TEMPORAL STRUCTURE assumptions (from baselines, Stage 3)

| #   | Assumption                                                                                                                                                                                                                                                                          | Status               | Evidence                                                                                                      | Next step                                                                                                                                                                                                                                                                                 |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A12 | **Demand level is idiosyncratic per geohash.** Known location -> predictable on a new day (geohash_mean nighttime=65.6). Unknown location -> not predictable from neighbors (geohash_mean spatial=0).                                                                               | CONFIRMED            | Stage 3 baselines                                                                                             | For the 1180 seen geohashes, lean on their own history. For the 10 cold ones, fall back to road attributes (A1-A3).                                                                                                                                                                       |
| A13 | **Per-cell day-over-day shift is real, not noise.** day48 -> day49 same (geohash,slot) only scores ~49 with corr 0.79; demand moves between days.                                                                                                                                   | CONFIRMED            | Stage 3 probe                                                                                                 | See A16 — the shift is SYSTEMATIC and UPWARD, not random.                                                                                                                                                                                                                                 |
| A16 | **Day 49 is systematically BUSIER than day 48** by ~1.7x — at NIGHT.                                                                                                                                                                                                                | REJECTED for daytime | Layer 3.B3 + exp01                                                                                            | The ~1.7x nighttime ratio does NOT hold in daytime. Best LB factor = 1.0-1.2 (negligible). DEAD END — do not scale for day 49.                                                                                                                                                            |
| A17 | **Specific (geohash, slot) cells are fairly stable** (median within-cell std 0.025 vs overall 0.142).                                                                                                                                                                               | CONFIRMED            | Layer 3.B2                                                                                                    | For the 1180 seen geohashes, their own day-48 (geohash,slot) value is a strong base prediction.                                                                                                                                                                                           |
| A18 | **Spatial PREFIX pooling + smoothing beats raw per-cell lookup** by ~4 LB points.                                                                                                                                                                                                   | CONFIRMED (by gap)   | exp01 control 79.6 vs reference 83.6                                                                          | The real model MUST: (a) pool across geohash prefixes (neighboring cells) and (b) shrink a single noisy day-48 observation toward a pooled/smoothed estimate. This is the next source of points.                                                                                          |
| A19 | **OOF encoding makes the model DISTRUST per-cell features that are actually gold at test time.** With whole-geohash holdout, te_geohash_slot is always a fallback during training, so the model learns to ignore it — but for the 1180 SEEN test geohashes it carries precise info. | CONFIRMED            | Stage 6 model: daytime~5, across_night 63.2 (below geohash_mean 65.6); te_geohash_slot importance near-bottom | Fix the train/test feature mismatch: either (a) a softer OOF that lets a geohash partially see its own history, or (b) a HYBRID = strong per-cell day-48 lookup blended with the model (model handles cold cells, lookup handles seen cells). Hybrid matches what the 83.6 reference did. |
| A20 | **A STABLE geohash all-day average beats per-(geohash,slot) detail.** geohash mean corr 0.826 vs geohash_slot 0.769. slot_adjust sweep is monotonic DOWN (0.0->65.6, 1.0->61.1).                                                                                                    | CONFIRMED            | diag02 + slot_adjust sweep                                                                                    | Lookup base = geohash all-day mean, slot_adjust=0. A single noisy day-48 slot is worse than pooling ~96 obs. (General principle: pooled stable estimate > precise noisy one.)                                                                                                             |
| A21 | **HYBRID (lookup + small model correction) beats both alone.** hybrid_blend (0.8*lookup+0.2*model for seen) across_night=69.1 vs lookup 65.6 vs model 63.2.                                                                                                                         | CONFIRMED            | Stage 7 predict                                                                                               | The model adds value as a CORRECTION on top of the stable lookup, not as the primary. Tune blend weight, then submit.                                                                                                                                                                     |

---

## 4. TARGET-SHAPE assumptions

| #   | Assumption                                                                                 | Status          | Evidence  | Next step                                                                 |
| --- | ------------------------------------------------------------------------------------------ | --------------- | --------- | ------------------------------------------------------------------------- |
| A14 | **The right-skew (3.73) may hurt the model; modeling log(demand) may help.**               | OPEN            | Layer 1.1 | Train once on demand, once on log1p(demand)+invert, compare daytime_mean. |
| A15 | **Clip predictions to (0,1].** demand is bounded; out-of-range predictions only add error. | OPEN (free win) | F3        | Always clip final output.                                                 |

---

## 5. LEADERBOARD RESULTS LOG (what we have actually submitted)

| Submission                                | Idea tested                                          | LIVE score                                |
| ----------------------------------------- | ---------------------------------------------------- | ----------------------------------------- |
| reference nofactor                        | level model, no day-over-day momentum                | **83.63** (current best / bar)            |
| reference damp50                          | half-weight morning momentum                         | 79.33                                     |
| reference damp75 / submission.csv         | strong morning momentum                              | 66.81                                     |
| reference damp100                         | full morning momentum                                | 50.63                                     |
| exp01 shift 1.00                          | our from-scratch level predictor, no shift (CONTROL) | 79.61                                     |
| exp01 shift 1.20                          | +20% day-49 scale                                    | 79.65 (best of sweep, but ~tie with 1.00) |
| exp01 shift 1.40                          | +40% day-49 scale                                    | 74.81                                     |
| exp01 shift 1.60                          | +60% day-49 scale                                    | 65.85                                     |
| exp01 shift 1.72                          | +72% day-49 scale (nighttime ratio)                  | 58.70                                     |
| batch1 lookup_only (geohash avg, no slot) | dropped slot info                                    | 69.08                                     |
| batch1 hybrid_w070                        | 0.7 geohash-avg + 0.3 model                          | 77.19                                     |
| batch1 hybrid_w050                        | 0.5 geohash-avg + 0.5 model                          | 79.92                                     |
| batch1 hybrid_hard                        | geohash-avg seen / model cold                        | 68.89                                     |
| batch2 raw_cell                           | raw (geohash,slot), full slot detail                 | 79.56                                     |
| batch2 smoothed_cell                      | (geohash,slot) smoothed toward geohash mean          | 69.30                                     |
| **batch2 blend_raw_w070**                 | **0.7 raw_cell + 0.3 model**                         | **86.33 (NEW BEST)**                      |
| batch2 blend_raw_w050                     | 0.5 raw_cell + 0.5 model                             | 86.04                                     |
| batch2 blend_smooth_w050                  | 0.5 smoothed + 0.5 model                             | 84.04                                     |
| batch2 blend_smooth_w030                  | 0.3 smoothed + 0.7 model                             | 81.48                                     |

DAYTIME LEADERBOARD LESSONS (override the nighttime fold, which kept misleading):

- RAW per-cell (geohash,slot) BEATS smoothed on daytime (79.6 vs 69.3). Smoothing
  washes out real daytime slot signal (esp. highways). A20 was a NIGHTTIME artifact.
- The model correction adds real daytime value: raw 79.6 -> blend 86.3 (+6.7).
- Best blend weight ~0.7 raw + 0.3 model. NEW BEST = 86.33, beats reference 83.6.
- The across_night fold is NOT trustworthy for daytime decisions. Use the LB.

BATCH 3 (weight tuning + corrector + three-way blend):
| b3_threeway (0.6 raw + 0.2 model + 0.2 corrector) | **87.47 (NEW BEST)** |
| b3_raw_w065 | 86.61 | b3_raw_w068 | 86.47 | b3_raw_w072 | 86.14 |
| b3_corrblend_w060 | 86.11 | b3_raw_w075 | 85.80 | b3_corrblend_w070 | 85.09 |
LESSON: three-way blend of DIVERSE predictors (raw cell + generic model +
day-49-feature corrector) beats any pair. Legitimate ensembling works.

DATASET PROVENANCE (forum finding, A25): the competition data is an UNMODIFIED
copy of the public 2019 Grab AI for SEA Traffic dataset. Day49 daytime labels
exist in a public GitHub repo. The 100 scores are dataset LOOKUPS, not models.
DECISION: we do NOT do the lookup. It violates rules (external data prohibited),
is non-reproducible from our code (organizers review code -> disqualification),
and has no defensible story for the prototype/presentation round. We pursue the
honest ML ceiling, which legitimate participants report at ~92-93.

BATCH 4 (lat/lon + 2000 trees + CatBoost corr2):
| b4_fourway (0.55 raw + 0.15 model + 0.15 corr + 0.15 corr2) | **87.65 (BEST)** |
| b4_threeway_corr2 | 87.34 | b4_blend_raw_corr2_w055 | 86.28 |
| b4_blend_raw_corr2_w060 | 85.90 | b4_corr2_only | 84.58 |
LESSON: heavy model power (lat/lon, 2000 trees, CatBoost) gained only +0.18 over
b3_threeway. The lever is NOT more model power.

BATCH 5 (corr3 = corr2 + spatio-temporal features):
| b5_fiveway | 87.41 | b5_threeway_corr3 | 86.98 | b5_corr3_only | 85.67 |
| b5_blend_raw_corr3_w050 | 85.57 | b5_blend_raw_corr3_w055 | 85.24 |
LESSON: spatio-temporal features lifted the NIGHTTIME fold +7.4 (diag05) but did
NOT beat the champion on the daytime leaderboard. THIRD confirmation that the
nighttime fold misleads on daytime. PLATEAU at ~87.65 confirmed: progressively
heavier machinery (more trees, CatBoost, lat/lon, spatial+temporal feats) moved
the score 86.33 -> 87.47 -> 87.65 -> stuck. Pushing past ~88 honestly needs a
STRUCTURALLY different idea (per-slot/roadtype weights, live day-49 trajectory,
rank target), not more model power. CHAMPION REMAINS b4_fourway = 87.65.

TWO KEY LESSONS FROM exp01:

1. The day-49 upward shift (A16) does NOT hold in daytime. Best factor is 1.0-1.2
   (negligible gain); anything higher hurts badly. Nighttime patterns do not
   transfer to daytime — same lesson as the momentum disaster. "Scale up for day
   49" is a CONFIRMED DEAD END. Do not revisit.
2. Our from-scratch level predictor (79.6) is ~4 points BELOW the reference (83.6)
   despite being "the same idea". The difference: the reference uses geohash-PREFIX
   spatial pooling + smoothing (shrink single noisy day-48 obs toward a pooled
   estimate). THIS is where the 4 points live. The real model must include
   prefix pooling + smoothing. (See A18.)

KEY LESSON: morning "today vs yesterday" momentum HURTS daytime prediction. The
more we trusted it, the worse we scored. Level beats momentum. (This is why A13
must be modeled as a _learned, damped_ shift, not raw morning extrapolation.)

---

## 6. UNTESTED IDEAS PARKING LOT (things to try if we stall)

LEAK-HUNT FINDINGS (diag03):

- **A22 BREAKTHROUGH: day-49's OWN features recover the day-over-day change.**
  Predicting day-49 morning from [day48 base + day49 Temperature/Weather/RoadType/
  lanes/slot] scores 83.2, vs base-alone 49.3. The corrector uses day-49 feature
  LEVELS (not deltas: temp-change corr with demand-change is ~0.017). Build the
  real model as: demand ~ f(day48_base_cell, own RoadType, slot, Temperature,
  Weather, lanes). Train on day48, apply to test with day-49 features. This is
  cleaner than blending a fixed lookup with a generic model.
- **A23: Index has lag-1 autocorrelation 0.32** (not ~0). Demand has ordering
  structure — rows near each other in Index have correlated demand. Possible
  hidden-key leak (rows grouped by geohash/region in Index order?). INVESTIGATE:
  this is the most likely remaining route to 100 beyond the ~83 feature ceiling.
  > > RESOLVED (diag04): DEAD END. Data is sorted by (day, slot, geohash). The
  > > autocorrelation is just "similar cells sit adjacent" — removing the per-cell
  > > mean does NOT kill it (rises to 0.56), and Index-neighbour prediction scores
  > > only 5.6. Train row i and test row i are NOT the same geohash (0.001 match).
  > > No positional leak. The 100 is likely public-LB overfit or an access we lack.
  > > REALISTIC GOAL: maximize the feature/base signal (current best 86.3), not 100.
- **A24: demand is continuous** (76715/77299 unique, no integer quantization).
  Not a discrete formula; there is a continuous/noise component that caps pure
  feature-modelling. 100 therefore likely needs the Index/ordering structure (A23),
  not more features.

MODEL STRATEGY DECISIONS (recorded so we don't over-engineer):

- **LightGBM FIRST, alone.** The relationship is simple and dominated by
  te_roadtype (corr 0.87). XGBoost/CatBoost would learn the same thing and
  predict near-identically, so ensembling gains ~0.1-0.3 pts at most. Only add a
  2nd model + blend IF we plateau and need the last point. Measure, don't assume.
- **GPU/T4 NOT needed.** 77k rows x 14 features is tiny; LightGBM trains in
  seconds on CPU. GPU overhead can even be slower at this scale. USE_GPU=False.
- **Log-transform (A14) is the better next experiment** than a 2nd model:
  train on log1p(demand), predict, invert. One-line change, may help the skew.

OTHER IDEAS:

- Treat the problem as near-deterministic: build a lookup of demand by
  (RoadType, NumberofLanes, slot, ...) and see how far pure grouping gets.
- For seen geohashes, use that geohash's own day-48 demand profile directly.
- Cold-start (10 geohashes): predict purely from road attributes.
- Try a transform other than log (sqrt, Box-Cox) if log helps but not enough.
- Investigate whether `Index` ordering leaks any structure.
- Check if Temperature/Weather differ between day48 and day49 in a way that
  explains the day-over-day shift (A13).
