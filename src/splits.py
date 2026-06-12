"""Stage 2 — Backtest splitter.

This file answers one question: given your training data, how do you carve out
a validation set that honestly estimates daytime performance?

Why this file exists before any model or feature code
------------------------------------------------------
In a forecasting problem, your validation scheme IS your experiment design.
If you build features and models while measuring against a leaky or wrong-regime
split, every conclusion you draw is contaminated. You lock down "how to measure
honesty" before you measure anything.

The three fold types in this file
----------------------------------
1. DAYTIME folds on day 48  [PRIMARY]
   Cut day 48 at several points in the early morning. Train on hours 00:00→cut,
   validate on slots matching the real test window (02:15→13:45 = slots 9→55).
   This is the right *regime* — daytime — even though it's within a single day.
   The leaderboard proved daytime regime matters more than across-day structure:
     nofactor (no trend)  LIVE=83.6   damp75 (morning trend)  LIVE=66.8
   The morning trend looked great on nighttime holdout (R2 0.87) but destroyed
   daytime performance. Daytime folds on day 48 would have caught this.

2. ACROSS-DAY holdout on day-49 morning  [SECONDARY / SANITY CHECK]
   Train on all of day 48, validate on day-49 slots 0→8 (00:00→02:00).
   This is the "across-day" structure, but it's nighttime only. We now know it's
   an optimistic and misleading signal for daytime. Keep it as a sanity check —
   if a model regresses badly here, something is wrong — but never use it as the
   primary decision signal.

3. COMBINED score  [REPORTING]
   Weighted average of daytime folds: the number you report for each experiment.

The golden rule encoded here
-----------------------------
Training data for fold N must never contain any row from the future relative to
that fold's validation window. No shuffling. No global statistics that include
validation rows. The split function returns DataFrame subsets (not just indices)
so downstream code cannot accidentally use the wrong rows.

Run directly to see a text timeline of all folds:
    python -m src.splits
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import load_raw


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# The real test window in slot-index terms. Slot = minute // 15.
# test starts at 02:15 (slot 9) and ends at 13:45 (slot 55).
TEST_SLOT_START = 9    # 02:15
TEST_SLOT_END   = 55   # 13:45

# The 9 train slots on day 49 that have ground truth (00:00 -> 02:00 = slots 0-8)
DAY49_TRAIN_SLOTS = list(range(0, 9))   # slots 0..8

# Early-morning cut points for the daytime folds. Each entry is the LAST slot
# that goes into training for that fold; validation is always TEST_SLOT_START..TEST_SLOT_END.
# Chosen so each fold has progressively more train context and the same val window.
#   cut=0  → train on slot 0 only       (00:00 context),  val 02:15-13:45
#   cut=4  → train on slots 0-4  (00:00-01:00 context),  val 02:15-13:45
#   cut=8  → train on slots 0-8  (00:00-02:00 context),  val 02:15-13:45
# The last cut (8) is the most realistic: 2 hours of morning context before the
# forecast horizon, matching what we have on day 49 (slots 0-8 in train).
DAYTIME_CUT_SLOTS = [0, 4, 8]


# --------------------------------------------------------------------------- #
# Data structure for a single fold
# --------------------------------------------------------------------------- #
@dataclass
class Fold:
    """A single train/validation split, ready to use.

    Fields
    ------
    name        : human-readable label e.g. "daytime_cut08"
    fold_type   : "daytime" or "across_day"
    train       : DataFrame of training rows (never contains val rows)
    val         : DataFrame of validation rows (the held-out future)
    cut_slot    : the slot index used as the cut (None for across_day fold)
    val_slots   : range of slot indices in the validation window
    """
    name:       str
    fold_type:  str
    train:      pd.DataFrame
    val:        pd.DataFrame
    cut_slot:   int | None
    val_slots:  tuple[int, int]   # (start, end) inclusive


# --------------------------------------------------------------------------- #
# Fold builders
# --------------------------------------------------------------------------- #
def _slot_to_time(slot: int) -> str:
    """Convert 0-95 slot index to 'H:MM' string for display."""
    total_min = slot * 15
    return f"{total_min // 60}:{total_min % 60:02d}"


def make_daytime_folds(
    train: pd.DataFrame,
    n_spatial_folds: int = 3,
    val_start: int = TEST_SLOT_START,
    val_end:   int = TEST_SLOT_END,
    seed: int = 42,
) -> list[Fold]:
    """Build SPATIAL daytime folds on day 48 (the corrected design).

    The naive 'cut day 48 in the morning and predict the afternoon' design is
    WRONG: it starves the model of daytime time-of-day information, so anything
    that depends on the diurnal shape (slot effects) scores ~0. We discovered
    this at the baseline stage — exactly what baselines are for.

    The realistic challenge in this problem is SPATIAL generalization: apply
    demand patterns learned from known locations to other/unseen locations
    (the test set has 10 cold-start geohashes and is scored on a daytime window).

    So each fold here:
      - splits day-48 GEOHASHES into train/holdout groups (GroupKFold style)
      - trains on ALL slots of the train geohashes (model learns the full
        diurnal shape from these locations)
      - validates on the DAYTIME slots (02:15-13:45) of the HELD-OUT geohashes
        (model must transfer the pattern to locations it never saw)

    This matches the test regime (daytime) AND the test challenge (new geography),
    while letting the model actually learn time-of-day effects.
    """
    d48 = train[train["day"] == 48].copy()
    geos = np.array(sorted(d48["geohash"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(geos)
    groups = np.array_split(geos, n_spatial_folds)   # k disjoint geohash sets

    folds = []
    for k, holdout_geos in enumerate(groups):
        holdout_set = set(holdout_geos)
        tr = d48[~d48["geohash"].isin(holdout_set)].copy()           # all slots, train geos
        val = d48[
            d48["geohash"].isin(holdout_set) &
            d48["slot"].between(val_start, val_end)                  # daytime slots only
        ].copy()

        # No geohash appears in both train and val -> true spatial holdout.
        assert len(set(tr["geohash"]) & set(val["geohash"])) == 0, \
            f"spatial fold {k}: geohash leak between train and val"

        folds.append(Fold(
            name       = f"daytime_spatial_f{k}",
            fold_type  = "daytime",
            train      = tr,
            val        = val,
            cut_slot   = None,
            val_slots  = (val_start, val_end),
        ))

    return folds


def make_across_day_fold(train: pd.DataFrame) -> Fold:
    """Build the across-day secondary holdout.

    Train on all of day 48. Validate on day-49 morning slots (ground truth
    available in train.csv because those 9 early slots were given to us).

    Use for:  detecting catastrophic regressions across days.
    Do NOT use for:  choosing between models or features. It lies about daytime.

    Why it lies:
        Our nighttime R² correlated NEGATIVELY with live daytime score:
        damp75 nighttime=0.87 → live=66.8
        nofactor nighttime=0.61 → live=83.6
    """
    d48  = train[train["day"] == 48].copy()
    d49e = train[
        (train["day"] == 49) &
        (train["slot"].isin(DAY49_TRAIN_SLOTS))
    ].copy()

    assert len(d49e) > 0, "No day-49 morning rows found — check train.csv"

    return Fold(
        name       = "across_day_night",
        fold_type  = "across_day",
        train      = d48,
        val        = d49e,
        cut_slot   = None,
        val_slots  = (DAY49_TRAIN_SLOTS[0], DAY49_TRAIN_SLOTS[-1]),
    )


def get_all_folds(train: pd.DataFrame) -> list[Fold]:
    """Return all folds: daytime primaries + across-day secondary.

    This is the single entry point downstream code should call.
    """
    daytime = make_daytime_folds(train)
    across  = make_across_day_fold(train)
    return daytime + [across]


# --------------------------------------------------------------------------- #
# Scoring helper
# --------------------------------------------------------------------------- #
def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """R² (coefficient of determination). Matches sklearn's implementation."""
    ss_res = ((actual - predicted) ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def competition_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """The exact competition metric: max(0, 100 * R²)."""
    return max(0.0, 100.0 * r2_score(actual, predicted))


def score_folds(
    folds: list[Fold],
    predict_fn,           # callable(val_df) -> np.ndarray of predictions
    target: str = "demand",
) -> dict:
    """Score every fold using predict_fn and return a results dict.

    Args
    ----
    folds      : list of Fold objects (from get_all_folds)
    predict_fn : a function that takes a validation DataFrame and returns
                 a numpy array of predicted demand values, one per row.
                 It must NOT look at val[target] — only the features.
    target     : name of the target column

    Returns
    -------
    dict with keys:
        "daytime_scores"    : list of competition scores for daytime folds
        "daytime_mean"      : mean of daytime scores  [PRIMARY METRIC]
        "daytime_std"       : std of daytime scores   (variance across cuts)
        "across_day_score"  : score for the nighttime across-day fold
        "per_fold"          : {fold.name: score} for all folds
    """
    results = {"per_fold": {}}
    daytime_scores = []

    for fold in folds:
        preds = predict_fn(fold.val)
        actual = fold.val[target].values
        score = competition_score(actual, preds)
        results["per_fold"][fold.name] = round(score, 4)

        if fold.fold_type == "daytime":
            daytime_scores.append(score)
        else:
            results["across_day_score"] = round(score, 4)

    results["daytime_scores"] = [round(s, 4) for s in daytime_scores]
    results["daytime_mean"]   = round(float(np.mean(daytime_scores)), 4)
    results["daytime_std"]    = round(float(np.std(daytime_scores)), 4)
    return results


# --------------------------------------------------------------------------- #
# Visualizer — always inspect your folds before trusting them
# --------------------------------------------------------------------------- #
def visualize_folds(folds: list[Fold]) -> None:
    """Print a text timeline of every fold so you can verify them visually.

    Read this output and ask: does each fold's val window look like the real
    test set? Is there any possible overlap between train and val? Does the
    progression of cuts make sense?
    """
    BAR = 48   # characters wide for the timeline
    SLOTS = 96

    def bar(lo, hi, char, width=BAR):
        """Fill positions [lo, hi] proportionally in a width-char string."""
        arr = ["."] * width
        for i in range(width):
            slot = int(i / width * SLOTS)
            if lo <= slot <= hi:
                arr[i] = char
        return "".join(arr)

    print("\n" + "=" * 70)
    print("FOLD TIMELINE  (T=train  V=val  .=unused)")
    print(f"Timeline: 0:00 {'':>20s} 12:00 {'':>12s} 23:45")
    print(f"         |{'':->46s}|")
    print(f"TEST:    [{bar(TEST_SLOT_START, TEST_SLOT_END, 'V')}]  "
          f"{_slot_to_time(TEST_SLOT_START)}-{_slot_to_time(TEST_SLOT_END)}")
    print()

    for fold in folds:
        if fold.fold_type == "daytime":
            # Spatial fold: trains on ALL slots of train-geohashes, validates on
            # daytime slots of held-out geohashes. Time-wise train spans 0-95.
            v_lo, v_hi = fold.val_slots
            arr = list(bar(0, 95, "T"))
            for i in range(BAR):
                slot = int(i / BAR * SLOTS)
                if v_lo <= slot <= v_hi:
                    arr[i] = "V"
            timeline = "".join(arr)
            n_tr_geo = fold.train["geohash"].nunique()
            n_val_geo = fold.val["geohash"].nunique()
            label = (f"{fold.name}  SPATIAL holdout"
                     f"  train={n_tr_geo} geos (all slots)"
                     f"  val={n_val_geo} held-out geos @ {_slot_to_time(v_lo)}-{_slot_to_time(v_hi)}"
                     f"  ({len(fold.train):5d}/{len(fold.val):5d} rows)")
        else:
            # across-day: train=day48-all, val=day49-morning
            v_lo, v_hi = fold.val_slots
            timeline = bar(0, 95, "T")
            label = (f"{fold.name}  train=day48-all"
                     f"  val=day49-{_slot_to_time(v_lo)}-{_slot_to_time(v_hi)}"
                     f"  (train={len(fold.train):5d} rows  val={len(fold.val):5d} rows)")

        print(f"  [{timeline}]")
        print(f"   {label}")
        print()

    print("REGIME ALIGNMENT CHECK")
    print("  Real test:   daytime (02:15-13:45)  <- what the leaderboard scores")
    print("  Daytime folds: val window = 02:15-13:45  [ALIGNED - use as primary]")
    print("  Across-day:  nighttime (00:00-02:00) [MISALIGNED - sanity check only]")
    print("=" * 70 + "\n")


# --------------------------------------------------------------------------- #
# Entry point — run this to verify your folds before fitting any model
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = load_config()
    train, test = load_raw(cfg)

    folds = get_all_folds(train)

    # Always visualize first
    visualize_folds(folds)

    # Print fold stats
    print("FOLD STATISTICS")
    print(f"{'Fold':<25s}  {'Type':<12s}  {'Train rows':>10s}  {'Val rows':>8s}")
    print("-" * 62)
    for f in folds:
        print(f"  {f.name:<23s}  {f.fold_type:<12s}  {len(f.train):>10,d}  {len(f.val):>8,d}")

    print()
    print("WHAT THIS MEANS FOR YOUR EXPERIMENT LOOP")
    print("  1. Every model/feature you test produces a predict_fn(val_df).")
    print("  2. Pass it to score_folds(folds, predict_fn) to get your local score.")
    print("  3. PRIMARY = daytime_mean (what you compare experiments on).")
    print("  4. SECONDARY = across_day_score (sanity check, do not optimize against).")
    print("  5. Submit to LB only when daytime_mean shows a clear improvement.")
    print()
    print("  Current bar to beat:  83.63  (reference nofactor, LIVE daytime score)")
