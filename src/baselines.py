"""Stage 3 — Baselines.

The most under-appreciated habit in applied ML: build the dumbest reasonable
predictor FIRST, score it on your splitter, and make every later model beat it.

Why baselines come before EDA and before any real model
--------------------------------------------------------
You need a reference line. If your fancy GBDT scores 78 on the daytime folds but
a one-line "predict yesterday's value" baseline scores 80, the GBDT is actively
HURTING you — and you would never know without the baseline number in hand.
Baselines calibrate the whole project: they tell you how much of the score comes
from trivial pattern-repetition vs. how much is left for real modeling to win.

Each baseline here is a `predict_fn`: a function that takes a validation
DataFrame and returns a numpy array of predicted demand, one per row. That is
the exact interface src.splits.score_folds expects, so baselines plug straight
into the experiment loop.

The baselines, from dumbest to least-dumb
-----------------------------------------
1. global_mean      — predict the overall train mean for every row. The floor.
2. slot_mean        — predict the mean demand for that time-of-day slot.
3. geohash_mean     — predict the mean demand for that location.
4. geohash_slot_mean— predict mean for that (location, slot) pair. The strongest
                      "level" baseline: where + when, no momentum.
5. seasonal_naive   — predict day-48's value at the same (geohash, slot).
                      The classic forecasting baseline.

CRITICAL leakage rule
---------------------
Every statistic (a mean, a lookup table) is computed ONLY from the fold's
training rows, then applied to the validation rows. A baseline that computes its
means on the full dataset (including val) would leak and report a fake-high score.
Each builder below takes `train_df` and returns a closure that only saw train_df.

Run directly to score all baselines on all folds:
    python -m src.baselines
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import load_raw
from src.splits import get_all_folds, score_folds


# --------------------------------------------------------------------------- #
# Baseline builders.
# Each takes the fold's TRAINING data and returns a predict_fn(val_df)->np.array.
# The returned function is a "closure": it remembers the stats it learned from
# train_df and applies them to whatever val_df it's later given.
# --------------------------------------------------------------------------- #
TARGET = "demand"


def build_global_mean(train_df: pd.DataFrame):
    """Predict the single overall mean for every row. The absolute floor.
    A model that can't beat this has learned nothing."""
    mu = train_df[TARGET].mean()

    def predict(val_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(val_df), mu)

    return predict


def build_slot_mean(train_df: pd.DataFrame):
    """Predict mean demand for the row's time-of-day slot.
    Captures the diurnal rhythm (assumption A1) and nothing else."""
    mu = train_df[TARGET].mean()
    table = train_df.groupby("slot")[TARGET].mean()

    def predict(val_df: pd.DataFrame) -> np.ndarray:
        # .map looks up each val slot; unseen slots fall back to global mean.
        return val_df["slot"].map(table).fillna(mu).to_numpy()

    return predict


def build_geohash_mean(train_df: pd.DataFrame):
    """Predict mean demand for the row's location.
    Captures the spatial level (assumption A2) and nothing else."""
    mu = train_df[TARGET].mean()
    table = train_df.groupby("geohash")[TARGET].mean()

    def predict(val_df: pd.DataFrame) -> np.ndarray:
        return val_df["geohash"].map(table).fillna(mu).to_numpy()

    return predict


def build_geohash_slot_mean(train_df: pd.DataFrame):
    """Predict mean demand for the (location, slot) pair.
    This is the 'level' model: where + when, no day-over-day momentum.
    Expected to be the strongest baseline given the leaderboard finding that
    momentum hurts and level helps.

    Falls back hierarchically when a (geohash, slot) pair is unseen in train:
        (geohash, slot)  ->  geohash mean  ->  slot mean  ->  global mean
    This fallback chain is what lets it handle cold-start cells.
    """
    mu = train_df[TARGET].mean()
    gs_table   = train_df.groupby(["geohash", "slot"])[TARGET].mean()
    geo_table  = train_df.groupby("geohash")[TARGET].mean()
    slot_table = train_df.groupby("slot")[TARGET].mean()

    def predict(val_df: pd.DataFrame) -> np.ndarray:
        # Primary lookup: (geohash, slot)
        idx = pd.MultiIndex.from_arrays([val_df["geohash"], val_df["slot"]])
        out = np.array(gs_table.reindex(idx).to_numpy(), dtype=float)  # writable copy
        # Fallback 1: geohash mean
        need = np.isnan(out)
        if need.any():
            out[need] = val_df["geohash"].map(geo_table).to_numpy()[need]
        # Fallback 2: slot mean
        need = np.isnan(out)
        if need.any():
            out[need] = val_df["slot"].map(slot_table).to_numpy()[need]
        # Fallback 3: global mean
        out[np.isnan(out)] = mu
        return out

    return predict


def build_seasonal_naive(train_df: pd.DataFrame):
    """Classic seasonal-naive: predict the value at the same (geohash, slot) on
    day 48. Functionally similar to geohash_slot_mean here because day 48 has one
    observation per (geohash, slot), but conceptually it's the 'last cycle'
    forecast — the textbook forecasting baseline.

    Built only from day-48 rows present in train_df.
    """
    mu = train_df[TARGET].mean()
    d48 = train_df[train_df["day"] == 48]
    table = d48.groupby(["geohash", "slot"])[TARGET].mean()
    slot_table = train_df.groupby("slot")[TARGET].mean()

    def predict(val_df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([val_df["geohash"], val_df["slot"]])
        out = np.array(table.reindex(idx).to_numpy(), dtype=float)  # writable copy
        need = np.isnan(out)
        if need.any():
            out[need] = val_df["slot"].map(slot_table).to_numpy()[need]
        out[np.isnan(out)] = mu
        return out

    return predict


# Registry so the runner can iterate over all baselines by name.
BASELINES = {
    "global_mean":       build_global_mean,
    "slot_mean":         build_slot_mean,
    "geohash_mean":      build_geohash_mean,
    "geohash_slot_mean": build_geohash_slot_mean,
    "seasonal_naive":    build_seasonal_naive,
}


# --------------------------------------------------------------------------- #
# Runner — score every baseline on every fold.
# --------------------------------------------------------------------------- #
def run_all_baselines(train: pd.DataFrame) -> pd.DataFrame:
    """For each baseline, build it on each fold's TRAIN and score on each fold's
    VAL. Returns a tidy results table.

    Note the structure: the baseline is REBUILT per fold using only that fold's
    training rows. This is the leakage-safe pattern — stats never see val.
    """
    folds = get_all_folds(train)
    rows = []

    for bname, builder in BASELINES.items():
        # score_folds needs a single predict_fn, but each fold needs its own
        # baseline built on its own train. So we build per-fold and score per-fold,
        # then aggregate manually to mirror score_folds' output shape.
        per_fold = {}
        daytime = []
        across = None
        for fold in folds:
            predict_fn = builder(fold.train)        # learns ONLY from fold.train
            from src.splits import competition_score
            preds = predict_fn(fold.val)
            sc = competition_score(fold.val[TARGET].to_numpy(), preds)
            per_fold[fold.name] = round(sc, 4)
            if fold.fold_type == "daytime":
                daytime.append(sc)
            else:
                across = sc
        rows.append({
            "baseline":         bname,
            "daytime_mean":     round(float(np.mean(daytime)), 4),
            "daytime_std":      round(float(np.std(daytime)), 4),
            "across_day_night": round(across, 4) if across is not None else None,
            **{f"fold_{k}": v for k, v in per_fold.items()},
        })

    return pd.DataFrame(rows).sort_values("daytime_mean", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Helper: build the best baseline on ALL training data and make a submission.
# Use this to get your first honest baseline submission file.
# --------------------------------------------------------------------------- #
def make_baseline_submission(train: pd.DataFrame, test: pd.DataFrame,
                             baseline_name: str, out_path: str) -> pd.DataFrame:
    """Build a baseline on ALL of train (day48 + day49-morning) and predict test.
    Writes a submission CSV in the required Index,demand format."""
    builder = BASELINES[baseline_name]
    predict_fn = builder(train)                 # uses ALL available history
    preds = np.clip(predict_fn(test), 0, 1)     # clip to valid demand range (A8)
    sub = pd.DataFrame({"Index": test["Index"].astype(int), "demand": preds})
    assert sub.shape == (41778, 2), f"bad shape {sub.shape}"
    assert sub["Index"].tolist() == test["Index"].tolist(), "Index order mismatch"
    sub.to_csv(out_path, index=False)
    return sub


if __name__ == "__main__":
    cfg = load_config()
    train, test = load_raw(cfg)

    print("Scoring all baselines on all folds...\n")
    table = run_all_baselines(train)

    # Pretty-print the headline columns
    show = ["baseline", "daytime_mean", "daytime_std", "across_day_night"]
    print(table[show].to_string(index=False))

    print("\n" + "=" * 64)
    print("HOW TO READ THIS")
    print("=" * 64)
    print("daytime_mean      = PRIMARY metric. Mean competition score across the")
    print("                    three daytime folds (02:15-13:45 regime).")
    print("daytime_std       = stability across folds. High std = fragile baseline.")
    print("across_day_night  = SECONDARY. Nighttime score. Note how it can DISAGREE")
    print("                    with daytime_mean — that disagreement is the whole")
    print("                    lesson of this problem.")
    print()
    best = table.iloc[0]
    print(f"Best baseline: '{best['baseline']}'  daytime_mean={best['daytime_mean']}")
    print(f"Live bar to beat (reference nofactor): 83.63")
    print()
    print("NEXT: any real model must beat the best baseline's daytime_mean.")
    print("If it can't, it is not helping. This is your calibration line.")
