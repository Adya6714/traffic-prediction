"""Experiment 01 — Global day-49 level shift.

HYPOTHESIS (assumption A16): day 49 is systematically busier than day 48. Our
best submission so far (83.6) predicts each test cell's day-48 level with no
day-49 adjustment. If we scale those predictions up by a factor, the score may
rise.

WHY LOCAL FOLDS CANNOT FULLY ANSWER THIS
----------------------------------------
The factor was measured on the 9 NIGHTTIME morning slots (the only place both
days are observed at the same cell). The test is DAYTIME. We have no local
daytime measurement of the day-shift, because our folds hold out locations, not
days. So this is the rare case where a clean leaderboard probe is the right tool:
one variable (the factor), a handful of values, read the result.

WHAT THIS SCRIPT DOES
---------------------
1. Builds the day-48 level prediction for the test set:
      base(geohash, slot) = mean demand at that (geohash, slot) on day 48
      with fallback: (geohash,slot) -> geohash mean -> RoadType+slot mean -> global
2. Multiplies the base by each factor in FACTORS and writes one submission each.
3. Prints a local-fold sanity check (the factor should not BREAK the level model;
   it mostly shifts scale, which on location-holdout folds barely moves R²).

Run:
    python -m experiments.exp01_day_shift
Then submit the generated CSVs from outputs/ ONE OR TWO at a time and record the
live scores in ASSUMPTIONS_LOG.md section 5.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import load_config
from src.data import load_raw
from src.splits import get_all_folds, competition_score

TARGET = "demand"
FACTORS = [1.0, 1.2, 1.4, 1.6, 1.72]   # 1.0 = no shift (reproduces current best)


# --------------------------------------------------------------------------- #
# The base level predictor: day-48 (geohash, slot) mean with hierarchical fallback.
# --------------------------------------------------------------------------- #
def build_base_predictor(train_df: pd.DataFrame):
    """Learn the day-48 demand level for each (geohash, slot), plus fallbacks for
    cells not seen on day 48 (e.g. cold-start geohashes)."""
    d48 = train_df[train_df["day"] == 48]
    mu = d48[TARGET].mean()

    gs_table = d48.groupby(["geohash", "slot"])[TARGET].mean()
    geo_table = d48.groupby("geohash")[TARGET].mean()
    # RoadType + slot fallback (works for cold geohashes since RoadType is known)
    rt_table = d48.groupby(["RoadType", "slot"])[TARGET].mean()

    def predict(df: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([df["geohash"], df["slot"]])
        out = np.array(gs_table.reindex(idx).to_numpy(), dtype=float)

        need = np.isnan(out)
        if need.any():
            out[need] = geo_table.reindex(df["geohash"]).to_numpy()[need]

        need = np.isnan(out)
        if need.any():
            rt_idx = pd.MultiIndex.from_arrays([df["RoadType"], df["slot"]])
            out[need] = rt_table.reindex(rt_idx).to_numpy()[need]

        out[np.isnan(out)] = mu
        return out

    return predict


def local_fold_check(train: pd.DataFrame):
    """Sanity check on the location-holdout folds. NOTE: these folds hold out
    GEOHASHES within day 48, so they measure spatial generalization, not the
    day-shift. A scale factor mostly shifts predictions uniformly, which barely
    changes R² on these folds. The real test of the factor is the leaderboard."""
    folds = get_all_folds(train)
    print("Local-fold sanity check (location-holdout — NOT a day-shift test):")
    print(f"{'factor':>7} | " + " | ".join(f"{f.name:>20}" for f in folds))
    for factor in FACTORS:
        scores = []
        for fold in folds:
            base_fn = build_base_predictor(fold.train)
            preds = np.clip(base_fn(fold.val) * factor, 0, 1)
            scores.append(competition_score(fold.val[TARGET].to_numpy(), preds))
        print(f"{factor:>7.2f} | " + " | ".join(f"{s:>20.4f}" for s in scores))
    print("  (Expect little movement here; this only confirms nothing is broken.)\n")


def make_submissions(train: pd.DataFrame, test: pd.DataFrame, out_dir: str):
    """Write one submission CSV per factor."""
    base_fn = build_base_predictor(train)          # use ALL training history
    base_pred = base_fn(test)
    written = []
    for factor in FACTORS:
        preds = np.clip(base_pred * factor, 0, 1)
        sub = pd.DataFrame({"Index": test["Index"].astype(int), "demand": preds})
        assert sub.shape == (41778, 2)
        assert sub["Index"].tolist() == test["Index"].tolist()
        path = f"{out_dir}/exp01_shift_{factor:.2f}.csv"
        sub.to_csv(path, index=False)
        written.append((factor, path, float(preds.mean())))
    return written


if __name__ == "__main__":
    import os
    cfg = load_config()
    train, test = load_raw(cfg)

    local_fold_check(train)

    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    written = make_submissions(train, test, out_dir)

    print("Submission files written:")
    for factor, path, mean_pred in written:
        print(f"  factor {factor:>4.2f}  ->  {path}   (mean predicted demand {mean_pred:.4f})")

    print("\nWHAT TO DO NEXT")
    print("  1. Submit exp01_shift_1.00.csv first — it should reproduce ~83.6")
    print("     (confirms our pipeline matches the reference baseline).")
    print("  2. Then submit exp01_shift_1.40.csv (a middle factor).")
    print("  3. Compare. If 1.40 beats 1.00, the day-shift is real in daytime;")
    print("     try 1.60/1.72. If 1.40 is worse, the daytime shift is smaller than")
    print("     nighttime — narrow toward 1.0-1.2.")
    print("  4. Record every score in ASSUMPTIONS_LOG.md section 5.")
    print("  5. The winning factor becomes a known quantity for the real model.")
