"""Diagnostic 04 — What does the Index ordering encode?

diag03 found lag-1 autocorrelation of demand (by Index order) = 0.32. That is NOT
random. This script investigates WHY, and whether it is exploitable to recover
day-49 daytime demand. Changes nothing; prints facts.

Checks:
  A. Is train sorted by something? (day, geohash, slot in Index order)
  B. Does the autocorrelation come from rows sharing a geohash being adjacent?
  C. How is the TEST set ordered, and does it interleave with train Index?
  D. Neighbour-recovery: can a row's demand be predicted by its Index-neighbours'
     demand? (the real test of an ordering leak)

Run:
    python -m experiments.diag04_index
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import load_config
from src.data import load_raw

TARGET = "demand"


def main():
    cfg = load_config()
    train, test = load_raw(cfg)

    print("=" * 64)
    print("A. IS TRAIN SORTED BY SOMETHING? (look at first 20 rows in Index order)")
    print("=" * 64)
    t = train.sort_values("Index")
    print(t[["Index", "day", "timestamp", "slot", "geohash", "RoadType", "demand"]].head(20).to_string(index=False))
    # how often does a key stay constant from one row to the next?
    for col in ["day", "geohash", "slot", "RoadType"]:
        same = (t[col].values[1:] == t[col].values[:-1]).mean()
        print(f"  fraction of adjacent rows with SAME {col}: {same:.3f}")

    print("\n" + "=" * 64)
    print("B. WHERE DOES THE AUTOCORRELATION COME FROM?")
    print("=" * 64)
    s = t["demand"].to_numpy()
    print(f"raw lag-1 autocorr (Index order): {np.corrcoef(s[:-1], s[1:])[0,1]:.4f}")
    # if we remove the per-geohash mean, does autocorrelation vanish? (then it's
    # just 'same geohash sits together'); if it stays, ordering carries extra info
    t = t.copy()
    t["resid"] = t["demand"] - t.groupby("geohash")["demand"].transform("mean")
    r = t["resid"].to_numpy()
    print(f"lag-1 autocorr after removing per-geohash mean: {np.corrcoef(r[:-1], r[1:])[0,1]:.4f}")
    t["resid2"] = t["demand"] - t.groupby(["geohash", "slot"])["demand"].transform("mean")
    r2 = t["resid2"].to_numpy()
    print(f"lag-1 autocorr after removing per-(geohash,slot) mean: {np.corrcoef(r2[:-1], r2[1:])[0,1]:.4f}")
    print("  (if it drops to ~0, the ordering is just grouping by geohash/cell —")
    print("   NOT an extra leak. if it stays high, ordering carries hidden signal.)")

    print("\n" + "=" * 64)
    print("C. TEST ORDERING + RELATION TO TRAIN")
    print("=" * 64)
    te = test.sort_values("Index")
    print(te[["Index", "day", "timestamp", "slot", "geohash", "RoadType"]].head(10).to_string(index=False))
    for col in ["geohash", "slot"]:
        same = (te[col].values[1:] == te[col].values[:-1]).mean()
        print(f"  test: fraction of adjacent rows with SAME {col}: {same:.3f}")
    # do train and test share (geohash) at the same Index position?
    merged = train[["Index", "geohash"]].merge(
        test[["Index", "geohash"]], on="Index", suffixes=("_train", "_test"))
    same_geo = (merged["geohash_train"] == merged["geohash_test"]).mean()
    print(f"  same Index in train and test -> same geohash? {same_geo:.3f}")
    print("  (if ~1.0, train row i and test row i are the SAME location -> the test")
    print("   demand may be recoverable from train row i's demand!)")

    print("\n" + "=" * 64)
    print("D. NEIGHBOUR RECOVERY: does Index position alone predict demand?")
    print("=" * 64)
    # predict each train row's demand as the average of its Index-neighbours
    s = train.sort_values("Index")["demand"].to_numpy()
    neigh = np.full(len(s), np.nan)
    neigh[1:-1] = (s[:-2] + s[2:]) / 2.0     # mean of the two Index-neighbours
    valid = ~np.isnan(neigh)
    from src.splits import competition_score
    print(f"predict demand = mean(Index-neighbours): R2-score = {competition_score(s[valid], neigh[valid]):.4f}")
    print("  (high => the answer leaks through ordering; low => no usable leak)")


if __name__ == "__main__":
    main()
