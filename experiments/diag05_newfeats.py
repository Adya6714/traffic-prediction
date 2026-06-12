"""Diagnostic 05 — Ground-up feature rethink. Test NEW signal sources before building.

We have plateaued at ~87.5 by blending day-48 cell lookups. This script tests
whether genuinely different signal sources carry information we have ignored.
Validated the honest way: predict day-49 MORNING (the labels we have) from day-48,
and see which new feature most improves over the plain day-48 cell value.

Ideas tested:
  T1 Spatial neighbours: avg day-48 demand of the K geographically closest cells
     at the same slot. (adjacent roads have correlated traffic)
  T2 Temporal window: day-48 demand at slots t-2..t+2 for the same cell.
     (the local curve shape, not just the point)
  T4 Recent day-49 trajectory: the cell's own most-recent observed day-49 value
     (slots 0-8) — a live signal for predicting the next slots.

Changes nothing; prints R2-score lift of each idea over the baseline.

Run:
    python -m experiments.diag05_newfeats
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb

from src.config import load_config
from src.data import load_raw
from src.geohash_decode import add_latlon
from src.splits import competition_score

TARGET = "demand"


def main():
    cfg = load_config()
    train, _ = load_raw(cfg)
    train = add_latlon(train)
    d48 = train[train["day"] == 48].copy()
    d49 = train[train["day"] == 49].copy()      # morning labels we can score against

    # ---- baseline: day-48 (geohash, slot) value ----
    base48 = d48.groupby(["geohash", "slot"])[TARGET].mean()
    idx = pd.MultiIndex.from_arrays([d49["geohash"], d49["slot"]])
    d49["base"] = base48.reindex(idx).to_numpy()
    dd = d49.dropna(subset=["base"]).copy()
    y = dd[TARGET].to_numpy()
    print(f"rows with a day-48 base: {len(dd)}")
    print(f"BASELINE (day-48 cell value) R2-score: {competition_score(y, dd['base'].to_numpy()):.4f}\n")

    # ---- T2: temporal window (day-48 neighbouring slots) ----
    print("T2 — temporal window (day-48 slots t-1,t+1,t-2,t+2):")
    piv = d48.pivot_table(index="geohash", columns="slot", values=TARGET, aggfunc="mean")
    def win(gh, s, off):
        try:
            return piv.loc[gh, s + off]
        except KeyError:
            return np.nan
    for off in [-1, 1, -2, 2]:
        col = f"base_off{off}"
        dd[col] = [win(g, s, off) for g, s in zip(dd["geohash"], dd["slot"])]
    win_cols = [f"base_off{o}" for o in [-1, 1, -2, 2]]
    print(f"  coverage of t-1 neighbour: {dd['base_off-1'].notna().mean()*100:.0f}%")

    # ---- T1: spatial neighbours (avg day-48 demand of K nearest cells, same slot) ----
    print("\nT1 — spatial neighbours (K=5 nearest cells, same slot):")
    cells = d48[["geohash", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    coords = cells[["lat", "lon"]].to_numpy()
    # brute-force nearest neighbours (only ~1200 cells -> fine)
    from numpy.linalg import norm
    K = 5
    nbr = {}
    for i, g in enumerate(cells["geohash"]):
        d = norm(coords - coords[i], axis=1)
        order = np.argsort(d)[1:K+1]            # skip self
        nbr[g] = cells["geohash"].to_numpy()[order]
    # neighbour mean demand at (slot)
    gs = d48.groupby(["geohash", "slot"])[TARGET].mean()
    def neigh_mean(g, s):
        vals = [gs.get((ng, s), np.nan) for ng in nbr.get(g, [])]
        vals = [v for v in vals if not np.isnan(v)]
        return np.mean(vals) if vals else np.nan
    dd["neigh"] = [neigh_mean(g, s) for g, s in zip(dd["geohash"], dd["slot"])]
    print(f"  coverage: {dd['neigh'].notna().mean()*100:.0f}%")
    print(f"  corr(neighbour mean, actual day49): {np.corrcoef(dd['neigh'].fillna(dd['neigh'].mean()), y)[0,1]:.3f}")

    # ---- model comparisons via CV ----
    def cv_score(cols):
        X = dd[cols].fillna(-1).to_numpy()
        oof = np.zeros(len(dd))
        for tr, va in KFold(5, shuffle=True, random_state=42).split(X):
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                  min_child_samples=20, verbose=-1)
            m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va])
        return competition_score(y, oof)

    print("\n--- CV R2-score with each feature set (target = day49 morning) ---")
    print(f"  base only:                  {cv_score(['base']):.4f}")
    print(f"  base + temporal window:     {cv_score(['base'] + win_cols):.4f}")
    print(f"  base + spatial neighbour:   {cv_score(['base', 'neigh']):.4f}")
    print(f"  base + window + neighbour:  {cv_score(['base'] + win_cols + ['neigh']):.4f}")
    print("\n  (a meaningful lift over 'base only' = that signal is real and worth building)")


if __name__ == "__main__":
    main()
