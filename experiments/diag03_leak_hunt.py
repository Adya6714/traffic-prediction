"""Diagnostic 03 — Leak hunt. Can demand be RECOVERED (not just modelled)?

A score of 100 = perfect R². Our determinism test showed demand is NOT a pure
function of the road columns, and raw_cell (day-48 copy) scored only 79.6, so
day-49 daytime is not a copy of day-48. The remaining route to 100: day-49's OWN
features (Temperature, Weather — present in the test set) drive the day-over-day
change via some recoverable relationship.

This script CHANGES NOTHING. It prints facts. Checks, fastest/highest-value first:

  A. Index structure  - is the target recoverable from row ordering / Index?
  B. demand value structure - is demand quantized / generated from a formula?
  C. Cross-day copy   - how close is day49 to day48 for the same cell?
  D. KILLER CHECK     - can [day48 base + day49 features] predict day49 MORNING
                        near-perfectly? If R2 jumps toward 1.0, we found the
                        structure and can apply it to daytime -> ~100.
  E. Delta link       - does the day-over-day CHANGE track Temperature/Weather change?

Run:
    python -m experiments.diag03_leak_hunt
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb

from src.config import load_config
from src.data import load_raw
from src.splits import competition_score

TARGET = "demand"


def main():
    cfg = load_config()
    train, test = load_raw(cfg)
    d48 = train[train["day"] == 48].copy()
    d49 = train[train["day"] == 49].copy()   # morning only, but HAS day-49 features+labels

    print("=" * 64)
    print("A. INDEX STRUCTURE")
    print("=" * 64)
    print(f"train Index: {train['Index'].min()}..{train['Index'].max()} (n={train['Index'].nunique()})")
    print(f"test  Index: {test['Index'].min()}..{test['Index'].max()} (n={test['Index'].nunique()})")
    overlap = set(train["Index"]) & set(test["Index"])
    print(f"train/test Index overlap: {len(overlap)}")
    print(f"corr(demand, Index) in train: {train['demand'].corr(train['Index']):.4f}")
    # is train sorted so demand has runs / autocorrelation by Index?
    s = train.sort_values("Index")["demand"].to_numpy()
    print(f"lag-1 autocorr of demand by Index order: {np.corrcoef(s[:-1], s[1:])[0,1]:.4f}")
    print("  (near 0 = no ordering leak; high = demand recoverable from neighbours)")

    print("\n" + "=" * 64)
    print("B. demand VALUE STRUCTURE")
    print("=" * 64)
    u = train["demand"].nunique()
    print(f"unique demand values: {u} of {len(train)} rows")
    print(f"min positive demand: {train['demand'][train['demand']>0].min():.2e}")
    # quantization check: is demand*N close to integer for small N?
    for N in [100, 1000, 10000]:
        frac_int = np.isclose(train["demand"]*N, np.round(train["demand"]*N), atol=1e-3).mean()
        print(f"  fraction where demand*{N} is ~integer: {frac_int:.3f}")
    print("  (high fraction at some N => demand is generated from a discrete count)")

    print("\n" + "=" * 64)
    print("C. CROSS-DAY COPY (cells observed on BOTH days, morning slots)")
    print("=" * 64)
    a = d48[["geohash", "slot", "demand"]].rename(columns={"demand": "d48"})
    b = d49[["geohash", "slot", "demand"]].rename(columns={"demand": "d49"})
    both = b.merge(a, on=["geohash", "slot"], how="inner")
    exact = np.isclose(both["d48"], both["d49"], atol=1e-6).mean()
    print(f"both-day cells: {len(both)}")
    print(f"fraction where day49 == day48 EXACTLY: {exact:.3f}")
    print(f"corr(day48, day49): {both['d48'].corr(both['d49']):.3f}")
    print("  (if exact fraction ~1.0, demand is a pure copy -> trivial 100)")

    print("\n" + "=" * 64)
    print("D. KILLER CHECK: do day-49's OWN features recover day-49 demand?")
    print("=" * 64)
    print("Target = day49 morning demand. We compare three predictors via 5-fold CV:")
    print("  (1) base only      = day48 same-cell value")
    print("  (2) base + day49 features (Temperature, Weather, RoadType, lanes, slot)")
    print("  If (2) >> (1), day-49 features explain the day-over-day change -> route to 100.\n")

    # build day48 base (geohash,slot) mean
    base48 = d48.groupby(["geohash", "slot"])[TARGET].mean()
    dd = d49.copy()
    idx = pd.MultiIndex.from_arrays([dd["geohash"], dd["slot"]])
    dd["base"] = base48.reindex(idx).to_numpy()
    dd = dd.dropna(subset=["base"]).copy()       # keep cells with a day48 base
    dd["weather_code"] = dd["Weather"].astype("category").cat.codes
    dd["road_code"] = dd["RoadType"].astype("category").cat.codes

    y = dd[TARGET].to_numpy()
    base_only = dd["base"].to_numpy()
    print(f"  (1) base-only R2-score: {competition_score(y, base_only):.4f}")

    feats = ["base", "Temperature", "weather_code", "road_code", "NumberofLanes", "slot"]
    X = dd[feats].fillna(-999).to_numpy()
    oof = np.zeros(len(dd))
    for tr, va in KFold(5, shuffle=True, random_state=42).split(X):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              min_child_samples=20, verbose=-1)
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    print(f"  (2) base + day49 features R2-score: {competition_score(y, oof):.4f}")
    print("\n  VERDICT: if (2) is MUCH higher than (1), build the real model with")
    print("  day-49 features as the day-over-day corrector. If about equal, day-49")
    print("  features do NOT explain the change and 100 is likely a different leak.")

    print("\n" + "=" * 64)
    print("E. DELTA LINK: does the day-over-day CHANGE track feature changes?")
    print("=" * 64)
    # need day49 features at the same cells
    f49 = d49.groupby(["geohash", "slot"]).agg(
        d49=("demand", "mean"), temp49=("Temperature", "mean")).reset_index()
    f48 = d48.groupby(["geohash", "slot"]).agg(
        d48=("demand", "mean"), temp48=("Temperature", "mean")).reset_index()
    md = f49.merge(f48, on=["geohash", "slot"], how="inner").dropna()
    md["d_demand"] = md["d49"] - md["d48"]
    md["d_temp"] = md["temp49"] - md["temp48"]
    print(f"corr(change in demand, change in Temperature): {md['d_demand'].corr(md['d_temp']):.4f}")
    print("  (strong corr => Temperature change drives demand change)")


if __name__ == "__main__":
    main()
