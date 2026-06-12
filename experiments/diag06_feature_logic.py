"""Diagnostic 06 — test the logical feature assumptions BEFORE building them.

We have plateaued at 87.90 via blend tuning. This tests four physically-motivated
feature ideas to see which carry real signal, validated on day-49 morning (the only
labels we have). We measure each as a lift over the plain day-48 cell value, the same
way diag03/diag05 did. NOTE: nighttime fold — use as a FILTER (which ideas are alive),
not as a daytime predictor. Anything alive here gets confirmed on the leaderboard.

Assumptions tested:
  A1 Multiplicative region day-factor: scale day-48 by day49/day48 morning RATIO per
     geohash prefix. (traffic scales multiplicatively, not additively)
  A2 Normalized profile stability: predict demand/cell_mean (the shape), check if the
     normalized profile is more stable across days than the raw level.
  A3 Spatial-NN fallback: for cells, how good is the nearest-neighbour estimate vs the
     coarse roadtype fallback? (better cold-cell handling)
  A4 Residual structure: does Weather/Temperature predict the RESIDUAL of a day-48
     cell prediction? (features dead on level may live on residual)

Run:
    python -m experiments.diag06_feature_logic
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.linalg import norm
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
    d49 = train[train["day"] == 49].copy()
    d48["p4"] = d48["geohash"].str[:4]
    d49["p4"] = d49["geohash"].str[:4]

    base48 = d48.groupby(["geohash", "slot"])[TARGET].mean()
    idx = pd.MultiIndex.from_arrays([d49["geohash"], d49["slot"]])
    d49["base"] = base48.reindex(idx).to_numpy()
    dd = d49.dropna(subset=["base"]).copy()
    y = dd[TARGET].to_numpy()
    print(f"rows with day-48 base: {len(dd)}")
    print(f"BASELINE day-48 cell value: {competition_score(y, dd['base'].to_numpy()):.4f}\n")

    # ---- A1: multiplicative region day-factor ----
    # estimate per-prefix ratio of day49-morning to day48 at the SAME morning slots,
    # then scale the base. Use ONLY slots present in d49 (0-8) to estimate the factor.
    morn_slots = sorted(d49["slot"].unique())
    d48_morn = d48[d48["slot"].isin(morn_slots)]
    f49 = d49.groupby("p4")[TARGET].mean()
    f48 = d48_morn.groupby("p4")[TARGET].mean()
    ratio = (f49 / f48).clip(0.5, 2.0)          # guard against extremes
    dd["day_factor"] = dd["p4"].map(ratio).fillna(1.0)
    a1 = np.clip(dd["base"].to_numpy() * dd["day_factor"].to_numpy(), 0, 1)
    print(f"A1 multiplicative day-factor (base * prefix ratio): {competition_score(y, a1):.4f}")
    print(f"   prefix ratio range: {ratio.min():.2f}..{ratio.max():.2f}, median {ratio.median():.2f}")

    # ---- A2: normalized profile stability ----
    # cell_mean from day48; profile = base / cell_mean. If profile is stable, then
    # predicting profile * (day49 cell level) should help. We test the ceiling:
    # how well does day48 profile * day49-cell-mean (estimated from morning) do?
    cell_mean48 = d48.groupby("geohash")[TARGET].mean()
    dd["cm48"] = dd["geohash"].map(cell_mean48)
    dd["profile"] = dd["base"] / dd["cm48"].replace(0, np.nan)
    # day49 cell level estimate from morning slots
    cell_mean49 = d49.groupby("geohash")[TARGET].mean()
    dd["cm49"] = dd["geohash"].map(cell_mean49)
    a2 = np.clip((dd["profile"] * dd["cm49"]).to_numpy(), 0, 1)
    valid = ~np.isnan(a2)
    print(f"\nA2 normalized profile * day49 cell-level: {competition_score(y[valid], a2[valid]):.4f}")
    print(f"   (compares 'shape stable, level shifts' hypothesis vs raw base)")

    # ---- A3: spatial-NN estimate vs base ----
    cells = d48[["geohash", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
    coords = cells[["lat", "lon"]].to_numpy(); ghs = cells["geohash"].to_numpy()
    nbr = {}
    for i, g in enumerate(ghs):
        d = norm(coords - coords[i], axis=1)
        nbr[g] = ghs[np.argsort(d)[1:6]]
    gs = d48.groupby(["geohash", "slot"])[TARGET].mean()
    def nn(g, s):
        v = [gs.get((ng, s), np.nan) for ng in nbr.get(g, [])]
        v = [x for x in v if not np.isnan(x)]
        return np.mean(v) if v else np.nan
    dd["nn"] = [nn(g, s) for g, s in zip(dd["geohash"], dd["slot"])]
    a3 = np.clip(dd["nn"].fillna(dd["base"]).to_numpy(), 0, 1)
    print(f"\nA3 spatial-NN estimate alone: {competition_score(y, a3):.4f}")
    # blended 50/50 with base
    a3b = np.clip(0.5*dd["base"].to_numpy() + 0.5*dd["nn"].fillna(dd["base"]).to_numpy(), 0, 1)
    print(f"   base/NN 50-50 blend: {competition_score(y, a3b):.4f}")

    # ---- A4: residual structure ----
    resid = y - dd["base"].to_numpy()
    dd["resid"] = resid
    dd["weather_code"] = dd["Weather"].astype("category").cat.codes
    print(f"\nA4 residual structure (does a 'dead' feature predict the error?):")
    print(f"   corr(residual, Temperature): {np.corrcoef(dd['resid'], dd['Temperature'].fillna(dd['Temperature'].mean()))[0,1]:.4f}")
    print(f"   corr(residual, weather_code): {np.corrcoef(dd['resid'], dd['weather_code'])[0,1]:.4f}")
    print(f"   residual std by RoadType:")
    print(dd.groupby('RoadType')['resid'].std().round(4).to_string())

    print("\nINTERPRETATION: any A1/A2/A3 that clears the baseline meaningfully is a")
    print("real lever -> build it and confirm on the leaderboard. A4 corr away from 0")
    print("means a 'dead' feature has life in the residual -> worth a residual model.")


if __name__ == "__main__":
    main()
