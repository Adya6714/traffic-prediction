"""Diagnostic 02 — Why does lookup_only score 50.8 on across_night when the
geohash_mean baseline scored 65.6?

This script CHANGES NOTHING. It only prints facts so we understand the gap before
deciding anything. We rebuild the across_night fold and compare, row by row:
  - what geohash_mean predicts (the baseline that got 65.6)
  - what our lookup predicts (the thing that got 50.8)
and we find where they diverge.

Run:
    python -m experiments.diag02_lookup_gap
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
from src.splits import make_across_day_fold, competition_score
from src.predict import build_lookup
from src.baselines import build_geohash_slot_mean, build_geohash_mean

TARGET = "demand"


def main():
    cfg = load_config()
    train, _ = load_raw(cfg)

    fold = make_across_day_fold(train)
    ref = fold.train          # all of day 48
    val = fold.val.copy()     # day 49 morning (slots 0-8)
    actual = val[TARGET].to_numpy()

    print(f"across_night fold: train={len(ref)} rows (day48), val={len(val)} rows (day49 morning)")
    print(f"val slots present: {sorted(val['slot'].unique())}")
    print()

    # --- the two predictors ---
    geo_mean_fn = build_geohash_mean(ref)            # baseline that scored 65.6
    geo_slot_fn = build_geohash_slot_mean(ref)       # baseline that scored 52.3 earlier
    lookup_fn, seen = build_lookup(ref)              # our lookup that scored 50.8

    p_geomean = geo_mean_fn(val)
    p_geoslot = geo_slot_fn(val)
    p_lookup  = lookup_fn(val)

    print("Scores on this fold (recomputed here):")
    print(f"  geohash_mean       : {competition_score(actual, p_geomean):.4f}")
    print(f"  geohash_slot_mean  : {competition_score(actual, p_geoslot):.4f}")
    print(f"  our lookup         : {competition_score(actual, p_lookup):.4f}")
    print()

    # --- how often does the lookup find an EXACT (geohash, slot) match? ---
    d48 = ref.loc[ref["day"] == 48].copy()
    pairs_df = d48.loc[:, ["geohash", "slot"]].drop_duplicates()
    gs_keys = set(map(tuple, pairs_df.to_numpy()))
    val_keys = list(map(tuple, val[["geohash", "slot"]].to_numpy()))
    exact = np.array([k in gs_keys for k in val_keys])
    print(f"lookup exact (geohash,slot) match rate on val: {exact.mean()*100:.1f}%")
    print(f"  -> {(~exact).sum()} of {len(val)} rows fall through to a fallback")
    print()

    # --- KEY QUESTION: does day-48 (geohash,slot) predict day-49 (geohash,slot)? ---
    # Compare exact-match rows only.
    print("On rows WITH an exact day-48 (geohash,slot) value:")
    if exact.any():
        print(f"  lookup       R2-score: {competition_score(actual[exact], p_lookup[exact]):.4f}")
        print(f"  geohash_mean R2-score: {competition_score(actual[exact], p_geomean[exact]):.4f}")
    print("\nOn rows WITHOUT an exact match (fallback rows):")
    if (~exact).any():
        print(f"  lookup       R2-score: {competition_score(actual[~exact], p_lookup[~exact]):.4f}")
        print(f"  geohash_mean R2-score: {competition_score(actual[~exact], p_geomean[~exact]):.4f}")
    print()

    # --- the smoking gun: correlation of each prediction with the truth ---
    print("Correlation with actual demand (higher = better aligned):")
    for nm, p in [("geohash_mean", p_geomean), ("geohash_slot", p_geoslot), ("lookup", p_lookup)]:
        print(f"  {nm:>14}: corr={np.corrcoef(p, actual)[0,1]:.4f}  "
              f"mean_pred={p.mean():.4f}  mean_actual={actual.mean():.4f}")
    print()

    # --- is it a SCALE problem? day49 morning vs day48 same slots ---
    # If day49 morning demand is systematically higher than day48 at those slots,
    # then day48-based lookup will be biased LOW -> hurts R2.
    print("Scale check (day49 morning vs day48 at the SAME early slots):")
    early = sorted(val["slot"].unique().tolist())
    d48_early_mean = float(d48.loc[d48["slot"].isin(early), TARGET].mean())
    d49_mean = actual.mean()
    print(f"  day48 mean demand at slots {early}: {d48_early_mean:.4f}")
    print(f"  day49 mean demand at those slots   : {d49_mean:.4f}")
    print(f"  ratio day49/day48                  : {d49_mean/d48_early_mean:.3f}")
    print("\n  (If this ratio is well above 1.0, the lookup is biased low on this")
    print("   fold — but remember exp01 showed the daytime ratio is ~1.0, so this")
    print("   nighttime bias would NOT carry to the real daytime test.)")


if __name__ == "__main__":
    main()
