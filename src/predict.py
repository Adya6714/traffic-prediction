"""Stage 7 — Hybrid predictors (lookup + model).

Why this file exists
--------------------
Our LightGBM model (src.model) currently scores 63.2 on the across-day fold —
BELOW the dumb geohash_mean baseline (65.6). The reason (assumption A19): to
prevent leakage we compute each location's history features from OTHER locations
during training, so the model learns to distrust the precise per-cell feature.
But for the 1180 SEEN test locations, that precise feature is gold.

The fix: don't force the model to learn the per-location level from scratch.
Instead, LOOK IT UP directly from day-48 history for seen locations, and use the
model only where lookup can't help (cold locations) or as a small correction.

Three predictors to compare (all scored on the same folds)
----------------------------------------------------------
1. lookup_only  : pure day-48 (geohash, slot) level with prefix fallback. No model.
                  This is essentially the 83.6 reference approach, rebuilt cleanly.
2. hybrid_hard  : seen location -> lookup ; cold location -> model.
3. hybrid_blend : seen location -> w*lookup + (1-w)*model ; cold -> model.
                  w (BLEND_W) controls how much we trust the lookup.

What to watch: across_day_night. It predicts SEEN locations on a new day, which
is what the real test mostly is. Beat the model's 63.2 and the baseline's 65.6.

Run:
    python -m src.predict                 # score all three on the folds
    python -m src.predict --submit hybrid_blend   # write a submission for one
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.data import load_raw
from src.features import FeatureBuilder
from src.model import train_model
from src.splits import get_all_folds, competition_score

TARGET = "demand"
BLEND_W = 0.8          # weight on the lookup in hybrid_blend (0.8 = trust lookup 80%)


# --------------------------------------------------------------------------- #
# The lookup. KEY INSIGHT from diag02: the geohash ALL-DAY average correlates
# better with the truth (0.826) than the per-slot value (0.769), because it
# pools ~96 day-48 observations into a stable estimate instead of trusting one
# noisy slot. So we LEAD with the geohash average, and only add a gentle slot
# adjustment (how that location's slot deviates from its own daily average),
# damped so a single noisy slot can't dominate.
# Built ONLY from the reference set -> leakage-safe in the folds.
# --------------------------------------------------------------------------- #
def build_lookup(reference: pd.DataFrame, slot_adjust: float = 0.0):
    """Return (lookup_fn, seen_geohashes).

    base level  = geohash all-day mean (stable; fallback p5 -> p4 -> RoadType -> global)
    slot_adjust = how much to nudge by the location's slot pattern, 0..1.
                  0.0 = pure geohash average (diag02 says this is strongest on
                  across_night). >0 reintroduces some per-slot signal. We expose
                  it so we can TEST whether any slot detail helps.
    """
    d48 = reference[reference["day"] == 48].copy()
    d48["p5"] = d48["geohash"].str[:5]
    d48["p4"] = d48["geohash"].str[:4]
    mu = d48[TARGET].mean()

    geo  = d48.groupby("geohash")[TARGET].mean()            # PRIMARY: stable level
    p5   = d48.groupby("p5")[TARGET].mean()                 # fallback: neighbourhood
    p4   = d48.groupby("p4")[TARGET].mean()                 # fallback: district
    rt   = d48.groupby("RoadType")[TARGET].mean()           # fallback: road type
    seen = set(d48["geohash"].unique())

    # slot pattern: ratio of (RoadType, slot) mean to RoadType mean. Captures the
    # highway daily/evening pattern (A6/A10) WITHOUT trusting a single cell's slot.
    rt_slot = d48.groupby(["RoadType", "slot"])[TARGET].mean()
    rt_mean = d48.groupby("RoadType")[TARGET].mean()
    slot_ratio = (rt_slot / rt_mean.reindex(rt_slot.index.get_level_values("RoadType")).values)

    def lookup_fn(df: pd.DataFrame) -> np.ndarray:
        x = df.copy()
        x["p5"] = x["geohash"].str[:5]
        x["p4"] = x["geohash"].str[:4]

        def look1(table, key):
            return np.array(table.reindex(x[key]).to_numpy(), dtype=float)

        # base: geohash mean with hierarchical fallback
        out = look1(geo, "geohash")
        for tab, key in [(p5, "p5"), (p4, "p4"), (rt, "RoadType")]:
            need = np.isnan(out)
            if need.any():
                out[need] = look1(tab, key)[need]
        out[np.isnan(out)] = mu

        # optional slot adjustment (road-type slot pattern), damped by slot_adjust
        if slot_adjust > 0:
            idx = pd.MultiIndex.from_arrays([x["RoadType"], x["slot"]])
            ratio = np.array(slot_ratio.reindex(idx).to_numpy(), dtype=float)
            ratio[np.isnan(ratio)] = 1.0
            factor = 1.0 + slot_adjust * (ratio - 1.0)
            out = out * factor

        return np.clip(out, 0, 1)

    return lookup_fn, seen


# --------------------------------------------------------------------------- #
# The three predictors. Each returns a predict_fn(df) -> raw demand.
# --------------------------------------------------------------------------- #
def make_lookup_only(reference):
    lookup_fn, _ = build_lookup(reference)
    return lookup_fn

def make_hybrid_hard(reference):
    lookup_fn, seen = build_lookup(reference)
    model_fn, _, _ = train_model(reference, transform="none", use_oof=False)

    def predict_fn(df):
        look = lookup_fn(df)
        mod = model_fn(df)
        is_seen = df["geohash"].isin(seen).to_numpy()
        return np.where(is_seen, look, mod)        # seen->lookup, cold->model
    return predict_fn

def make_hybrid_blend(reference, w=BLEND_W):
    lookup_fn, seen = build_lookup(reference)
    model_fn, _, _ = train_model(reference, transform="none", use_oof=False)

    def predict_fn(df):
        look = lookup_fn(df)
        mod = model_fn(df)
        is_seen = df["geohash"].isin(seen).to_numpy()
        blended = w * look + (1 - w) * mod         # seen -> weighted mix
        return np.where(is_seen, blended, mod)     # cold -> pure model
    return predict_fn


PREDICTORS = {
    "lookup_only":  make_lookup_only,
    "hybrid_hard":  make_hybrid_hard,
    "hybrid_blend": make_hybrid_blend,
}


# --------------------------------------------------------------------------- #
# Scoring on the folds.
# --------------------------------------------------------------------------- #
def score_predictor(train, builder):
    folds = get_all_folds(train)
    per_fold, daytime, across = {}, [], None
    for fold in folds:
        predict_fn = builder(fold.train)
        preds = predict_fn(fold.val)
        sc = competition_score(fold.val[TARGET].to_numpy(), preds)
        per_fold[fold.name] = round(sc, 4)
        if fold.fold_type == "daytime":
            daytime.append(sc)
        else:
            across = sc
    return {"daytime_mean": round(float(np.mean(daytime)), 4),
            "daytime_std": round(float(np.std(daytime)), 4),
            "across_day_night": round(across, 4),
            "per_fold": per_fold}


def make_submission(train, test, name, out_path):
    predict_fn = PREDICTORS[name](train)
    preds = np.clip(predict_fn(test), 0, 1)
    sub = pd.DataFrame({"Index": test["Index"].astype(int), "demand": preds})
    assert sub.shape == (41778, 2) and sub["Index"].tolist() == test["Index"].tolist()
    sub.to_csv(out_path, index=False)
    return sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", choices=list(PREDICTORS), default=None,
                    help="write a submission CSV for this predictor")
    args = ap.parse_args()

    set_seed(42)
    cfg = load_config()
    train, test = load_raw(cfg)

    print("Scoring predictors on the folds...\n")
    print("Reference points:  geohash_mean baseline across_night=65.6 | "
          "LightGBM model across_night=63.2 | LIVE bar=83.6\n")

    # First: how much (if any) slot detail should the lookup use? diag02 says the
    # geohash all-day average (slot_adjust=0) is strongest. Confirm by sweeping.
    print("Lookup slot_adjust sweep (0.0 = pure geohash average):")
    print(f"{'slot_adjust':>12} | {'daytime_mean':>12} | {'across_night':>12}")
    print("-" * 42)
    for sa in [0.0, 0.25, 0.5, 1.0]:
        r = score_predictor(train, lambda ref, _sa=sa: build_lookup(ref, slot_adjust=_sa)[0])
        print(f"{sa:>12.2f} | {r['daytime_mean']:>12.4f} | {r['across_day_night']:>12.4f}")
    print()

    print(f"{'predictor':>14} | {'daytime_mean':>12} | {'daytime_std':>11} | {'across_night':>12}")
    print("-" * 60)
    for name, builder in PREDICTORS.items():
        r = score_predictor(train, builder)
        print(f"{name:>14} | {r['daytime_mean']:>12.4f} | "
              f"{r['daytime_std']:>11.4f} | {r['across_day_night']:>12.4f}")

    # Blend-weight sweep: how much to trust the lookup vs the model (seen cells).
    # w=1.0 is pure lookup; w=0.0 is pure model. Find the best on across_night.
    print("\nBlend-weight sweep (w = weight on lookup, for SEEN locations):")
    print(f"{'w':>6} | {'daytime_mean':>12} | {'across_night':>12}")
    print("-" * 36)
    for w in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        r = score_predictor(train, lambda ref, _w=w: make_hybrid_blend(ref, w=_w))
        print(f"{w:>6.2f} | {r['daytime_mean']:>12.4f} | {r['across_day_night']:>12.4f}")

    print("\nThe predictor with the highest across_night is the best bet for the")
    print("real (mostly-seen) test set. To create its submission:")
    print("    python -m src.predict --submit <name>")

    if args.submit:
        out = f"{cfg['paths']['output_dir']}/predict_{args.submit}.csv"
        sub = make_submission(train, test, args.submit, out)
        print(f"\nWrote: {out}   (mean predicted demand {sub['demand'].mean():.4f})")
