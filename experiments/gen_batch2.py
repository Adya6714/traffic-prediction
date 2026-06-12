"""Batch 2 — restore per-cell SLOT information (the thing batch1 wrongly dropped).

batch1 lesson: switching the lookup to a geohash ALL-DAY average (no slot) scored
only 69 on the daytime leaderboard, WORSE than the earlier per-(geohash,slot)
lookup (exp01 ~79.6). The nighttime fold misled us: slot detail is noise at night
but MATTERS in daytime (highways have a daytime/evening pattern). And more model
weight kept helping (69->77->80) because the model retains the slot features.

So this batch goes back to per-cell SLOT predictions and blends with the model:
  batch2_raw_cell.csv            day-48 (geohash,slot) mean, fallback chain
                                 (control: should reproduce ~79.6 from exp01)
  batch2_smoothed_cell.csv       FeatureBuilder te_geohash_slot (slot, but smoothed
                                 toward the geohash mean -> less noisy than raw)
  batch2_blend_raw_w070.csv      0.7*raw_cell  + 0.3*model
  batch2_blend_raw_w050.csv      0.5*raw_cell  + 0.5*model
  batch2_blend_smooth_w050.csv   0.5*smoothed  + 0.5*model
  batch2_blend_smooth_w030.csv   0.3*smoothed  + 0.7*model

Submit all six, record scores. Questions answered:
  - raw vs smoothed per-cell: which base is better on daytime?
  - how much model weight is best when the base KEEPS slot info?

Run:
    python -m experiments.gen_batch2
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.data import load_raw
from src.features import FeatureBuilder
from src.model import train_model

TARGET = "demand"


def build_raw_cell(reference: pd.DataFrame):
    """Raw day-48 (geohash, slot) mean with hierarchical fallback.
    Keeps full slot detail (no smoothing). This is the exp01-style lookup."""
    d48 = reference[reference["day"] == 48].copy()
    d48["p4"] = d48["geohash"].str[:4]
    mu = d48[TARGET].mean()
    gs  = d48.groupby(["geohash", "slot"])[TARGET].mean()
    geo = d48.groupby("geohash")[TARGET].mean()
    p4s = d48.groupby(["p4", "slot"])[TARGET].mean()
    rts = d48.groupby(["RoadType", "slot"])[TARGET].mean()

    def fn(df):
        x = df.copy(); x["p4"] = x["geohash"].str[:4]
        idx = pd.MultiIndex.from_arrays([x["geohash"], x["slot"]])
        out = np.array(gs.reindex(idx).to_numpy(), dtype=float)
        need = np.isnan(out)
        if need.any():
            out[need] = x["geohash"].map(geo).to_numpy()[need]
        need = np.isnan(out)
        if need.any():
            out[need] = np.array(p4s.reindex(pd.MultiIndex.from_arrays([x["p4"], x["slot"]])).to_numpy())[need]
        need = np.isnan(out)
        if need.any():
            rt = x["RoadType"].fillna("missing")
            out[need] = np.array(rts.reindex(pd.MultiIndex.from_arrays([rt, x["slot"]])).to_numpy())[need]
        out[np.isnan(out)] = mu
        return np.clip(out, 0, 1)
    return fn


def write(predict_fn, test, out_dir, fname):
    preds = np.clip(predict_fn(test), 0, 1)
    sub = pd.DataFrame({"Index": test["Index"].astype(int), "demand": preds})
    assert sub.shape == (41778, 2) and sub["Index"].tolist() == test["Index"].tolist()
    sub.to_csv(os.path.join(out_dir, fname), index=False)
    return float(preds.mean())


if __name__ == "__main__":
    set_seed(42)
    cfg = load_config()
    train, test = load_raw(cfg)
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("Building bases and model on all training data...\n")
    raw_fn = build_raw_cell(train)
    fb = FeatureBuilder().fit(train)
    smooth_fn = lambda df: fb.transform(df)["te_geohash_slot"].to_numpy()
    model_fn, _, _ = train_model(train, transform="none", use_oof=True)

    # precompute on test
    raw_p   = raw_fn(test)
    smooth_p = smooth_fn(test)
    model_p = model_fn(test)

    def blend(a, b, w):  # w on first
        return np.clip(w * a + (1 - w) * b, 0, 1)

    jobs = [
        ("batch2_raw_cell.csv",          raw_p),
        ("batch2_smoothed_cell.csv",     smooth_p),
        ("batch2_blend_raw_w070.csv",    blend(raw_p, model_p, 0.7)),
        ("batch2_blend_raw_w050.csv",    blend(raw_p, model_p, 0.5)),
        ("batch2_blend_smooth_w050.csv", blend(smooth_p, model_p, 0.5)),
        ("batch2_blend_smooth_w030.csv", blend(smooth_p, model_p, 0.3)),
    ]

    print(f"{'file':>32} | {'mean pred':>10}")
    print("-" * 46)
    for fname, preds in jobs:
        mean_pred = write(lambda df, p=preds: p, test, out_dir, fname)
        print(f"{fname:>32} | {mean_pred:>10.4f}")

    print(f"\nAll written to {out_dir}/")
    print("\nSUBMIT and record each in ASSUMPTIONS_LOG.md section 5.")
    print("Current floor to beat: 83.6 (reference nofactor).")
    print("Compare raw_cell vs smoothed_cell, and find the best blend weight.")
