"""Batch 8 — multiplicative day-level adjustment (the new lever from diag06).

diag06 finding: our blend predicts day-49 at DAY-48 LEVELS, but day 49 is ~30%
busier (median region ratio 1.30, multiplicative). Scaling the day-48 base by the
region's day49/day48 MORNING ratio lifted the nighttime fold +22. A1 and A2 are the
same insight: shape is stable, the LEVEL shifts and is recoverable from day-49 morning.

This applies a multiplicative day-factor on top of the CHAMPION per-roadtype blend.
KEY RISK: the factor is estimated from MORNING (slots 0-8) but applied to DAYTIME
(slots 9-55); the daytime busy-ratio may differ. So we test DAMPED versions too:
  factor_damped = 1 + damp * (factor - 1),  damp in {1.0, 0.75, 0.5, 0.25}
damp=1 is full multiplicative; damp=0 is the current champion (no scaling).

We also try two region granularities for the ratio (prefix-4 vs prefix-5).

Files:
  agent_b8_champion_repro.csv      damp=0  (must equal current champion 87.90)
  agent_b8_damp025_p4.csv          damp=0.25, prefix-4 ratio
  agent_b8_damp050_p4.csv          damp=0.50, prefix-4 ratio
  agent_b8_damp075_p4.csv          damp=0.75, prefix-4 ratio
  agent_b8_damp100_p4.csv          damp=1.00, prefix-4 ratio (full)
  agent_b8_damp050_p5.csv          damp=0.50, prefix-5 ratio (finer region)

Run:
    python -m experiments.gen_batch8
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.data import load_raw
from src.model import train_model
from src.corrector import train_corrector
from experiments.gen_batch2 import build_raw_cell
from experiments.gen_batch4 import train_corr2

TARGET = "demand"

# champion per-roadtype four-way weights (raw, model, corr, corr2)
WEIGHTS = {
    "Highway":     (0.45, 0.25, 0.15, 0.15),
    "Residential": (0.65, 0.10, 0.12, 0.13),
    "_default":    (0.55, 0.15, 0.15, 0.15),
}


def champion_blend(test, raw_p, model_p, corr_p, corr2_p):
    rt = test["RoadType"].fillna("missing").to_numpy()
    out = np.zeros(len(test))
    for i in range(len(test)):
        w = WEIGHTS.get(rt[i], WEIGHTS["_default"])
        out[i] = w[0]*raw_p[i] + w[1]*model_p[i] + w[2]*corr_p[i] + w[3]*corr2_p[i]
    return np.clip(out, 0, 1)


def day_factor(train, test, prefix_len):
    """Region day49/day48 morning ratio, mapped onto test rows. Estimated on the
    MORNING slots present in day 49 (the only overlap), then applied to daytime."""
    d48 = train[train["day"] == 48].copy()
    d49 = train[train["day"] == 49].copy()
    morn = sorted(d49["slot"].unique())
    key = f"p{prefix_len}"
    for df in (d48, d49, test):
        df[key] = df["geohash"].str[:prefix_len]
    f49 = d49.groupby(key)[TARGET].mean()
    f48 = d48[d48["slot"].isin(morn)].groupby(key)[TARGET].mean()
    ratio = (f49 / f48).clip(0.5, 2.0)
    return test[key].map(ratio).fillna(1.0).to_numpy()


def write(preds, test, out_dir, fname):
    preds = np.clip(preds, 0, 1)
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

    print("Building predictors (~3-5 min)...\n")
    raw_p   = build_raw_cell(train)(test)
    model_p = train_model(train, transform="none", use_oof=True)[0](test)
    corr_p  = train_corrector(train)[0](test)
    corr2_p = train_corr2(train)(test)

    champ = champion_blend(test, raw_p, model_p, corr_p, corr2_p)
    fac_p4 = day_factor(train, test, 4)
    fac_p5 = day_factor(train, test, 5)

    def apply_damp(base, factor, damp):
        return np.clip(base * (1 + damp * (factor - 1)), 0, 1)

    jobs = [
        ("agent_b8_champion_repro.csv", apply_damp(champ, fac_p4, 0.00)),
        ("agent_b8_damp025_p4.csv",     apply_damp(champ, fac_p4, 0.25)),
        ("agent_b8_damp050_p4.csv",     apply_damp(champ, fac_p4, 0.50)),
        ("agent_b8_damp075_p4.csv",     apply_damp(champ, fac_p4, 0.75)),
        ("agent_b8_damp100_p4.csv",     apply_damp(champ, fac_p4, 1.00)),
        ("agent_b8_damp050_p5.csv",     apply_damp(champ, fac_p5, 0.50)),
    ]
    print(f"{'file':>32} | {'mean pred':>10}")
    print("-" * 46)
    for fname, preds in jobs:
        mp = write(preds, test, out_dir, fname)
        print(f"{fname:>32} | {mp:>10.4f}")

    print(f"\nchampion (damp=0) mean pred should match prior champion ~0.11-0.12.")
    print(f"prefix-4 day-factor: median {np.median(fac_p4):.3f}, range {fac_p4.min():.2f}-{fac_p4.max():.2f}")
    print("\nSubmit all. Champion to beat: 87.90. Watch whether scaling helps DAYTIME")
    print("(diag06 showed +22 on nighttime; daytime transfer is the open question).")
