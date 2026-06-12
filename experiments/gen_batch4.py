"""Batch 4 — the legitimate climb toward the ~92 ceiling.

Champion: b3_threeway = 0.6*raw + 0.2*model + 0.2*corrector -> LIVE 87.47.

Forum-revealed legitimate techniques we had NOT used:
  1. Decode geohash -> continuous lat/lon (interpolate across nearby cells).
  2. More trees + lower learning rate (2000 trees @ 0.02 vs our 500 @ 0.03).
  3. CatBoost as a second learner, blended with LightGBM.

This batch builds a STRONGER corrector (call it `corr2`) that adds lat/lon and
uses both LightGBM (more trees) and CatBoost, then blends it into the champion
recipe. We compare several blends to find the new best.

Files:
  b4_corr2_only.csv             the stronger corrector alone (diagnostic)
  b4_threeway_corr2.csv         0.6 raw + 0.2 model + 0.2 corr2  (champion w/ upgrade)
  b4_blend_raw_corr2_w060.csv   0.6 raw + 0.4 corr2
  b4_blend_raw_corr2_w055.csv   0.55 raw + 0.45 corr2
  b4_fourway.csv                0.55 raw + 0.15 model + 0.15 corr + 0.15 corr2

Run:
    python -m experiments.gen_batch4
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

from src.config import load_config, set_seed
from src.data import load_raw
from src.geohash_decode import add_latlon
from src.model import train_model
from src.corrector import train_corrector, _own_features, _base_tables, _apply_base, ROAD_MAP
from experiments.gen_batch2 import build_raw_cell

TARGET = "demand"
CORR2_FEATURES = ["base_cell", "base_geo", "base_p4", "lat", "lon",
                  "slot", "sin_slot", "cos_slot", "NumberofLanes",
                  "RoadType_code", "LargeVehicles_code", "Temperature", "weather_code"]


def build_corr2_features(reference, df):
    """Corrector features + lat/lon."""
    tab = _base_tables(reference[reference["day"] == 48])
    base = _apply_base(df, tab)
    own = _own_features(df)
    ll = add_latlon(df)[["lat", "lon"]]
    out = pd.concat([base, own, ll.set_index(df.index)], axis=1)
    return out[CORR2_FEATURES]


def train_corr2(reference):
    """Stronger corrector: lat/lon + LightGBM(2000 trees) + CatBoost, blended.
    Leave-one-out base_cell for training (same anti-leak trick as corrector)."""
    ref = reference.copy()
    d48 = ref[ref["day"] == 48]
    g_sum = d48.groupby("geohash")[TARGET].transform("sum")
    g_cnt = d48.groupby("geohash")[TARGET].transform("count")
    loo_geo = (g_sum - d48[TARGET]) / (g_cnt - 1).clip(lower=1)

    Xtr = build_corr2_features(ref, d48).copy()
    Xtr["base_cell"] = loo_geo.values
    ytr = d48[TARGET].to_numpy()

    lgbm = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=63,
                             min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, random_state=42, verbose=-1)
    lgbm.fit(Xtr[CORR2_FEATURES], ytr)

    cb = CatBoostRegressor(iterations=2000, learning_rate=0.02, depth=8,
                           l2_leaf_reg=3.0, random_seed=42, verbose=0)
    cb.fit(Xtr[CORR2_FEATURES], ytr)

    def predict_fn(df):
        X = build_corr2_features(ref, df)
        p = 0.5 * lgbm.predict(X[CORR2_FEATURES]) + 0.5 * cb.predict(X[CORR2_FEATURES])
        return np.clip(p, 0, 1)

    return predict_fn


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

    print("Building all predictors (CatBoost + 2000-tree LGBM, ~2-4 min)...\n")
    raw_p   = build_raw_cell(train)(test)
    model_p = train_model(train, transform="none", use_oof=True)[0](test)
    corr_p  = train_corrector(train)[0](test)
    corr2_p = train_corr2(train)(test)

    jobs = [
        ("b4_corr2_only.csv",           corr2_p),
        ("b4_threeway_corr2.csv",       0.6*raw_p + 0.2*model_p + 0.2*corr2_p),
        ("b4_blend_raw_corr2_w060.csv", 0.6*raw_p + 0.4*corr2_p),
        ("b4_blend_raw_corr2_w055.csv", 0.55*raw_p + 0.45*corr2_p),
        ("b4_fourway.csv",              0.55*raw_p + 0.15*model_p + 0.15*corr_p + 0.15*corr2_p),
    ]

    print(f"{'file':>30} | {'mean pred':>10}")
    print("-" * 44)
    for fname, preds in jobs:
        mp = write(preds, test, out_dir, fname)
        print(f"{fname:>30} | {mp:>10.4f}")

    print(f"\nAll written to {out_dir}/")
    print("Champion to beat: 87.47 (b3_threeway). Submit all, record, keep the best.")
    print("Legit ceiling reported by forum participants: ~92-93.")
