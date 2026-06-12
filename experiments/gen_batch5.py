"""Batch 5 — fold the spatio-temporal signals into the model (the real test).

diag05 proved temporal-window + spatial-neighbour features lift nighttime CV
64 -> 71.4 (+7.4). This batch builds `corr3`: the corr2 model (lat/lon, 2000-tree
LightGBM + CatBoost) PLUS the spatio-temporal features (win-2,win-1,win1,win2,
neigh_mean,trend), then blends it into the champion recipe.

Champion to beat: b4_fourway = 87.65.

Files:
  b5_corr3_only.csv            the new model alone (diagnostic)
  b5_threeway_corr3.csv        0.6 raw + 0.2 model + 0.2 corr3
  b5_blend_raw_corr3_w055.csv  0.55 raw + 0.45 corr3
  b5_blend_raw_corr3_w050.csv  0.50 raw + 0.50 corr3
  b5_fiveway.csv               0.5 raw + 0.1 model + 0.1 corr + 0.1 corr2 + 0.2 corr3

Run:
    python -m experiments.gen_batch5
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
from src.spatiotemporal import SpatioTemporal
from src.model import train_model
from src.corrector import train_corrector, _own_features, _base_tables, _apply_base
from experiments.gen_batch2 import build_raw_cell
from experiments.gen_batch4 import train_corr2

TARGET = "demand"
ST_FEATURES = ["win-2", "win-1", "win1", "win2", "neigh_mean", "trend"]
CORR3_FEATURES = (["base_cell", "base_geo", "base_p4", "lat", "lon",
                   "slot", "sin_slot", "cos_slot", "NumberofLanes",
                   "RoadType_code", "LargeVehicles_code", "Temperature", "weather_code"]
                  + ST_FEATURES)


def build_corr3_features(reference, st, df):
    tab = _base_tables(reference[reference["day"] == 48])
    base = _apply_base(df, tab)
    own = _own_features(df)
    ll = add_latlon(df)[["lat", "lon"]].set_index(df.index)
    stf = st.transform(df)
    out = pd.concat([base, own, ll, stf], axis=1)
    return out[CORR3_FEATURES]


def train_corr3(reference):
    ref = reference.copy()
    st = SpatioTemporal().fit(ref)
    d48 = ref[ref["day"] == 48]
    g_sum = d48.groupby("geohash")[TARGET].transform("sum")
    g_cnt = d48.groupby("geohash")[TARGET].transform("count")
    loo_geo = (g_sum - d48[TARGET]) / (g_cnt - 1).clip(lower=1)

    Xtr = build_corr3_features(ref, st, d48).copy()
    Xtr["base_cell"] = loo_geo.values
    ytr = d48[TARGET].to_numpy()

    lgbm = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=63,
                             min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, random_state=42, verbose=-1)
    lgbm.fit(Xtr[CORR3_FEATURES], ytr)
    cb = CatBoostRegressor(iterations=2000, learning_rate=0.02, depth=8,
                           l2_leaf_reg=3.0, random_seed=42, verbose=0)
    cb.fit(Xtr[CORR3_FEATURES], ytr)

    def predict_fn(df):
        X = build_corr3_features(ref, st, df)
        p = 0.5 * lgbm.predict(X[CORR3_FEATURES]) + 0.5 * cb.predict(X[CORR3_FEATURES])
        return np.clip(p, 0, 1)
    return predict_fn, lgbm


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

    print("Building predictors (this is the heavy one, ~4-6 min)...\n")
    raw_p   = build_raw_cell(train)(test)
    model_p = train_model(train, transform="none", use_oof=True)[0](test)
    corr_p  = train_corrector(train)[0](test)
    corr2_p = train_corr2(train)(test)
    corr3_fn, lgbm3 = train_corr3(train)
    corr3_p = corr3_fn(test)

    jobs = [
        ("b5_corr3_only.csv",           corr3_p),
        ("b5_threeway_corr3.csv",       0.6*raw_p + 0.2*model_p + 0.2*corr3_p),
        ("b5_blend_raw_corr3_w055.csv", 0.55*raw_p + 0.45*corr3_p),
        ("b5_blend_raw_corr3_w050.csv", 0.50*raw_p + 0.50*corr3_p),
        ("b5_fiveway.csv",              0.5*raw_p + 0.1*model_p + 0.1*corr_p + 0.1*corr2_p + 0.2*corr3_p),
    ]

    print(f"{'file':>30} | {'mean pred':>10}")
    print("-" * 44)
    for fname, preds in jobs:
        mp = write(preds, test, out_dir, fname)
        print(f"{fname:>30} | {mp:>10.4f}")

    # show what corr3 leans on - did it use the new features?
    imp = pd.Series(lgbm3.feature_importances_, index=CORR3_FEATURES).sort_values(ascending=False)
    print("\ncorr3 feature importance (did the spatio-temporal feats matter?):")
    print(imp.to_string())
    print(f"\nAll written to {out_dir}/. Champion to beat: 87.65. Submit all, record.")
