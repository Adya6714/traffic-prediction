"""Stage 8 — The corrector model (the synthesis).

diag03 check D proved: [day-48 base cell value + the row's OWN day-49 features]
predicts day-49 demand far better (83) than the base alone (49). So instead of
blending a fixed lookup with a generic model, we train ONE model that takes the
day-48 base AS A FEATURE alongside the row's own attributes. The model learns how
a row's own RoadType/slot/Temperature/Weather/lanes bend its day-48 base value.

Why this is leakage-safe
------------------------
- base_cell = day-48 (geohash, slot) mean. For a day-49 test row this is genuine
  history (a different day), no leak. For TRAINING on day-48 rows we must avoid a
  row seeing its own demand as its base, so we build the base out-of-fold by
  geohash (a geohash's base comes from... itself on day 48 — which IS the row).
  To prevent that, for training we use a leave-one-out base: each day-48 row's
  base = the geohash's mean over its OTHER slots, plus prefix fallback. At test
  time we use the full day-48 (geohash, slot) mean.

How we validate
---------------
The honest signal is the across_day fold (train day48 -> predict day49 morning),
because it actually spans days like the real task. We report it, plus the daytime
spatial folds for the cold-location view. But the LEADERBOARD is the real judge.

Run:
    python -m src.corrector            # report validation
    python -m src.corrector --submit   # also write a submission
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import load_config, set_seed
from src.data import load_raw
from src.splits import get_all_folds, competition_score

TARGET = "demand"

OWN_FEATURES = ["slot", "sin_slot", "cos_slot", "NumberofLanes",
                "RoadType_code", "LargeVehicles_code", "Temperature", "weather_code"]
FEATURES = ["base_cell", "base_geo", "base_p4"] + OWN_FEATURES


# --------------------------------------------------------------------------- #
# Base tables (day-48 history) and own-feature encoding.
# --------------------------------------------------------------------------- #
ROAD_MAP = {"Highway": 0, "Street": 1, "Residential": 2, "missing": 3}


def _own_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["slot"] = df["slot"]
    x["sin_slot"] = np.sin(2 * np.pi * df["slot"] / 96)
    x["cos_slot"] = np.cos(2 * np.pi * df["slot"] / 96)
    x["NumberofLanes"] = df["NumberofLanes"]
    x["RoadType_code"] = df["RoadType"].fillna("missing").map(ROAD_MAP).fillna(3).astype(int)
    x["LargeVehicles_code"] = (df["LargeVehicles"] == "Allowed").astype(int)
    x["Temperature"] = df["Temperature"]
    x["weather_code"] = df["Weather"].astype("category").cat.codes
    return x


def _base_tables(d48: pd.DataFrame):
    d48 = d48.copy()
    d48["p4"] = d48["geohash"].str[:4]
    mu = d48[TARGET].mean()
    gs  = d48.groupby(["geohash", "slot"])[TARGET].mean()
    geo = d48.groupby("geohash")[TARGET].mean()
    p4  = d48.groupby("p4")[TARGET].mean()
    rts = d48.groupby(["RoadType", "slot"])[TARGET].mean()
    return dict(gs=gs, geo=geo, p4=p4, rts=rts, mu=mu)


def _apply_base(df, tab):
    x = df.copy(); x["p4"] = x["geohash"].str[:4]
    idx = pd.MultiIndex.from_arrays([x["geohash"], x["slot"]])
    base_cell = np.array(tab["gs"].reindex(idx).to_numpy(), dtype=float)
    base_geo  = np.array(x["geohash"].map(tab["geo"]).to_numpy(), dtype=float)
    base_p4   = np.array(x["p4"].map(tab["p4"]).to_numpy(), dtype=float)
    # fallbacks for base_cell: geohash -> p4 -> roadtype+slot -> global
    need = np.isnan(base_cell); base_cell[need] = base_geo[need]
    need = np.isnan(base_cell); base_cell[need] = base_p4[need]
    need = np.isnan(base_cell)
    if need.any():
        rt = x["RoadType"].fillna("missing")
        base_cell[need] = np.array(tab["rts"].reindex(
            pd.MultiIndex.from_arrays([rt, x["slot"]])).to_numpy())[need]
    base_cell[np.isnan(base_cell)] = tab["mu"]
    base_geo[np.isnan(base_geo)] = tab["mu"]
    base_p4[np.isnan(base_p4)] = tab["mu"]
    out = pd.DataFrame(index=df.index)
    out["base_cell"] = base_cell; out["base_geo"] = base_geo; out["base_p4"] = base_p4
    return out


def build_features(reference: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Features for `df`, with base tables learned from `reference`'s day 48."""
    tab = _base_tables(reference[reference["day"] == 48])
    base = _apply_base(df, tab)
    own = _own_features(df)
    return pd.concat([base, own], axis=1)[FEATURES]


def lgb_params():
    return dict(n_estimators=500, learning_rate=0.03, num_leaves=63,
                min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=42, verbose=-1)


def train_corrector(reference: pd.DataFrame):
    """Train on `reference`. We train the model to predict demand from base+own.
    Training rows are day-48; their base_cell would equal their own demand (1 obs
    per cell), so we DROP base_cell's self-information by training base on day-48
    but predicting day-48 — handled by using the geohash leave-one-slot-out mean
    as base_cell for training only."""
    ref = reference.copy()
    d48 = ref[ref["day"] == 48]

    # Leave-one-out base_cell for training: geohash mean over OTHER slots.
    g_sum = d48.groupby("geohash")[TARGET].transform("sum")
    g_cnt = d48.groupby("geohash")[TARGET].transform("count")
    loo_geo = (g_sum - d48[TARGET]) / (g_cnt - 1).clip(lower=1)

    Xtr = build_features(ref, d48).copy()
    Xtr["base_cell"] = loo_geo.values            # replace self-leaking base
    ytr = d48[TARGET].to_numpy()

    model = lgb.LGBMRegressor(**lgb_params())
    model.fit(Xtr[FEATURES], ytr)

    def predict_fn(df):
        X = build_features(ref, df)
        return np.clip(model.predict(X[FEATURES]), 0, 1)

    return predict_fn, model


def score_on_folds(train):
    folds = get_all_folds(train)
    daytime, across, per = [], None, {}
    for fold in folds:
        predict_fn, _ = train_corrector(fold.train)
        sc = competition_score(fold.val[TARGET].to_numpy(), predict_fn(fold.val))
        per[fold.name] = round(sc, 4)
        (daytime.append(sc) if fold.fold_type == "daytime" else None)
        if fold.fold_type != "daytime":
            across = sc
    return {"daytime_mean": round(float(np.mean(daytime)), 4),
            "across_day_night": round(across, 4), "per_fold": per}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    set_seed(42)
    cfg = load_config()
    train, test = load_raw(cfg)

    print("Scoring corrector on folds...\n")
    r = score_on_folds(train)
    print(f"  daytime_mean (cold-location view): {r['daytime_mean']}")
    print(f"  across_day_night (across-day view): {r['across_day_night']}")
    print(f"  per-fold: {r['per_fold']}")
    print("\nReference points: best so far = batch2_blend_raw_w070 LIVE 86.33;")
    print("diag03 check D showed base+features ~ 83 on nighttime morning.")

    if args.submit:
        predict_fn, model = train_corrector(train)
        preds = predict_fn(test)
        sub = pd.DataFrame({"Index": test["Index"].astype(int), "demand": preds})
        assert sub.shape == (41778, 2) and sub["Index"].tolist() == test["Index"].tolist()
        out = f"{cfg['paths']['output_dir']}/corrector.csv"
        sub.to_csv(out, index=False)
        print(f"\nWrote {out}  (mean pred {preds.mean():.4f})")
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("\nFeature importance:")
        print(imp.to_string())
