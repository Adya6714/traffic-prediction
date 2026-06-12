"""Stage 6 — The model (LightGBM).

We feed the 14 features from src.features into a LightGBM regressor and let it
learn how to combine them. Wired through src.splits.score_folds so every result
is measured the same honest way as the baselines.

Design choices (see ASSUMPTIONS_LOG.md section 6)
-------------------------------------------------
- ONE model (LightGBM) first. The relationship is simple (te_roadtype corr 0.87),
  so a 2nd library would predict near-identically. Add CatBoost + blend only if
  we plateau.
- CPU only. 77k x 14 is tiny; trains in seconds. USE_GPU flag exists but stays off.
- TARGET_TRANSFORM lets us test raw demand vs log1p(demand) (assumption A14).

How leakage is handled
----------------------
- Fold scoring: for each fold we fit the FeatureBuilder on fold.train and
  transform fold.val. In spatial folds the val geohashes are absent from train,
  so their history features are honest fallbacks (proven in the features stage).
- Training matrix for the FINAL model: built with fit_transform_oof so each
  training row's encodings come from OTHER geohashes only.

Run:
    python -m src.model              # scores LightGBM on the folds, both transforms
    python -m src.model --submit     # also trains on all data and writes a submission
"""
from __future__ import annotations
import argparse
from typing import Any
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import load_config, set_seed
from src.data import load_raw
from src.features import FeatureBuilder, ALL_FEATURES
from src.splits import get_all_folds, competition_score

TARGET = "demand"
USE_GPU = False                 # keep False — tiny data, CPU is faster
TARGET_TRANSFORM = "none"       # "none" or "log" (A14). Set per experiment.


def lgb_params() -> dict[str, Any]:
    p: dict[str, Any] = dict(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    if USE_GPU:
        p["device"] = "gpu"
    return p


# --------------------------------------------------------------------------- #
# Target transform helpers (A14). R2 is always measured on the RAW scale.
# --------------------------------------------------------------------------- #
def to_target(y: np.ndarray, transform: str) -> np.ndarray:
    return np.log1p(y) if transform == "log" else y

def from_target(p: np.ndarray, transform: str) -> np.ndarray:
    return np.expm1(p) if transform == "log" else p


# --------------------------------------------------------------------------- #
# Train one LightGBM on a reference set, return a predict_fn for raw demand.
# --------------------------------------------------------------------------- #
def train_model(reference: pd.DataFrame, transform: str = TARGET_TRANSFORM,
                use_oof: bool = True):
    """Fit FeatureBuilder + LightGBM on `reference`. Returns predict_fn(df)->raw demand."""
    fb = FeatureBuilder()
    if use_oof:
        X = fb.fit_transform_oof(reference)      # leakage-safe training features
    else:
        X = fb.fit_transform(reference)          # simpler; fb left fitted on reference
    y = to_target(reference[TARGET].to_numpy(), transform)

    model = lgb.LGBMRegressor(**lgb_params())
    model.fit(X[ALL_FEATURES], y)

    def predict_fn(df: pd.DataFrame) -> np.ndarray:
        feats = fb.transform(df)
        raw_preds = np.asarray(model.predict(feats[ALL_FEATURES]), dtype=float)
        preds = from_target(raw_preds, transform)
        return np.clip(preds, 0, 1)              # demand is bounded (A15)

    return predict_fn, model, fb


# --------------------------------------------------------------------------- #
# Score the model on the folds (the honest comparison).
# --------------------------------------------------------------------------- #
def score_on_folds(train: pd.DataFrame, transform: str) -> dict:
    folds = get_all_folds(train)
    per_fold, daytime, across = {}, [], 0.0
    for fold in folds:
        # train a fresh model on THIS fold's train, predict its val
        predict_fn, _, _ = train_model(fold.train, transform=transform, use_oof=False)
        preds = predict_fn(fold.val)
        sc = competition_score(fold.val[TARGET].to_numpy(), preds)
        per_fold[fold.name] = round(sc, 4)
        if fold.fold_type == "daytime":
            daytime.append(sc)
        else:
            across = sc
    return {
        "transform": transform,
        "daytime_mean": round(float(np.mean(daytime)), 4),
        "daytime_std": round(float(np.std(daytime)), 4),
        "across_day_night": round(across, 4),
        "per_fold": per_fold,
    }


def make_submission(train, test, out_path, transform):
    predict_fn, model, _ = train_model(train, transform=transform, use_oof=True)
    preds = predict_fn(test)
    sub = pd.DataFrame({"Index": test["Index"].astype(int), "demand": preds})
    assert sub.shape == (41778, 2) and sub["Index"].tolist() == test["Index"].tolist()
    sub.to_csv(out_path, index=False)
    return sub, model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="also write a submission CSV")
    args = ap.parse_args()

    set_seed(42)
    cfg = load_config()
    train, test = load_raw(cfg)

    print("Scoring LightGBM on the folds (this takes ~10-30s)...\n")
    print(f"{'transform':>10} | {'daytime_mean':>12} | {'daytime_std':>11} | {'across_night':>12}")
    print("-" * 56)
    results = {}
    for transform in ["none", "log"]:
        r = score_on_folds(train, transform)
        results[transform] = r
        print(f"{transform:>10} | {r['daytime_mean']:>12.4f} | "
              f"{r['daytime_std']:>11.4f} | {r['across_day_night']:>12.4f}")

    best = max(results, key=lambda t: results[t]["daytime_mean"])
    print(f"\nBest transform on daytime folds: '{best}'")
    print("Compare to baselines (geohash_mean daytime ~0, across_night ~65.6) and")
    print("to the live bar 83.6. Remember: daytime folds hold out LOCATIONS, so they")
    print("under-state performance on the real (mostly-seen) test set.")

    if args.submit:
        out = f"{cfg['paths']['output_dir']}/model_lgbm_{best}.csv"
        sub, model = make_submission(train, test, out, transform=best)
        print(f"\nWrote submission: {out}")
        print(f"  predicted demand mean={sub['demand'].mean():.4f}")
        # feature importance — which features the model actually used
        fb = FeatureBuilder().fit(train)
        imp = pd.Series(model.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
        print("\nFeature importance (how often the model split on each):")
        print(imp.to_string())
