"""Stage 1 - Data contract, loading, and validation.

This module does three things, in order:
  1. load_raw()      - read train/test exactly as given, add parsed time columns.
  2. validate()      - assert every assumption we hold about the data. Fail loud.
  3. profile()       - print a human-readable summary (shape, missingness, splits).

The philosophy: a forecasting pipeline that silently accepts bad data produces a
confident wrong answer. We would rather crash at load time with a clear message
than discover a broken assumption three stages downstream in a leaderboard score.

Run it directly to see the report:
    python -m src.data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _parse_time(df: pd.DataFrame) -> pd.DataFrame:
    """timestamp is 'H:M' on a 15-minute grid. Derive minute-of-day and a slot
    index 0..95. These are the backbone of every time feature later, so we parse
    once, here, where the raw data enters the pipeline."""
    h_m = df["timestamp"].str.split(":", expand=True).astype(int)
    df["minute"] = h_m[0] * 60 + h_m[1]
    df["slot"] = df["minute"] // 15
    return df


def load_raw(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    train = pd.read_csv(cfg["paths"]["train"])
    test = pd.read_csv(cfg["paths"]["test"])
    train = _parse_time(train)
    test = _parse_time(test)
    return train, test


# --------------------------------------------------------------------------- #
# Validation - each assumption is one assertion with a message you can act on.
# --------------------------------------------------------------------------- #
def validate(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> None:
    d, c = cfg["data"], cfg["contract"]
    target, keys = d["target"], d["panel_keys"]

    # --- column contract: test == train minus the target ---
    missing_in_train = (set(d["categorical"]) | set(d["numeric"]) | set(keys)
                        | {d["index_col"], target}) - set(train.columns)
    assert not missing_in_train, f"train is missing expected columns: {missing_in_train}"
    assert target not in test.columns, "test should NOT contain the target"
    assert set(train.columns) - {target} == set(test.columns), (
        "train and test column sets differ beyond the target "
        f"-> train-only={set(train.columns)-set(test.columns)-{target}}, "
        f"test-only={set(test.columns)-set(train.columns)}"
    )

    # --- panel uniqueness: one row per (day, timestamp, geohash) ---
    for name, df in [("train", train), ("test", test)]:
        dup = df.duplicated(subset=keys).sum()
        assert dup == 0, f"{name} has {dup} duplicate rows on panel keys {keys}"

    # --- geohash shape ---
    for name, df in [("train", train), ("test", test)]:
        bad = (df["geohash"].str.len() != d["geohash_len"]).sum()
        assert bad == 0, f"{name} has {bad} geohashes not of length {d['geohash_len']}"

    # --- target bounds and validity (train only) ---
    y = train[target]
    assert y.notna().all(), "target has missing values in train"
    assert (y > c["target_min_exclusive"]).all(), "target has values <= 0"
    assert (y <= c["target_max_inclusive"]).all(), "target has values > 1"

    # --- time grid: slots in range, days as expected ---
    for name, df in [("train", train), ("test", test)]:
        assert df["slot"].between(0, d["n_slots"] - 1).all(), f"{name} slot out of [0,{d['n_slots']-1}]"
    assert set(train["day"]).issubset(set(c["expected_days"])), \
        f"train days {sorted(set(train['day']))} not within {c['expected_days']}"

    # --- missingness within tolerance (drift detector) ---
    for col, tol in c["max_missing"].items():
        for name, df in [("train", train), ("test", test)]:
            frac = df[col].isna().mean()
            assert frac <= tol, f"{name}.{col} missingness {frac:.3f} exceeds tolerance {tol}"

    print("[validate] all data-contract assertions passed.")


# --------------------------------------------------------------------------- #
# Profiling - the facts you need before deciding the split and features.
# --------------------------------------------------------------------------- #
def profile(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> None:
    d = cfg["data"]
    tgt = d["target"]

    def slot_to_str(s):
        return f"{s*15//60}:{s*15%60:02d}"

    print("\n========== DATA PROFILE ==========")
    print(f"train shape: {train.shape}   test shape: {test.shape}")

    print("\n--- temporal coverage (the split structure) ---")
    for name, df in [("train", train), ("test", test)]:
        for day in sorted(df["day"].unique()):
            sub = df[df["day"] == day]
            slots = sorted(sub["slot"].unique())
            print(f"  {name} day {day}: {len(sub):6d} rows | "
                  f"{len(slots):3d} slots [{slot_to_str(slots[0])} .. {slot_to_str(slots[-1])}]")

    print("\n--- spatial coverage ---")
    g_tr, g_te = set(train["geohash"]), set(test["geohash"])
    print(f"  geohashes: train={len(g_tr)}  test={len(g_te)}  "
          f"test seen in train={len(g_te & g_tr)}  cold-start={len(g_te - g_tr)}")

    print("\n--- target distribution (train) ---")
    print(train[tgt].describe()[["min", "25%", "50%", "75%", "max"]].to_string())
    print(f"  skew={train[tgt].skew():.2f}  (heavily right-tailed -> consider a transform)")

    print("\n--- missingness (%) ---")
    miss = pd.DataFrame({
        "train": (train.isna().mean() * 100).round(2),
        "test": (test.reindex(columns=train.columns).isna().mean() * 100).round(2),
    })
    print(miss[miss.sum(axis=1) > 0].to_string())

    print("\n--- categoricals (train value counts) ---")
    for col in d["categorical"]:
        print(f"  {col}: {train[col].value_counts(dropna=False).to_dict()}")
    print("==================================\n")


if __name__ == "__main__":
    cfg = load_config()
    train, test = load_raw(cfg)
    validate(train, test, cfg)
    profile(train, test, cfg)
