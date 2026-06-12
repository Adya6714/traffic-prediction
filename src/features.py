"""Stage 5 — Feature engineering.

This file turns what we LEARNED in EDA into columns a model can use. Every
feature here traces back to a specific assumption in ASSUMPTIONS_LOG.md.

The features we build (and why)
-------------------------------
RAW features (the row's own attributes):
  - slot, sin_slot, cos_slot   -> time of day (A6: matters, esp. highways)
  - NumberofLanes              -> A2 (weak once RoadType known, but cheap to keep)
  - RoadType (encoded)         -> A1: the strongest level driver
  - LargeVehicles (encoded)    -> A3 (redundant with RoadType, cheap to keep)

HISTORY features (what demand usually looks like for rows "like this"):
  - te_geohash         -> this location's typical demand (A12, A17)
  - te_p5, te_p4       -> typical demand of the location's NEIGHBOURHOOD
                          (geohash prefixes = nearby cells). This is the PREFIX
                          POOLING that A18 says is worth ~4 leaderboard points,
                          and it is what saves the 10 cold-start geohashes.
  - te_geohash_slot    -> this location's typical demand AT THIS TIME (A17)
  - te_p4_slot         -> the neighbourhood's typical demand at this time
  - te_roadtype        -> typical demand for this road type (A1)
  - te_roadtype_slot   -> road type's demand at this time (captures highway
                          evening spike, A6/A10)

DROPPED (rejected in EDA): Weather (A5), Temperature (A7), Landmarks (A4).

The leakage rule, made concrete
--------------------------------
"te_" features are AVERAGES OF THE TARGET for a group. If we computed them on the
same rows we then train on, the average would include each row's own answer ->
leakage -> fake-high local score, real-world failure.

So the builder uses the sklearn-style fit/transform split:
  - fit(reference_df):  LEARN the average tables from a reference set.
  - transform(df):      APPLY those tables to any rows.
You fit on training data and transform test data. The tables never see test
demand. In our spatial folds, val geohashes are held out entirely, so val rows'
demand is absent from the tables too — clean by construction.

For training a model ON the same rows used to build the tables, use
fit_transform_oof(), which computes each row's features from OTHER rows only
(out-of-fold). That is the textbook fix for target-encoding leakage.

Smoothing (shrinkage)
---------------------
A single day-48 observation for a (geohash, slot) cell is noisy. We "shrink" it
toward a more stable parent average:
    smoothed = (sum_of_group + m * parent_mean) / (count_of_group + m)
With few observations (small count), the estimate leans on the parent; with many,
it trusts the group. `m` controls how much we distrust small groups. This
shrinkage is the other half of the ~4-point gap from A18.

Run a quick self-check:
    python -m src.features
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

TARGET = "demand"


class FeatureBuilder:
    """Builds model features with leakage-safe, smoothed, hierarchical encodings."""

    def __init__(self, smoothing: dict | None = None):
        # How hard to shrink each encoding toward its parent. Bigger = more shrink
        # (trust small groups less). Tuned to sensible defaults; adjust later.
        self.m = smoothing or {
            "geohash": 10.0,
            "p5": 15.0,
            "p4": 20.0,
            "geohash_slot": 5.0,
            "p4_slot": 10.0,
            "roadtype": 50.0,
            "roadtype_slot": 50.0,
        }
        self.tables_: dict = {}
        self.global_mean_: float = np.nan
        self.roadtype_levels_ = ["Highway", "Street", "Residential", "missing"]

    # ----------------------- helpers ----------------------- #
    @staticmethod
    def _add_keys(df: pd.DataFrame) -> pd.DataFrame:
        """Add prefix and helper columns used for grouping."""
        df = df.copy()
        df["RoadType"] = df["RoadType"].fillna("missing")
        df["p5"] = df["geohash"].str[:5]
        df["p4"] = df["geohash"].str[:4]
        return df

    def _smoothed_table(self, ref, keys, prior, m):
        """Return a Series mapping each group (by `keys`) to its shrunk mean demand.
        `prior` is either a float (e.g. global mean) or a Series the parent mean
        is looked up from per group."""
        agg = ref.groupby(keys)[TARGET].agg(["sum", "count"])
        if isinstance(prior, (int, float)):
            prior_vals = prior
        else:
            # prior is a Series indexed by the PARENT key (first element of keys)
            parent_key = keys[0] if isinstance(keys, list) else keys
            parent_of_group = agg.index.get_level_values(parent_key) \
                if isinstance(agg.index, pd.MultiIndex) else agg.index
            prior_vals = pd.Series(parent_of_group, index=agg.index).map(prior).values
        return (agg["sum"] + m * prior_vals) / (agg["count"] + m)

    # ----------------------- fit ----------------------- #
    def fit(self, reference: pd.DataFrame) -> "FeatureBuilder":
        """Learn all encoding tables from `reference` (the allowed training data)."""
        ref = self._add_keys(reference)
        g = ref[TARGET].mean()
        self.global_mean_ = float(g)

        # Coarse-to-fine, each shrunk toward its parent.
        t = {}
        t["roadtype"]      = self._smoothed_table(ref, ["RoadType"], g, self.m["roadtype"])
        t["roadtype_slot"] = self._smoothed_table(ref, ["RoadType", "slot"], t["roadtype"], self.m["roadtype_slot"])
        t["p4"]            = self._smoothed_table(ref, ["p4"], g, self.m["p4"])
        t["p5"]            = self._smoothed_table(ref, ["p5"], g, self.m["p5"])
        t["geohash"]       = self._smoothed_table(ref, ["geohash"], g, self.m["geohash"])
        t["p4_slot"]       = self._smoothed_table(ref, ["p4", "slot"], t["p4"], self.m["p4_slot"])
        t["geohash_slot"]  = self._smoothed_table(ref, ["geohash", "slot"], t["geohash"], self.m["geohash_slot"])
        self.tables_ = t
        return self

    # ----------------------- transform ----------------------- #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Produce the feature matrix for `df` using the learned tables."""
        if not self.tables_:
            raise RuntimeError("call fit() before transform()")
        x = self._add_keys(df)
        g = self.global_mean_
        out = pd.DataFrame(index=df.index)

        # ---- raw time features ----
        out["slot"] = x["slot"]
        out["sin_slot"] = np.sin(2 * np.pi * x["slot"] / 96)
        out["cos_slot"] = np.cos(2 * np.pi * x["slot"] / 96)
        out["NumberofLanes"] = x["NumberofLanes"]

        # ---- raw categoricals as small integers (model-friendly) ----
        rt_map = {name: i for i, name in enumerate(self.roadtype_levels_)}
        missing_code = rt_map["missing"]
        out["RoadType_code"] = x["RoadType"].apply(rt_map.get, args=(missing_code,)).astype(int)
        out["LargeVehicles_code"] = (x["LargeVehicles"] == "Allowed").astype(int)

        # ---- history (target-encoded) features ----
        def lookup(table, keys):
            if isinstance(keys, list):
                idx = pd.MultiIndex.from_arrays([x[k] for k in keys])
            else:
                idx = pd.Index(x[keys])
            return np.array(table.reindex(idx).to_numpy(), dtype=float)

        out["te_roadtype"]      = lookup(self.tables_["roadtype"], "RoadType")
        out["te_roadtype_slot"] = lookup(self.tables_["roadtype_slot"], ["RoadType", "slot"])
        out["te_p4"]            = lookup(self.tables_["p4"], "p4")
        out["te_p5"]            = lookup(self.tables_["p5"], "p5")
        out["te_geohash"]       = lookup(self.tables_["geohash"], "geohash")
        out["te_p4_slot"]       = lookup(self.tables_["p4_slot"], ["p4", "slot"])
        out["te_geohash_slot"]  = lookup(self.tables_["geohash_slot"], ["geohash", "slot"])

        # ---- a flag so the model knows when the fine location signal is missing ----
        out["is_cold_geohash"] = out["te_geohash"].isna().astype(int)

        # ---- fallback chain: fill any NaN encoding from coarser ones, then global ----
        out["te_geohash"]      = out["te_geohash"].fillna(out["te_p5"]).fillna(out["te_p4"]).fillna(out["te_roadtype"]).fillna(g)
        out["te_geohash_slot"] = out["te_geohash_slot"].fillna(out["te_p4_slot"]).fillna(out["te_roadtype_slot"]).fillna(out["te_geohash"]).fillna(g)
        for c in ["te_p5", "te_p4", "te_p4_slot", "te_roadtype", "te_roadtype_slot"]:
            out[c] = out[c].fillna(g)

        return out

    def fit_transform(self, reference: pd.DataFrame, df: pd.DataFrame | None = None) -> pd.DataFrame:
        self.fit(reference)
        return self.transform(reference if df is None else df)

    # ----------------------- leakage-safe training features ----------------------- #
    def fit_transform_oof(self, train: pd.DataFrame, n_folds: int = 5,
                          group: str = "geohash", seed: int = 42) -> pd.DataFrame:
        """Build training features WITHOUT leakage: each row's encodings come from
        OTHER rows only. We split by `group` (geohash) so a geohash's own rows do
        not encode themselves — mirroring how test geohashes are encoded from train.

        Use this to create the matrix you TRAIN the model on. For the final test
        prediction, use fit(all_train) then transform(test) instead.
        """
        groups = np.array(sorted(train[group].unique()))
        rng = np.random.RandomState(seed)
        rng.shuffle(groups)
        chunks = np.array_split(groups, n_folds)

        parts = []
        for chunk in chunks:
            hold = chunk.tolist()
            ref = train.loc[~train[group].isin(hold)]
            tgt = train.loc[train[group].isin(hold)]
            self.fit(ref)
            parts.append(self.transform(tgt))
        feats = pd.concat(parts).sort_index()
        # refit on ALL train so the builder is ready for the real test afterwards
        self.fit(train)
        return feats


# feature column groups (handy for the model stage)
RAW_FEATURES = ["slot", "sin_slot", "cos_slot", "NumberofLanes",
                "RoadType_code", "LargeVehicles_code"]
HISTORY_FEATURES = ["te_roadtype", "te_roadtype_slot", "te_p4", "te_p5",
                    "te_geohash", "te_p4_slot", "te_geohash_slot", "is_cold_geohash"]
ALL_FEATURES = RAW_FEATURES + HISTORY_FEATURES


if __name__ == "__main__":
    cfg = load_config()
    train, test = load_raw(cfg)

    print("Fitting FeatureBuilder on day 48, transforming a sample...\n")
    fb = FeatureBuilder()
    day48 = train.loc[train["day"] == 48]
    fb.fit(day48)
    feats = fb.transform(test)

    print("Feature columns produced:")
    for c in ALL_FEATURES:
        print(f"  {c}")
    print(f"\nShape of test feature matrix: {feats.shape}")
    print("\nAny missing values left? (should be all 0)")
    print(feats[ALL_FEATURES].isna().sum().to_string())
    print("\nFirst few rows:")
    print(feats[ALL_FEATURES].head().round(4).to_string())

    print("\nSanity: cold-start geohashes in test flagged:",
          int(feats["is_cold_geohash"].sum()), "rows")
    print("\nCorrelation of each history feature with NOTHING here (no target in test).")
    print("That check happens in the model stage via the folds.")
