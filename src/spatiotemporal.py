"""Stage 9 — Spatio-temporal features (the ground-up lever).

diag05 proved two NEW signals lift nighttime CV from 64 -> 71.4:
  - TEMPORAL WINDOW: day-48 demand at slots t-2,t-1,t+1,t+2 for the same cell.
    Captures the local CURVE SHAPE (rising/falling), not just the single point.
  - SPATIAL NEIGHBOURS: average day-48 demand of the K geographically nearest
    cells at the same slot. Adjacent roads have correlated traffic.

Both are built ONLY from day-48 (the reference), so for a day-49/test row they
use a different day's history -> leakage-safe. For training on day-48 rows we use
a leave-one-out variant of the cell's own value (handled in the model stage); the
window/neighbour features come from OTHER cells/slots so they do not leak the row.

Run:
    python -m src.spatiotemporal
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.linalg import norm

from src.config import load_config
from src.data import load_raw
from src.geohash_decode import add_latlon

TARGET = "demand"
WIN_OFFSETS = [-2, -1, 1, 2]
K_NEIGHBOURS = 5


class SpatioTemporal:
    """Learns day-48 temporal-window and spatial-neighbour tables from a reference,
    then attaches them as features to any dataframe."""

    def __init__(self, k=K_NEIGHBOURS, offsets=WIN_OFFSETS):
        self.k = k
        self.offsets = offsets

    def fit(self, reference: pd.DataFrame):
        d48 = add_latlon(reference[reference["day"] == 48].copy())
        self.mu_ = d48[TARGET].mean()

        # temporal: pivot geohash x slot -> demand
        self.piv_ = d48.pivot_table(index="geohash", columns="slot",
                                    values=TARGET, aggfunc="mean")
        # per-(geohash,slot) mean for neighbour lookups
        self.gs_ = d48.groupby(["geohash", "slot"])[TARGET].mean()

        # spatial: K nearest cells by lat/lon
        cells = d48[["geohash", "lat", "lon"]].drop_duplicates().reset_index(drop=True)
        coords = cells[["lat", "lon"]].to_numpy()
        ghs = cells["geohash"].to_numpy()
        self.nbr_ = {}
        for i, g in enumerate(ghs):
            d = norm(coords - coords[i], axis=1)
            self.nbr_[g] = ghs[np.argsort(d)[1:self.k + 1]]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        out = pd.DataFrame(index=df.index)

        # temporal window features
        def win(gh, s, off):
            try:
                v = self.piv_.loc[gh, s + off]
                return v if not pd.isna(v) else np.nan
            except KeyError:
                return np.nan
        for off in self.offsets:
            out[f"win{off}"] = [win(g, s, off) for g, s in zip(x["geohash"], x["slot"])]

        # spatial neighbour mean (same slot)
        def neigh(g, s):
            vals = [self.gs_.get((ng, s), np.nan) for ng in self.nbr_.get(g, [])]
            vals = [v for v in vals if not np.isnan(v)]
            return np.mean(vals) if vals else np.nan
        out["neigh_mean"] = [neigh(g, s) for g, s in zip(x["geohash"], x["slot"])]

        # local temporal trend: (t+1 avg) - (t-1 avg), a rise/fall indicator
        out["trend"] = out[["win1", "win2"]].mean(axis=1) - out[["win-1", "win-2"]].mean(axis=1)

        # fill missing with global mean (cold cells / edge slots)
        for c in out.columns:
            out[c] = out[c].fillna(self.mu_)
        return out

    @property
    def feature_names(self):
        return [f"win{o}" for o in self.offsets] + ["neigh_mean", "trend"]


if __name__ == "__main__":
    cfg = load_config()
    train, test = load_raw(cfg)
    st = SpatioTemporal().fit(train)
    f = st.transform(test)
    print("Spatio-temporal features for test:")
    print(f.head().round(4).to_string())
    print("\nfeatures:", st.feature_names)
    print("missing values:", int(f.isna().sum().sum()))
