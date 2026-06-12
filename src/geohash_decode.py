"""Geohash decoding -> continuous latitude/longitude.

The forum's legitimate top scorers (~91-93) decode the 6-char geohash into the
centre lat/lon of its cell and feed those as CONTINUOUS features. This lets a
tree model interpolate across NEARBY locations (close lat/lon -> similar demand)
instead of treating each geohash as an isolated category. It is the spatial
analogue of why cyclical time encoding helps.

This is the standard public geohash algorithm (base-32, bit-interleaving). No
external library or data — we only decode the strings already in the dataset.

Run a quick check:
    python -m src.geohash_decode
"""
from __future__ import annotations
import numpy as np
import pandas as pd

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODE = {c: i for i, c in enumerate(_BASE32)}


def decode_geohash(gh: str) -> tuple[float, float]:
    """Return (lat, lon) of the centre of the geohash cell."""
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True  # start with longitude
    for ch in gh:
        idx = _DECODE[ch]
        for bit in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_lo + lon_hi) / 2
                if idx & bit:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if idx & bit:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2


def add_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """Add lat/lon columns by decoding each unique geohash once (fast)."""
    uniq = df["geohash"].unique()
    table = {g: decode_geohash(g) for g in uniq}
    lat = df["geohash"].map(lambda g: table[g][0])
    lon = df["geohash"].map(lambda g: table[g][1])
    out = df.copy()
    out["lat"] = lat.values
    out["lon"] = lon.values
    return out


if __name__ == "__main__":
    from src.config import load_config
    from src.data import load_raw
    cfg = load_config()
    train, _ = load_raw(cfg)
    d = add_latlon(train.head(1000))
    print("Sample decoded coordinates:")
    print(d[["geohash", "lat", "lon"]].drop_duplicates().head(10).to_string(index=False))
    print(f"\nlat range: {d['lat'].min():.4f} .. {d['lat'].max():.4f}")
    print(f"lon range: {d['lon'].min():.4f} .. {d['lon'].max():.4f}")
    print("(forum says these land in the Indian Ocean ~ lat -5.x, lon 90.x)")
