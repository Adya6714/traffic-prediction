"""Agent batch 7 — refine per-RoadType weights around b6 winner (87.90).

b6 LB: hwy_model_up 87.90 beat champion; early-slot and combined adjustments hurt.
Grid small perturbations on Highway model weight and Residential raw weight only.

Run:
    python -m experiments.gen_batch7
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.data import load_raw
from src.model import train_model
from src.corrector import train_corrector
from experiments.gen_batch2 import build_raw_cell
from experiments.gen_batch4 import train_corr2

CHAMPION = np.array([0.55, 0.15, 0.15, 0.15], dtype=float)
HWY_WIN = np.array([0.45, 0.25, 0.15, 0.15], dtype=float)
RES_WIN = np.array([0.65, 0.10, 0.12, 0.13], dtype=float)


def _normalize(W: np.ndarray) -> np.ndarray:
    s = W.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return W / s


def apply_road_weights(df: pd.DataFrame, hwy: np.ndarray, res: np.ndarray) -> np.ndarray:
    W = np.tile(CHAMPION, (len(df), 1))
    road = df["RoadType"].fillna("missing").to_numpy()
    W[road == "Highway"] = hwy
    W[road == "Residential"] = res
    return _normalize(W)


def blend_four(raw, model, corr, corr2, W: np.ndarray) -> np.ndarray:
    P = np.column_stack([raw, model, corr, corr2])
    return np.clip((P * W).sum(axis=1), 0, 1)


def write(preds: np.ndarray, test: pd.DataFrame, out_dir: str, fname: str) -> float:
    sub = pd.DataFrame({"Index": test["Index"].astype(int), "demand": preds})
    assert sub.shape == (41778, 2)
    assert sub["Index"].tolist() == test["Index"].tolist()
    sub.to_csv(os.path.join(out_dir, fname), index=False)
    return float(preds.mean())


if __name__ == "__main__":
    set_seed(42)
    cfg = load_config()
    train, test = load_raw(cfg)
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print("Building four predictors (~2-4 min)...\n")
    raw_p = build_raw_cell(train)(test)
    model_p = train_model(train, transform="none", use_oof=True)[0](test)
    corr_p = train_corrector(train)[0](test)
    corr2_p = train_corr2(train)(test)

    schemes = [
        ("agent_b7_winner_repro.csv", HWY_WIN, RES_WIN),
        ("agent_b7_hwy_model030.csv", np.array([0.40, 0.30, 0.15, 0.15]), RES_WIN),
        ("agent_b7_hwy_model020.csv", np.array([0.50, 0.20, 0.15, 0.15]), RES_WIN),
        ("agent_b7_res_raw070.csv", HWY_WIN, np.array([0.70, 0.08, 0.11, 0.11])),
        ("agent_b7_hwy030_res070.csv", np.array([0.40, 0.30, 0.15, 0.15]), np.array([0.70, 0.08, 0.11, 0.11])),
    ]

    print(f"{'file':>32} | {'mean pred':>10}")
    print("-" * 46)
    for fname, hwy, res in schemes:
        W = apply_road_weights(test, hwy, res)
        preds = blend_four(raw_p, model_p, corr_p, corr2_p, W)
        mp = write(preds, test, out_dir, fname)
        print(f"{fname:>32} | {mp:>10.4f}")

    print(f"\nAll written to {out_dir}/")
    print("Champion to beat: 87.90 (agent_b6_hwy_model_up).")
