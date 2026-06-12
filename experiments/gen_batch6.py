"""Agent batch 6 — per-RoadType / per-slot blend weights (idea #1).

Hypothesis: the champion uses one global four-way split (0.55/0.15/0.15/0.15).
Highways are peaky (model + corr2 help); residential is flat (raw lookup suffices);
early test slots (9-12) follow day-49 morning and may want more corrector weight.

Reuses the same four predictors as b4_fourway; only the blend weights change per row.

Run:
    python -m experiments.gen_batch6

Outputs (submit batch, record scores in AGENT_WORKLOG.md):
  agent_b6_champion_repro.csv      control = global champion weights
  agent_b6_hwy_model_up.csv        Highway: more model; Residential: more raw
  agent_b6_early_corr_up.csv       slots 9-12: more corr + corr2
  agent_b6_combined_v1.csv         Highway + early-slot adjustments together
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

TARGET = "demand"

# Champion four-way weights (must sum to 1)
CHAMPION = np.array([0.55, 0.15, 0.15, 0.15], dtype=float)


def _normalize(W: np.ndarray) -> np.ndarray:
    s = W.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return W / s


def weights_champion(n: int) -> np.ndarray:
    return np.tile(CHAMPION, (n, 1))


def weights_hwy_model_up(df: pd.DataFrame) -> np.ndarray:
    W = weights_champion(len(df))
    road = df["RoadType"].fillna("missing").to_numpy()
    W[road == "Highway"] = [0.45, 0.25, 0.15, 0.15]
    W[road == "Residential"] = [0.65, 0.10, 0.12, 0.13]
    return _normalize(W)


def weights_early_corr_up(df: pd.DataFrame) -> np.ndarray:
    W = weights_champion(len(df))
    early = df["slot"].to_numpy() <= 12
    W[early] = [0.45, 0.10, 0.25, 0.20]
    return _normalize(W)


def weights_combined_v1(df: pd.DataFrame) -> np.ndarray:
    W = weights_champion(len(df))
    road = df["RoadType"].fillna("missing").to_numpy()
    slot = df["slot"].to_numpy()
    W[road == "Highway"] = [0.45, 0.25, 0.15, 0.15]
    W[road == "Residential"] = [0.65, 0.10, 0.12, 0.13]
    early = slot <= 12
    W[early] = [0.42, 0.10, 0.28, 0.20]
    return _normalize(W)


def blend_four(
    df: pd.DataFrame,
    raw: np.ndarray,
    model: np.ndarray,
    corr: np.ndarray,
    corr2: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
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

    print("Building four predictors (same as b4_fourway, ~2-4 min)...\n")
    raw_p = build_raw_cell(train)(test)
    model_p = train_model(train, transform="none", use_oof=True)[0](test)
    corr_p = train_corrector(train)[0](test)
    corr2_p = train_corr2(train)(test)

    schemes = [
        ("agent_b6_champion_repro.csv", weights_champion),
        ("agent_b6_hwy_model_up.csv", weights_hwy_model_up),
        ("agent_b6_early_corr_up.csv", weights_early_corr_up),
        ("agent_b6_combined_v1.csv", weights_combined_v1),
    ]

    print(f"{'file':>32} | {'mean pred':>10}")
    print("-" * 46)
    for fname, w_fn in schemes:
        W = w_fn(test) if w_fn is not weights_champion else weights_champion(len(test))
        preds = blend_four(test, raw_p, model_p, corr_p, corr2_p, W)
        mp = write(preds, test, out_dir, fname)
        print(f"{fname:>32} | {mp:>10.4f}")

    print(f"\nAll written to {out_dir}/")
    print("Champion to beat: 87.65 (b4_fourway). Submit all four, record in AGENT_WORKLOG.md.")
