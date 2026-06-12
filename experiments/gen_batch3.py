"""Batch 3 — the honest climb past 86.33.

Champion so far: batch2_blend_raw_w070 = 0.7*raw_cell + 0.3*model  -> LIVE 86.33.
This batch makes SAFE, incremental improvements:

  1. Fine-tune the blend weight around 0.7 (0.75 / 0.72 / 0.68 / 0.65).
  2. Strengthen the model: train it with day-49-style OWN features (Temperature,
     slot, RoadType, lanes, weather) PLUS the day-48 base, which diag03 check D
     showed lifts day-over-day prediction to ~83. Call it `corr` (corrector).
  3. Blend raw_cell with the STRONGER corrector model instead of the generic one.

Files (submit all, record scores):
  b3_raw_w075.csv / w072 / w068 / w065   raw_cell + generic model, weight sweep
  b3_corrblend_w070.csv / w060           raw_cell + CORRECTOR model
  b3_threeway.csv                        0.6*raw + 0.2*model + 0.2*corrector

Run:
    python -m experiments.gen_batch3
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.data import load_raw
from src.model import train_model
from src.corrector import train_corrector
from experiments.gen_batch2 import build_raw_cell


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

    print("Building bases and models on all training data (takes a minute)...\n")
    raw_fn = build_raw_cell(train)
    model_fn, _, _ = train_model(train, transform="none", use_oof=True)
    corr_fn, _ = train_corrector(train)

    raw_p   = raw_fn(test)
    model_p = model_fn(test)
    corr_p  = corr_fn(test)

    def blend(a, b, w):
        return w * a + (1 - w) * b

    jobs = [
        # 1. weight sweep around champion (raw + generic model)
        ("b3_raw_w075.csv",       blend(raw_p, model_p, 0.75)),
        ("b3_raw_w072.csv",       blend(raw_p, model_p, 0.72)),
        ("b3_raw_w068.csv",       blend(raw_p, model_p, 0.68)),
        ("b3_raw_w065.csv",       blend(raw_p, model_p, 0.65)),
        # 2. raw + the STRONGER corrector model
        ("b3_corrblend_w070.csv", blend(raw_p, corr_p, 0.70)),
        ("b3_corrblend_w060.csv", blend(raw_p, corr_p, 0.60)),
        # 3. three-way
        ("b3_threeway.csv",       0.6 * raw_p + 0.2 * model_p + 0.2 * corr_p),
    ]

    print(f"{'file':>26} | {'mean pred':>10}")
    print("-" * 40)
    for fname, preds in jobs:
        mp = write(preds, test, out_dir, fname)
        print(f"{fname:>26} | {mp:>10.4f}")

    print(f"\nAll written to {out_dir}/")
    print("Champion to beat: 86.33 (batch2_blend_raw_w070).")
    print("Submit all, record scores. Keep whichever beats 86.33 as the new champion.")
