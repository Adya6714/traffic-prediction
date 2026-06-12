"""Generate Batch 1 submission files for the leaderboard.

With unlimited submissions, the leaderboard is now our measuring instrument. This
script writes FOUR submission CSVs, each isolating ONE idea, so we can submit them
and read what actually works on the DAYTIME test (not the nighttime fold proxy).

Files written to outputs/:
  batch1_lookup_only.csv     pure geohash all-day average (our clean anchor;
                             should land near the 83.6 reference family)
  batch1_hybrid_w070.csv     0.7*lookup + 0.3*model for seen, model for cold
  batch1_hybrid_w050.csv     0.5*lookup + 0.5*model for seen, model for cold
  batch1_hybrid_hard.csv     lookup for seen, model for cold

The question this batch answers: does the model CORRECTION help or hurt on daytime?
The nighttime fold said "more model helps" but we suspect that is a nighttime
artifact. The leaderboard will settle it.

Run:
    python -m experiments.gen_batch1
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from src.config import load_config, set_seed
from src.data import load_raw
from src.predict import build_lookup, make_lookup_only, make_hybrid_hard, make_hybrid_blend

TARGET = "demand"


def write_submission(predict_fn, test, out_path):
    preds = np.clip(predict_fn(test), 0, 1)
    sub = pd.DataFrame({"Index": test["Index"].astype(int), "demand": preds})
    assert sub.shape == (41778, 2), f"bad shape {sub.shape}"
    assert sub["Index"].tolist() == test["Index"].tolist(), "Index order mismatch"
    sub.to_csv(out_path, index=False)
    return float(preds.mean())


if __name__ == "__main__":
    set_seed(42)
    cfg = load_config()
    train, test = load_raw(cfg)
    out_dir = cfg["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Build each predictor on ALL training data (day48 + day49 morning).
    print("Building predictors on all training data and writing submissions...\n")

    jobs = [
        ("batch1_lookup_only.csv", make_lookup_only(train)),
        ("batch1_hybrid_w070.csv", make_hybrid_blend(train, w=0.7)),
        ("batch1_hybrid_w050.csv", make_hybrid_blend(train, w=0.5)),
        ("batch1_hybrid_hard.csv", make_hybrid_hard(train)),
    ]

    print(f"{'file':>26} | {'mean predicted demand':>22}")
    print("-" * 52)
    for fname, fn in jobs:
        path = os.path.join(out_dir, fname)
        mean_pred = write_submission(fn, test, path)
        print(f"{fname:>26} | {mean_pred:>22.4f}")

    print(f"\nAll four written to: {out_dir}/")
    print("\nSUBMIT ORDER (record each score in ASSUMPTIONS_LOG.md section 5):")
    print("  1. batch1_lookup_only.csv   <- expect ~83.6, confirms anchor")
    print("  2. batch1_hybrid_w070.csv")
    print("  3. batch1_hybrid_w050.csv")
    print("  4. batch1_hybrid_hard.csv")
    print("\nReading the result:")
    print("  - If lookup_only is BEST -> the model hurts on daytime; go pure lookup.")
    print("  - If a hybrid is BEST   -> the model helps; tune the weight next batch.")
