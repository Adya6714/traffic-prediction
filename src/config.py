"""Load config.yaml and expose it as a plain dict, plus a global seed setter.

Why this exists: every other module imports `load_config()` instead of reading
files or hard-coding paths. Change config.yaml, and the whole pipeline follows.
"""
from __future__ import annotations
import os
import random
from pathlib import Path

import numpy as np
import yaml

# Repo root = parent of the src/ directory this file lives in.
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike = "config.yaml") -> dict:
    cfg_path = ROOT / path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Resolve all paths to absolute, anchored at repo root, so the pipeline
    # runs identically no matter what directory you launch it from.
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str(ROOT / v)
    return cfg


def set_seed(seed: int) -> None:
    """Make runs reproducible. Tree libraries take their own seed at fit time;
    this covers Python/NumPy and the PYTHONHASHSEED used by some encoders."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    cfg = load_config()
    print("Repo root:", ROOT)
    print("Resolved paths:")
    for k, v in cfg["paths"].items():
        exists = "ok" if Path(v).exists() else "MISSING"
        print(f"  {k:20s} {exists:8s} {v}")
