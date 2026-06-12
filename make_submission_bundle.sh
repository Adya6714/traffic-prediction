#!/usr/bin/env bash
# GridLock 2.0 — verification bundle (notebook + minimal source to reproduce champion).
# Run from repo root:  bash make_submission_bundle.sh
set -euo pipefail

BUNDLE=submission_bundle
rm -rf "$BUNDLE" "$BUNDLE.zip"
mkdir -p "$BUNDLE/src" "$BUNDLE/experiments" "$BUNDLE/outputs"

# Required: methodology writeup + reproduction notebook
cp docs/APPROACH.md "$BUNDLE/"
cp reproduce.ipynb "$BUNDLE/"

# Runtime config and dependencies
cp config.yaml requirements.txt "$BUNDLE/"

# Minimal src/ modules on the champion import chain
for f in __init__ config data features splits model corrector geohash_decode; do
  cp "src/$f.py" "$BUNDLE/src/"
done

# Experiment helpers imported by the notebook / gen_batch6
for b in 2 4 6; do
  cp "experiments/gen_batch$b.py" "$BUNDLE/experiments/"
done
cp experiments/__init__.py "$BUNDLE/experiments/"

# Reference submission (the file we submitted to the leaderboard)
cp outputs/agent_b6_hwy_model_up.csv "$BUNDLE/outputs/"

cat > "$BUNDLE/README.txt" << 'TXT'
GridLock 2.0 — source verification bundle
=========================================

LEADERBOARD SUBMISSION
  outputs/agent_b6_hwy_model_up.csv
  Public score: 87.90 (exact: 87.89558)

START HERE
  1. Read APPROACH.md (methodology, features, tools, experiment log).
  2. Place competition train.csv and test.csv in dataset/ (see config.yaml).
  3. pip install -r requirements.txt
  4. Open reproduce.ipynb and Run All cells.
     -> writes outputs/agent_b6_hwy_model_up.csv

Alternative (same result):
  python -m experiments.gen_batch6

No external data. Deterministic (seed=42). The bundled CSV is the submitted file;
re-running reproduce.ipynb reproduces it byte-for-byte.
TXT

( cd "$BUNDLE" && zip -r "../$BUNDLE.zip" . >/dev/null )
echo "Built $BUNDLE.zip"
echo "Contents:"
( cd "$BUNDLE" && find . -type f | sort )
