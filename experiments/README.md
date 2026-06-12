# Experiments

Batch generators produce submission CSVs into `outputs/`. Diagnostics validate assumptions locally.

## Champion (run this)

```bash
python -m experiments.gen_batch6
```

Writes `outputs/agent_b6_hwy_model_up.csv` (LB score **87.90**).

## Batch history

| Script | What it tested |
|--------|----------------|
| `gen_batch1.py` | Lookup vs hybrid blends |
| `gen_batch2.py` | Per-cell slot lookup + model blend |
| `gen_batch3.py` | Three-way blend (raw + model + corr) |
| `gen_batch4.py` | corr2 + four-way blend → 87.65 |
| `gen_batch5.py` | Spatio-temporal corr3 (did not beat champion) |
| `gen_batch6.py` | **Per-RoadType weights → 87.90 champion** |
| `gen_batch7.py` | Weight refinement (plateaued) |
| `gen_batch8.py` | Multiplicative day-factor (hurt daytime) |

## Diagnostics

| Script | Purpose |
|--------|---------|
| `diag02_lookup_gap.py` | Lookup coverage analysis |
| `diag03_leak_hunt.py` | Leakage checks |
| `diag04_index.py` | Index ordering leak test |
| `diag05_newfeats.py` | New feature probes |
| `diag06_feature_logic.py` | Day-factor / profile tests |
| `exp01_day_shift.py` | Additive day-49 scaling sweep |
