# Traffic Demand Prediction

Short-horizon spatiotemporal forecast of normalized traffic `demand` per
`(geohash, 15-min slot)`. Metric: `max(0, 100 * R2)`.

## The one fact that drives everything

The split is a **forward forecast**, not a random shuffle:

- train = day 48 (full) + day 49 morning (00:00-02:00)
- test = day 49 daytime (02:15-13:45)
  Only two days of history exist. Protecting against temporal leakage is the
  whole game. Never random-split. Never fit statistics on data that includes
  the rows you are predicting.

## Pipeline stages (build order)

1.  [done] Data contract + validation .... src/config.py, src/data.py
2.  [next] Backtest split design ......... src/splits.py
3.          EDA (hypothesis-driven) ....... notebooks/01_eda.ipynb
4.          Baselines (naive forecasts) ... src/baselines.py
5.          Feature engineering ........... src/features.py
6.          Models (LGBM/XGB/CatBoost) .... src/models/
7.          Hyperparameter optimization ... src/tune.py
8.          Ensemble / blend .............. src/ensemble.py
9.          Calibration + post-process .... src/postprocess.py
10.       Evaluation (exact metric) ..... src/evaluate.py
11.       End-to-end run ................ run.py

## Run

```bash
pip install -r requirements.txt
python -m src.config     # verify paths resolve
python -m src.data       # validate contract + print data profile
```

## Submission damping experiments

| Factor damp | Live score |
| --- | --- |
| `0.00` (ignore the trend) | `83.63` |
| `0.50` | `79.33` |
| `0.75` (`submission.csv`) | `66.81` |
| `1.00` (trust it fully) | `50.63` |
