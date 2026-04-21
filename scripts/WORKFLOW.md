# Script Workflow Guide

Scripts are grouped by verb: `prepare_*` (build CSVs), `fit_*` (fit preprocessors), `train_*` (train generators), `generate.py` (sample synthetic data), `run_*` (multi-cell experiments), `analyze_*` (read artifacts, emit plots/tables).

## Scripts

| Script | Purpose |
|---|---|
| `prepare_autoregressive.py` | Build autoregressive training CSV with lag-1 features from the flat MIMIC table. |
| `fit_autoregressive_preprocessor.py` | Fit `TabularPreprocessor` on the full autoregressive CSV (captures rare categories and true min/max). |
| `train_autoregressive.py` | Train HS3F or Forest-Flow on the autoregressive matrix. |
| `prepare_hour0.py` | Extract hour-0 rows (demographics + initial vitals). |
| `fit_hour0_preprocessor.py` | Fit the hour-0 preprocessor. |
| `train_hour0.py` | Train the IID hour-0 generator. |
| `generate.py` | Sample synthetic trajectories (`--use-hour0` for fully synthetic). |
| `run_sweep.py` | Hyperparameter sweep over `(nt, n_noise)` with matched seeds, TSTR + quality + privacy. |
| `run_backbone_comparison.py` | Matched HS3F vs Forest-Flow vs CTGAN at a single cell. |
| `analyze_sweep.py` | Heatmaps, line plots, and extrapolation fits from `sweep_results.csv`. |
| `analyze_column_coverage.py` | Streaming non-null / distribution stats per column. |
| `analyze_column_removal.py` | Build removal recommendations from coverage stats. |

## End-to-end pipeline

### Autoregressive-only (real initial conditions)

```bash
uv run python scripts/prepare_autoregressive.py
uv run python scripts/fit_autoregressive_preprocessor.py   # one-off, fits on FULL dataset
uv run python scripts/train_autoregressive.py              # default backbone: hs3f
uv run python scripts/generate.py
```

### Fully synthetic (hour-0 + autoregressive)

```bash
# Hour-0 branch
uv run python scripts/prepare_hour0.py
uv run python scripts/fit_hour0_preprocessor.py
uv run python scripts/train_hour0.py

# Autoregressive branch (reuses preprocessor from above)
uv run python scripts/prepare_autoregressive.py
uv run python scripts/fit_autoregressive_preprocessor.py
uv run python scripts/train_autoregressive.py

# Stitch together
uv run python scripts/generate.py --use-hour0 --n-patients 100 --n-timesteps 48
```

### Sweep

```bash
uv run python scripts/run_sweep.py \
    --train-rows 50000 \
    --nt-values 1,5,10,20 \
    --noise-values 1,3,5

uv run python scripts/analyze_sweep.py
```

The sweep is resumable: existing rows in `results/sweep_results.csv` are skipped on rerun. Each cell runs TSTR three times with trajectory seeds `{42, 11, 50}` and reports mean / seed-level stdev / 95% CI half-width.

## Artifact flow

```
data/processed/flat_table.csv
    → prepare_autoregressive.py → data/processed/autoregressive_data.csv
    → fit_autoregressive_preprocessor.py → artifacts/preprocessor_full.pkl
    → train_autoregressive.py → artifacts/forest_flow_model.pkl, artifacts/preprocessor.pkl
    → generate.py → results/synthetic_patients.csv

data/processed/flat_table.csv
    → prepare_hour0.py → data/processed/hour0_data.csv
    → fit_hour0_preprocessor.py → artifacts/hour0_preprocessor.pkl
    → train_hour0.py → artifacts/hour0_forest_flow.pkl
```

Quality, privacy, and TSTR evaluation live in `src/lexisflow/evaluation/` and are invoked by `run_sweep.py` per cell; call the helpers directly from a notebook or REPL for ad-hoc inspection of a single run.

## Common overrides

```bash
# Quick-iteration training
uv run python scripts/train_autoregressive.py --max-rows 5000 --nt 10 --n-noise 10 --n-jobs 8

# Production config
uv run python scripts/train_autoregressive.py --nt 50 --n-noise 100 --max-rows 0
```

Key knobs: `nt` (flow time levels), `n_noise` (noise multipliers per row during training — the dominant cost driver), `max-rows` (0 = all rows), `n-jobs` (XGBoost parallelism).
