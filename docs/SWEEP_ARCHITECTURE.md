# Sweep Architecture

This page describes only the implemented sweep pipeline.

## Public Entrypoint

Run the full pipeline for one dataset:

```bash
uv run python scripts/run_sweep.py --dataset mimic
uv run python scripts/run_sweep.py --dataset challenge2012
```

Optional flags:

```bash
uv run python scripts/run_sweep.py --dataset mimic --profile smoke
uv run python scripts/run_sweep.py --dataset mimic --reset
```

`scripts/run_sweep.py` orchestrates:

1. `scripts/<dataset>/prepare_hour0.py`
2. `scripts/<dataset>/prepare_autoregressive.py`
3. `scripts/<dataset>/fit_hour0_preprocessor.py`
4. `scripts/<dataset>/fit_autoregressive_preprocessor.py`
5. `scripts/<dataset>/run_sweep.py`

Implementation: `scripts/run_sweep.py`.

## Dataset Defaults

Defaults are centralized in `src/lexisflow/config/datasets.py`:

- path configuration (`DatasetConfig`)
- sweep grids and row/sample budgets (`SweepDefaults`)
- patient split fractions (`SplitConfig`)
- profile presets (`full`, `smoke`)

Current dataset presets:

- `mimic`
- `challenge2012`

## Per-Cell Sweep Lifecycle

Implemented in `scripts/mimic/run_sweep.py`,
`scripts/challenge2012/run_sweep.py`, and `src/lexisflow/sweep/`.

For each `(nt, n_noise)`:

1. train hour-0 IID model
2. train autoregressive model
3. generate synthetic trajectories for each seed in
   `TSTR_TRAJECTORY_SAMPLING_SEEDS` (currently `(42, 11, 50)`)
4. evaluate utility/quality/privacy/trajectory metrics per seed
5. aggregate mean/std/CI95
6. append one schema-normalized row to results CSV

## Resumability and Schema

- Existing `(nt, n_noise)` rows are skipped on rerun.
- Result rows are normalized to canonical schema before append.
- Failed cells are written as explicit error rows.

Implementation:

- `src/lexisflow/sweep/schema.py`
- `src/lexisflow/sweep/metrics.py`
- `src/lexisflow/sweep/evaluation.py`
- `src/lexisflow/sweep/data_prep.py`

## Cache Behavior

Transformed autoregressive arrays are cached and reused when signatures match.

Implementation: `src/lexisflow/sweep/cache.py`.

## Outputs

MIMIC:

- results: `results/sweep_results.csv`
- artifacts: `artifacts/sweep/`

Challenge 2012:

- results: `results/challenge2012_sweep_results.csv`
- artifacts: `artifacts/challenge2012/sweep/`

Common plots:

- `scripts/common/analyze_sweep.py`
- output under `results/sweep_plots/` for MIMIC defaults
