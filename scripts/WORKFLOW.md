# Script Workflow Guide

The public pipeline command is now a single entrypoint:

```bash
uv run python scripts/run_sweep.py --dataset mimic
```

Use `--dataset challenge2012` for the public PhysioNet benchmark and `--reset`
to clear generated data/preprocessors/cache/results before rerunning.
Use `--profile smoke` for a faster, reduced-budget validation run.
Use `--n-jobs` to control CPU parallelism in the sweep stage.

All other scripts in `scripts/mimic/`, `scripts/challenge2012/`, and
`scripts/common/` are internal pipeline steps called by `scripts/run_sweep.py`.

## Scripts

| Script | Purpose |
|---|---|
| `run_sweep.py` | Single public command. Orchestrates prepare + fit + sweep for `mimic` or `challenge2012`. |
| `mimic/*.py` | MIMIC-specific internal pipeline scripts called by `run_sweep.py --dataset mimic`. |
| `challenge2012/*.py` | Challenge 2012 internal scripts called by `run_sweep.py --dataset challenge2012`. |
| `common/analyze_sweep.py` | Heatmaps, line plots, and extrapolation fits from sweep results CSVs. |
| `common/evaluate_hour0_models.py` | Post-sweep hour-0-only fidelity diagnostics (no rollout/TSTR), with fixed real sample and multi-seed synthetic averaging. |
| `common/analyze_hour0_diagnostics.py` | Heatmaps/line charts/summary for hour-0 diagnostics CSVs. |
| `common/analyze_column_coverage.py` | Streaming non-null / distribution stats per column. |
| `common/analyze_column_removal.py` | Build removal recommendations from coverage stats. |

## End-to-end pipeline

### Sweep (single command)

```bash
uv run python scripts/run_sweep.py --dataset mimic
uv run python scripts/run_sweep.py --dataset challenge2012
uv run python scripts/run_sweep.py --dataset mimic --profile smoke
uv run python scripts/run_sweep.py --dataset challenge2012 --profile smoke
uv run python scripts/run_sweep.py --dataset mimic --n-jobs 8
uv run python scripts/run_sweep.py --dataset mimic --reset

uv run python scripts/common/analyze_sweep.py
uv run python scripts/common/analyze_sweep.py --dataset challenge2012
uv run python scripts/common/analyze_sweep.py --results-path results/challenge2012_sweep_results.csv
```

The sweep is resumable: existing rows in the dataset-specific results CSV are
skipped on rerun. Each cell runs TSTR three times with trajectory seeds
`{42, 11, 50}` and reports mean / seed-level stdev / 95% CI half-width.

### Hour-0 diagnostics (post-sweep)

Run these **after** sweep completes, because they read saved hour-0 model
artifacts (`hour0_nt*_noise*.pkl`) from `artifacts/.../sweep/`.

```bash
# MIMIC
uv run python scripts/common/evaluate_hour0_models.py \
  --dataset mimic \
  --models-dir artifacts/sweep \
  --min-nt 1 --min-nnoise 1 \
  --n-synth 5000 --n-real 5000 \
  --seed 42 --synth-seeds 42,11,50 \
  --output-csv results/hour0_diagnostics.csv

uv run python scripts/common/analyze_hour0_diagnostics.py \
  --results-path results/hour0_diagnostics.csv \
  --output-dir results/hour0_diagnostics_plots

# Challenge 2012
uv run python scripts/common/evaluate_hour0_models.py \
  --dataset challenge2012 \
  --models-dir artifacts/challenge2012/sweep \
  --min-nt 1 --min-nnoise 1 \
  --n-synth 5000 --n-real 5000 \
  --seed 42 --synth-seeds 42,11,50 \
  --output-csv results/challenge2012_hour0_diagnostics.csv

uv run python scripts/common/analyze_hour0_diagnostics.py \
  --results-path results/challenge2012_hour0_diagnostics.csv \
  --output-dir results/challenge2012_hour0_diagnostics_plots

# Optional bounded grid (example: nt <= 13, n_noise <= 13)
uv run python scripts/common/evaluate_hour0_models.py \
  --dataset challenge2012 \
  --models-dir artifacts/challenge2012/sweep \
  --min-nt 1 --min-nnoise 1 --max-nt 13 --max-nnoise 13 \
  --n-synth 5000 --n-real 5000 \
  --seed 42 --synth-seeds 42,11,50 \
  --output-csv results/challenge2012_hour0_diagnostics_nt13_noise13.csv
```

## Artifact flow

```
scripts/run_sweep.py --dataset <dataset>
    → scripts/<dataset>/prepare_hour0.py
    → scripts/<dataset>/prepare_autoregressive.py
    → scripts/<dataset>/fit_hour0_preprocessor.py
    → scripts/<dataset>/fit_autoregressive_preprocessor.py
    → scripts/<dataset>/run_sweep.py
    → results/<dataset-specific>_sweep_results.csv
```

Quality, privacy, and TSTR evaluation live in `src/lexisflow/evaluation/` and
are invoked by the dataset sweep drivers.

## Internal tuning

Default paths and sweep/split budgets live in
`src/lexisflow/config/datasets.py`:

- `DatasetConfig`: dataset-specific script/data/artifact/result paths
- `SweepDefaults`: `nt` grid, `n_noise` grid, row/sample budgets, privacy cap
- `SplitConfig`: patient-level test/holdout split fractions and shuffle seed
- `sweep_profiles`: `full` and `smoke` presets per dataset

Fine-grained sweep overrides (for `nt`, `n_noise`, row budgets, parallelism,
and privacy toggles) are still available directly on the dataset-specific sweep
drivers:

```bash
uv run python scripts/mimic/run_sweep.py --profile full --nt-values 1,3,5 --noise-values 1,3,5
uv run python scripts/challenge2012/run_sweep.py --profile smoke --train-rows 30000 --synth-samples 15000
```

CLI precedence is: explicit flags > profile defaults > dataset defaults.
The top-level `scripts/run_sweep.py --n-jobs <int>` forwards the worker override
to the dataset-specific sweep driver.
