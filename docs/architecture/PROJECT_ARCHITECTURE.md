# LexisFlow Project Architecture

Implementation-only architecture summary for the current repository state.

## Scope

LexisFlow implements a two-stage synthetic trajectory pipeline with sweep-based
evaluation:

1. Hour-0 IID generator (initial patient state)
2. Autoregressive transition generator (subsequent timesteps)
3. Integrated evaluation (utility, quality, privacy, trajectory coherence)
4. Dataset-aware sweep orchestration (`mimic`, `challenge2012`)

## Top-Level Structure

- `scripts/`
  - public orchestrator: `scripts/run_sweep.py`
  - dataset drivers: `scripts/mimic/*.py`, `scripts/challenge2012/*.py`
  - analysis: `scripts/common/analyze_sweep.py`
- `src/lexisflow/`
  - `config/` dataset defaults and paths
  - `data/` preprocessing helpers and feature utilities
  - `models/` model implementations and trajectory sampling
  - `evaluation/` metrics and TSTR framework
  - `sweep/` reusable sweep internals (cache, schema, training, evaluation)
- `data/` prepared CSV datasets (not generally committed)
- `artifacts/` preprocessors, cache, and model artifacts
- `results/` sweep CSV outputs and generated plots

## Pipeline Flow

Public command:

```bash
uv run python scripts/run_sweep.py --dataset <mimic|challenge2012>
```

Execution order:

1. `prepare_hour0.py`
2. `prepare_autoregressive.py`
3. `fit_hour0_preprocessor.py`
4. `fit_autoregressive_preprocessor.py`
5. `run_sweep.py`

This flow is implemented in `scripts/run_sweep.py` and parameterized by
`src/lexisflow/config/datasets.py`.

## Dataset Configuration Model

`src/lexisflow/config/datasets.py` defines:

- `DatasetConfig`: script paths, data paths, artifact paths, results paths
- `SweepDefaults`: `nt_values`, `noise_values`, row/sample budgets
- `SplitConfig`: patient-level test/holdout split fractions and seed
- `sweep_profiles`: currently `full` and `smoke`

## Sweep Runtime Design

Core sweep logic is shared under `src/lexisflow/sweep/`.

Per `(nt, n_noise)` cell:

1. train hour-0 model (`train_hour0`)
2. train autoregressive model (`train_autoregressive`)
3. generate synthetic trajectories for configured seeds
4. run TSTR + quality + privacy + trajectory metrics
5. aggregate seed-level mean/std/CI95
6. append normalized row to results CSV

Key modules:

- `sweep/training.py`
- `sweep/generation.py`
- `sweep/evaluation.py`
- `sweep/schema.py`
- `sweep/metrics.py`
- `sweep/cache.py`

## Model Layer

Implemented models in `src/lexisflow/models/`:

- `forest_flow.py`
- `hs3f.py`
- `ctgan_adapter.py`
- `sampling.py` (trajectory rollout)
- `iterator.py` (streaming iterator)

Model selection is exposed in sweep drivers via `--model-type
{forest-flow,hs3f,ctgan}`.

## Evaluation Layer

Implemented in `src/lexisflow/evaluation/`:

- `tstr_framework.py` (patient-level mortality and LOS utility)
- `quality_metrics.py` (KS, correlation Frobenius, range violations)
- `privacy_metrics.py` (DCR and membership-inference diagnostics)
- `trajectory_metrics.py` (autocorrelation distance, stay-length KS,
  transition smoothness, temporal drift)

## Results and Artifacts

MIMIC:

- results CSV: `results/sweep_results.csv`
- artifacts: `artifacts/sweep/`

Challenge 2012:

- results CSV: `results/challenge2012_sweep_results.csv`
- artifacts: `artifacts/challenge2012/sweep/`

Plot generation:

- `scripts/common/analyze_sweep.py`

## What This Document Does Not Cover

- historical design proposals that are no longer in code
- roadmap/recommendation sections
- claims about unimplemented components

For sweep operator details, see `docs/SWEEP_ARCHITECTURE.md`.
