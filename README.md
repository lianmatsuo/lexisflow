# LexisFlow: Synthetic MIMIC-III Data Generation

Generate synthetic ICU patient trajectories with **flow-matched gradient-boosted generators**. The default sweep backbone is **HS3F** (sequential heterogeneous routing); **ForestFlow** (all-continuous flow) and a **CTGAN** baseline are also implemented and can be selected on the dataset sweep drivers or compared in a dedicated matched-cell script.

## Features

- **Fully synthetic generation:** Hour-0 model generates initial patient states without real data
- **Autoregressive trajectories:** Generate realistic patient timelines with temporal continuity
- **CPU-friendly core generators:** HS3F and ForestFlow train via XGBoost; the optional CTGAN baseline uses PyTorch and may use a GPU if available
- **Missing data:** Gaps are handled **up front** in preparation, not only inside the booster: hour-0 scripts impute numerics with the **median** and categoricals with the **mode** or **`Unknown`**; autoregressive tables build lag-1 condition columns with a **sentinel fill** (default **`-1.0`**) when no history exists (`prepare_autoregressive_data`). The public Challenge 2012 autoregressive prep also **forward-/backward-fills** within patient, then **global medians**, before lagging. Training matrices are therefore mostly **explicitly filled**; XGBoost’s native NA splits are not the main story.
- **Quality evaluation:** KS tests, correlation preservation, TSTR evaluation
- **Patient-level sequence utility:** Multi-task TSTR on **mortality and length-of-stay (LOS) only** — sequence classifiers train on full synthetic trajectories and test on real trajectories (no additional utility heads in the sweep evaluation bundle)
- **Privacy evaluation:** DCR diagnostics + membership inference risk
- **Modular architecture:** Clean, professional package structure

## Quick Start

### 1. Install

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Run Complete Workflow (Single Command)

```bash
# MIMIC pipeline: prepare data + fit preprocessors + run full sweep
uv run python scripts/run_sweep.py --dataset mimic

# Fast smoke preset for quick validation
uv run python scripts/run_sweep.py --dataset mimic --profile smoke

# Override CPU parallelism for the sweep stage
uv run python scripts/run_sweep.py --dataset mimic --n-jobs 8

# Optional: clear generated data/preprocessors/cache/results, then rerun
uv run python scripts/run_sweep.py --dataset mimic --reset
```

### 3. Public Reproducibility Workflow (Challenge 2012)

```bash
# Public benchmark pipeline: same command surface, different dataset preset
uv run python scripts/run_sweep.py --dataset challenge2012

# Public benchmark smoke run
uv run python scripts/run_sweep.py --dataset challenge2012 --profile smoke
```

**Two-Stage Architecture (per backbone):**
```
┌──────────────┐
│ Hour-0 Model │ → Demographics + Hour-0 vitals (IID)
│              │
└──────┬───────┘
       ↓
┌──────────────────────────────┐
│ Autoregressive generator     │ → Hours 1, 2, 3, …, N
│ (default: HS3F; or FF/CTGAN) │
└──────────────────────────────┘
```

See [docs/SWEEP_ARCHITECTURE.md](docs/SWEEP_ARCHITECTURE.md) for the implemented sweep lifecycle and outputs.

### 4. Advanced: Hyperparameter Sweep

```bash
# Full pipeline + sweep (MIMIC defaults)
uv run python scripts/run_sweep.py --dataset mimic

# Public benchmark pipeline + sweep (Challenge 2012)
uv run python scripts/run_sweep.py --dataset challenge2012

# Optional: clear generated data/preprocessors/cache/results before rerun
uv run python scripts/run_sweep.py --dataset mimic --reset

# Visualize results
uv run python scripts/common/analyze_sweep.py
uv run python scripts/common/analyze_sweep.py --dataset challenge2012
```

**Unified Pipeline Command:**
`scripts/run_sweep.py` is the single public entrypoint. It automatically runs
prepare scripts, preprocessor fitting, and then the sweep.

Configuration defaults are centralised in
`src/lexisflow/config/datasets.py`:
- `DatasetConfig`: dataset-specific paths and artifact locations
- `SweepDefaults`: sweep grids and row/sample/privacy budgets
- `SplitConfig`: patient-level split fractions and split seed
- `sweep_profiles`: profile presets (`full`, `smoke`)

CLI precedence is: explicit flag override > selected profile defaults > dataset defaults.
`--n-jobs` can be set on the top-level command to override parallel worker count
for the dataset sweep step on machines with different CPU core counts.

For **one chosen generator backbone** (default **`hs3f`**), each sweep cell trains **two stages** with the same `(nt, n_noise)` hyperparameters:
- **Hour-0 (IID)** and **autoregressive** checkpoints are written under `artifacts/sweep/` (for example `hour0_nt{nt}_noise{n_noise}.pkl` and `autoregressive_nt{nt}_noise{n_noise}.pkl`).
- A single sweep invocation uses **one** `--model-type` (default `hs3f`). The hyperparameter grid characterises that backbone only; switching to ForestFlow or CTGAN means **rerunning** the driver with the other `--model-type` (see below)—not a simultaneous multi-backbone matrix.
- **Matched comparison** of HS3F vs ForestFlow vs CTGAN at a **single shared cell** (same rows, synthetic budget, trajectory seeds) is a separate orchestrator: `scripts/mimic/run_backbone_comparison.py`.
- **Utility:** patient-level sequence TSTR covers **mortality and LOS only** (see `src/lexisflow/sweep/evaluation.py`).
- Each sweep cell aggregates TSTR over three trajectory sampling seeds (`42`, `11`, `50`); reported metrics include means, seed-level standard deviations, and 95% CI half-widths where applicable.

The top-level command `scripts/run_sweep.py` forwards **`--profile`** and **`--n-jobs`** to the dataset sweep only. To sweep **ForestFlow** or **CTGAN**, run the dataset driver directly after data/preprocessor steps, for example:
`uv run python scripts/mimic/run_sweep.py --model-type forest-flow` or
`uv run python scripts/challenge2012/run_sweep.py --model-type ctgan` (plus your usual `--profile` / `--n-jobs`).

Some standalone MIMIC scripts still use the legacy filename `artifacts/hour0_forest_flow.pkl`; promote checkpoints from `artifacts/sweep/` if your workflow expects that path (see `scripts/WORKFLOW.md`).

### 5. Public Reproducibility Example (no MIMIC access required)

MIMIC-III requires credentialed PhysioNet access. For reviewers without it,
the project includes a full pipeline on the **PhysioNet Challenge 2012** ICU
benchmark (ODC-BY, open access) using the same column schema so all TSTR /
quality / temporal / privacy metrics work unchanged. See
[docs/challenge2012.md](docs/challenge2012.md).

```bash
uv run python scripts/run_sweep.py --dataset challenge2012
```

## Documentation

📚 **Key Documentation:**

| Document | Description |
|----------|-------------|
| [PROJECT_ARCHITECTURE.md](docs/architecture/PROJECT_ARCHITECTURE.md) | Implementation-only system architecture |
| [scripts/WORKFLOW.md](scripts/WORKFLOW.md) | Complete workflow guide & script usage |
| [docs/SWEEP_ARCHITECTURE.md](docs/SWEEP_ARCHITECTURE.md) | Sweep execution flow, schema, cache, and outputs |
| [docs/challenge2012.md](docs/challenge2012.md) | Public reproducibility guide (Challenge 2012) |
| [report_latex/](report_latex/) | LaTeX thesis |

## Testing

Run unit tests to verify core functionality:

```bash
# Run all tests (tests are colocated with code in src/lexisflow/)
uv run pytest

# Run specific package tests
uv run pytest src/lexisflow/data/tests/
uv run pytest src/lexisflow/models/tests/
uv run pytest src/lexisflow/evaluation/tests/

# Run specific test file
uv run pytest src/lexisflow/data/tests/test_transformers.py

# Run with coverage
uv run pytest --cov=lexisflow --cov-report=html
```

## Project Structure

```
lexisflow/
├── src/lexisflow/               # Core library modules
│   ├── data/                    # Data loading, preprocessing, autoregressive prep
│   │   ├── transformers.py      # TabularPreprocessor
│   │   ├── autoregressive.py    # Lag feature creation
│   │   ├── feature_utils.py     # Centralized feature detection utilities
│   │   ├── loaders.py           # Data loading utilities
│   │   └── tests/               # Unit tests
│   ├── models/                  # HS3F, ForestFlow, CTGAN, sampling
│   │   ├── hs3f.py              # HS3F model (default sweep backbone)
│   │   ├── forest_flow.py       # ForestFlow model
│   │   ├── ctgan_adapter.py     # CTGAN baseline
│   │   ├── sampling.py          # Trajectory generation
│   │   ├── iterator.py          # Memory-efficient training
│   │   └── tests/               # Unit tests
│   ├── evaluation/              # Quality, privacy, TSTR metrics
│   │   ├── quality_metrics.py   # KS tests, correlation, violations
│   │   ├── privacy_metrics.py   # DCR, membership inference
│   │   ├── tstr_framework.py    # TSTR evaluation
│   │   └── tests/               # Unit tests
│   ├── config/                  # Dataset presets + sweep/split defaults
│   │   ├── __init__.py
│   │   └── datasets.py          # DatasetConfig, SweepDefaults, SplitConfig
│   └── sweep/                   # Sweep orchestration helpers
│       ├── training.py          # Generator factory + trainers
│       ├── generation.py        # Autoregressive sampling
│       ├── evaluation.py        # Per-seed TSTR evaluation
│       ├── data_prep.py         # Preprocessor + cache loading
│       └── tests/               # Unit tests
├── scripts/                     # Workflow scripts
│   ├── run_sweep.py                  # Public orchestrator (--dataset, --profile, --n-jobs, --reset)
│   ├── mimic/                        # MIMIC internal pipeline scripts
│   ├── challenge2012/                # Challenge 2012 internal pipeline scripts
│   └── common/                       # Shared analysis utilities (e.g., analyze_sweep.py)
├── data/                        # Data files (gitignored)
│   └── processed/               # Preprocessed CSVs
├── artifacts/                   # Trained model artifacts (gitignored)
│   └── sweep/                   # Per-cell sweep models
├── results/                     # Generated outputs
├── report_latex/                # LaTeX thesis
└── docs/                        # Documentation
```

## Key Parameters

| Parameter | Quick Test | Production | Description |
|-----------|------------|------------|-------------|
| `nt` | 10 | 50 | Flow time steps |
| `n_noise` | 10 | 100 | Noise samples (most critical!) |
| `max_rows` | 5000 | null | Training data size |

## References

- **ForestFlow lineage:** [Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees](https://arxiv.org/abs/2309.09968)
- **MIMIC-III:** [PhysioNet](https://physionet.org/content/mimiciii/)

## License

This project uses MIMIC-III data which requires credentialing through PhysioNet.
