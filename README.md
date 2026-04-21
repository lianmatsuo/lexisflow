# LexisFlow: Synthetic MIMIC-III Data Generation

Generate synthetic ICU patient data using **Forest-Flow** — a state-of-the-art generative model that uses XGBoost instead of neural networks.

## Features

- **🆕 Fully synthetic generation:** Hour-0 model generates initial patient states without real data
- **Autoregressive trajectories:** Generate realistic patient timelines with temporal continuity
- **No GPU required:** Trains on CPUs using XGBoost
- **Handles missing data natively:** XGBoost learns optimal missing value handling
- **Quality evaluation:** KS tests, correlation preservation, TSTR evaluation
- **Patient-level sequence utility:** Multi-task TSTR (mortality, vasopressor, LOS) trains on full synthetic trajectories and tests on real trajectories
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

**Two-Stage Architecture:**
```
┌──────────────┐
│ Hour-0 Model │ → Demographics + Hour-0 vitals (IID)
│ (New!)       │
└──────┬───────┘
       ↓
┌─────────────────┐
│ Forest-Flow     │ → Hours 1, 2, 3, ..., N (Autoregressive)
│                 │
└─────────────────┘
```

**Benefits:**
- ✅ Fully synthetic (privacy-preserving)
- ✅ Unlimited generation capacity
- ✅ No real data required as input
- ✅ Controllable demographics (future: conditional generation)

See [docs/SWEEP_ARCHITECTURE.md](docs/SWEEP_ARCHITECTURE.md) for complete details.

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

**🆕 Unified Pipeline Command:**
`scripts/run_sweep.py` is the single public entrypoint. It automatically runs
prepare scripts, preprocessor fitting, and then the sweep.

Configuration defaults are centralised in
`src/lexisflow/config/datasets.py`:
- `DatasetConfig`: dataset-specific paths and artifact locations
- `SweepDefaults`: sweep grids and row/sample/privacy budgets
- `SplitConfig`: patient-level split fractions and split seed
- `sweep_profiles`: profile presets (`full`, `smoke`)

CLI precedence is: explicit flag override > selected profile defaults > dataset defaults.

The sweep still trains **both models** with identical hyperparameters:
- Each `(nt, n_noise)` combination trains hour-0 IID + autoregressive models
- Ensures fair comparison across hyperparameter settings
- Mortality utility is evaluated with a patient-level sequence TSTR model over autoregressive trajectories
- Each sweep cell runs TSTR three times with trajectory sampling seeds `42`, `11`, and `50`; reported key utility/quality/privacy metrics include mean, seed-level standard deviation, and 95% confidence-interval half-widths
- Hour-0 models saved to `artifacts/sweep/hour0_nt{nt}_noise{n_noise}.pkl`
- Select best configuration from sweep results and copy to `artifacts/hour0_forest_flow.pkl`

### 5. Public Reproducibility Example (no MIMIC access required)

MIMIC-III requires credentialed PhysioNet access. For reviewers without it,
`scripts/challenge2012/` runs the full pipeline on the **PhysioNet Challenge
2012** ICU benchmark (ODC-BY, open access) using the same column schema so
all TSTR / quality / temporal / privacy metrics work unchanged. See
[scripts/challenge2012/README.md](scripts/challenge2012/README.md).

```bash
uv run python scripts/run_sweep.py --dataset challenge2012
```

## Documentation

📚 **Key Documentation:**

| Document | Description |
|----------|-------------|
| [PROJECT_ARCHITECTURE.md](docs/architecture/PROJECT_ARCHITECTURE.md) | Comprehensive project overview & critical analysis |
| [scripts/WORKFLOW.md](scripts/WORKFLOW.md) | Complete workflow guide & script usage |
| [docs/SWEEP_ARCHITECTURE.md](docs/SWEEP_ARCHITECTURE.md) | Hyperparameter sweep details & TSTR |
| [scripts/challenge2012/README.md](scripts/challenge2012/README.md) | Public reproducibility example (Challenge 2012) |
| [report_latex/](report_latex/) | LaTeX thesis |

## Code Quality Improvements

Recent complexity reduction efforts have significantly improved code maintainability:

### ✅ Completed Simplifications
1. **Unified data format:** Single CSV format (removed Parquet/memmap code paths)
2. **Centralized preprocessor:** Single canonical location (`artifacts/preprocessor_full.pkl`)
3. **Flattened column names:** Tuple columns like `('Heart Rate', 'mean')` → string `'Heart_Rate_mean'`
4. **Centralized feature detection:** New `feature_utils.py` module eliminates code duplication
5. **Removed legacy files:** Cleaned up outdated code and documentation

### Feature Utilities Module
The `src/lexisflow/data/feature_utils.py` module provides centralized utilities:
- `flatten_column_names()` - Convert tuple columns to strings
- `is_lagged()` - Detect lagged features
- `is_binary_feature()` - Identify binary 0/1 features
- `identify_feature_types()` - Classify numeric/binary/categorical columns
- `KNOWN_BINARY_FEATURES` - Clinical binary features from domain knowledge

**Benefits:**
- 30-40% less code to maintain
- Consistent feature detection across scripts
- Easier testing and onboarding
- Fewer edge cases and bugs

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
│   ├── models/                  # Forest-Flow, HS3F, CTGAN, sampling
│   │   ├── forest_flow.py       # Core ForestFlow model
│   │   ├── hs3f.py              # HS3F model
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
│   ├── run_sweep.py                  # Public orchestrator (--dataset, --profile, --reset)
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

**🎉 Clean Import Structure:**
```python
# Professional, modular imports
from lexisflow.data.transformers import TabularPreprocessor
from lexisflow.data.autoregressive import prepare_autoregressive_data
from lexisflow.models.forest_flow import ForestFlow
from lexisflow.models.sampling import sample_trajectory
from lexisflow.evaluation.quality_metrics import compute_quality_metrics
from lexisflow.mortality.model import MortalityClassifier
```

## Key Parameters

| Parameter | Quick Test | Production | Description |
|-----------|------------|------------|-------------|
| `nt` | 10 | 50 | Flow time steps |
| `n_noise` | 10 | 100 | Noise samples (most critical!) |
| `max_rows` | 5000 | null | Training data size |

## References

- **Paper:** [Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees](https://arxiv.org/abs/2309.09968)
- **MIMIC-III:** [PhysioNet](https://physionet.org/content/mimiciii/)

## License

This project uses MIMIC-III data which requires credentialing through PhysioNet.
