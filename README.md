# Synth-Gen: Synthetic MIMIC-III Data Generation

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

### 2. Run Complete Workflow

```bash
# Step 1: Preprocess data into autoregressive format
uv run python scripts/prepare_autoregressive.py

# Step 1b: Fit preprocessor on FULL dataset (CRITICAL!)
# Ensures correct min/max ranges and captures all categorical values
uv run python scripts/fit_autoregressive_preprocessor.py

# Step 2: Train model (uses pre-fitted preprocessor)
uv run python scripts/train_autoregressive.py

# Optional: quick/scalable overrides for local hardware
# - use fewer rows for fast iteration
# - tune nt / n_noise / n_jobs without editing code
uv run python scripts/train_autoregressive.py \
  --max-rows 5000 \
  --nt 10 \
  --n-noise 10 \
  --n-jobs 8

# Step 3: Generate synthetic data
uv run python scripts/generate.py

# Step 4: Evaluate privacy risk (DCR + membership inference)
uv run python -m synth_gen.evaluation.privacy_metrics \
  --real-data data/processed/flat_table.csv \
  --synthetic-data results/synthetic_patients.csv \
  --output-path results/privacy_metrics.json
```

**Outputs:**
- `data/processed/autoregressive_data.csv` - Preprocessed data
- `artifacts/preprocessor_full.pkl` - Pre-fitted preprocessor (on full data)
- `artifacts/forest_flow_model.pkl` - Trained model
- `results/synthetic_patients.csv` - Generated data
- `results/quality_metrics.txt` - Quality report
- `results/privacy_metrics.json` - Privacy risk report

**⚠️ Why Step 1b matters**: Fitting the preprocessor on a sample will learn incorrect min/max ranges and miss rare categorical values, degrading synthetic data quality. Step 1b runs once on the full dataset to ensure proper scaling.

### 3. 🆕 Fully Synthetic Generation (Hour-0 Model)

Generate patients without requiring real data as input:

```bash
# Step 0: Prepare and train Hour-0 model
uv run python scripts/prepare_hour0.py       # Extract hour-0 states
uv run python scripts/fit_hour0_preprocessor.py  # Fit preprocessor
uv run python scripts/train_hour0.py       # Train IID model

# Then run normal workflow (steps 1, 1b, 2 from above)

# Step 3: Generate fully synthetic patients (no real data needed!)
uv run python scripts/generate.py --use-hour0 --n-patients 100 --n-timesteps 48
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
# Trains BOTH hour-0 and autoregressive models with each hyperparameter combination
uv run python scripts/run_sweep.py

# Visualize results
uv run python scripts/analyze_sweep.py
```

**🆕 Unified Training:**
The sweep automatically trains **both models** with identical hyperparameters:
- Each `(nt, n_noise)` combination trains hour-0 IID + autoregressive models
- Ensures fair comparison across hyperparameter settings
- Mortality utility is evaluated with a patient-level sequence TSTR model over autoregressive trajectories
- Each sweep cell runs TSTR three times with trajectory sampling seeds `42`, `11`, and `50`; reported key utility/quality/privacy metrics include mean, seed-level standard deviation, and 95% confidence-interval half-widths
- Hour-0 models saved to `artifacts/sweep/hour0_nt{nt}_noise{n_noise}.pkl`
- Select best configuration from sweep results and copy to `artifacts/hour0_forest_flow.pkl`

## Documentation

📚 **Key Documentation:**

| Document | Description |
|----------|-------------|
| [PROJECT_ARCHITECTURE.md](docs/architecture/PROJECT_ARCHITECTURE.md) | Comprehensive project overview & critical analysis |
| [scripts/WORKFLOW.md](scripts/WORKFLOW.md) | Complete workflow guide & script usage |
| [docs/SWEEP_ARCHITECTURE.md](docs/SWEEP_ARCHITECTURE.md) | Hyperparameter sweep details & TSTR |
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
The `src/synth_gen/data/feature_utils.py` module provides centralized utilities:
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
# Run all tests (tests are colocated with code in src/synth_gen/)
uv run pytest

# Run specific package tests
uv run pytest src/synth_gen/data/tests/
uv run pytest src/synth_gen/models/tests/
uv run pytest src/synth_gen/evaluation/tests/

# Run specific test file
uv run pytest src/synth_gen/data/tests/test_transformers.py

# Run with coverage
uv run pytest --cov=synth_gen --cov-report=html
```

## Project Structure

```
synth-gen/
├── src/synth_gen/               # Core library modules
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
│   └── sweep/                   # Sweep orchestration helpers
│       ├── training.py          # Generator factory + trainers
│       ├── generation.py        # Autoregressive sampling
│       ├── evaluation.py        # Per-seed TSTR evaluation
│       ├── data_prep.py         # Preprocessor + cache loading
│       └── tests/               # Unit tests
├── scripts/                     # Workflow scripts
│   ├── prepare_hour0.py              # Extract hour-0 states
│   ├── fit_hour0_preprocessor.py     # Fit hour-0 preprocessor
│   ├── train_hour0.py                # Train IID model
│   ├── prepare_autoregressive.py     # Create autoregressive format
│   ├── fit_autoregressive_preprocessor.py   # Fit preprocessor on full data
│   ├── train_autoregressive.py       # Train autoregressive model
│   ├── generate.py                   # Generate synthetic patients
│   ├── run_sweep.py                  # Hyperparameter sweep
│   └── analyze_sweep.py              # Visualize sweep results
├── data/                        # Data files (gitignored)
│   └── processed/               # Preprocessed CSVs
├── artifacts/                   # Trained model artifacts (gitignored)
│   └── sweep/                   # Per-cell sweep models
├── results/                     # Generated outputs
├── configs/                     # YAML configurations
├── report_latex/                # LaTeX thesis
└── docs/                        # Documentation
```

**🎉 Clean Import Structure:**
```python
# Professional, modular imports
from synth_gen.data.transformers import TabularPreprocessor
from synth_gen.data.autoregressive import prepare_autoregressive_data
from synth_gen.models.forest_flow import ForestFlow
from synth_gen.models.sampling import sample_trajectory
from synth_gen.evaluation.quality_metrics import compute_quality_metrics
from synth_gen.mortality.model import MortalityClassifier
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
