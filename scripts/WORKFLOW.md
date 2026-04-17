# Script Workflow Guide

This directory contains both **numbered workflow scripts** (new, recommended) and **legacy scripts** (preserved for compatibility).

## 🆕 New Numbered Workflow (Recommended)

Use these scripts for a clear, step-by-step workflow:

### Main Workflow (Autoregressive Model)
```
01_preprocess_data.py     → Prepare autoregressive data
01b_fit_preprocessor.py   → Fit preprocessor on FULL dataset ⚠️ IMPORTANT
02_train_model.py         → Train generative model (HS3F or Forest-Flow)
03_generate_synthetic.py  → Generate synthetic patients
04_evaluate_quality.py    → Assess data quality
```

### Hour-0 Workflow (NEW! - Fully Synthetic Generation)
```
00_prepare_hour0_data.py       → Extract hour-0 patient states
00b_fit_hour0_preprocessor.py  → Fit hour-0 preprocessor
00c_train_hour0_model.py       → Train hour-0 IID model
05_evaluate_hour0_model.py     → Evaluate hour-0 quality
```

### Complete Workflow (Without Hour-0 Model)

```bash
# Step 1: Preprocess raw data into autoregressive format
uv run python scripts/01_preprocess_data.py

# Step 1b: Fit preprocessor on FULL dataset (CRITICAL!)
# This ensures correct min/max ranges and captures all rare categorical values
uv run python scripts/01b_fit_preprocessor.py

# Step 2: Train generative model (default: HS3F)
# Will automatically use pre-fitted preprocessor from step 1b
uv run python scripts/02_train_model.py

# Step 3: Generate synthetic patient data (using real initial conditions)
uv run python scripts/03_generate_synthetic.py

# Step 4: Evaluate quality metrics
uv run python scripts/04_evaluate_quality.py
```

### Complete Workflow (WITH Hour-0 Model - Fully Synthetic)

```bash
# Phase 1: Prepare and train Hour-0 model
# =========================================

# Step 0: Prepare hour-0 data (demographics + initial vitals)
uv run python scripts/00_prepare_hour0_data.py

# Step 0b: Fit hour-0 preprocessor
uv run python scripts/00b_fit_hour0_preprocessor.py

# Step 0c: Train hour-0 IID model
uv run python scripts/00c_train_hour0_model.py

# Step 0d (optional): Evaluate hour-0 model quality
uv run python scripts/05_evaluate_hour0_model.py

# Phase 2: Prepare and train autoregressive model
# ================================================

# Step 1: Preprocess raw data into autoregressive format
uv run python scripts/01_preprocess_data.py

# Step 1b: Fit preprocessor on FULL dataset
uv run python scripts/01b_fit_preprocessor.py

# Step 2: Train autoregressive Forest-Flow model
uv run python scripts/02_train_model.py

# Phase 3: Generate fully synthetic data
# =======================================

# Step 3: Generate synthetic patients (using hour-0 model)
uv run python scripts/03_generate_synthetic.py --use-hour0 --n-patients 100 --n-timesteps 48

# Step 4: Evaluate quality metrics
uv run python scripts/04_evaluate_quality.py
```

## 🏗️ Two-Stage Generation Architecture

### Original Architecture (Real Initial Conditions)
```
┌─────────────────────────────────────┐
│ Real Patient Data (MIMIC-III)      │
│ • Static features (age, gender)     │
│ • Initial conditions (hr_lag1)      │
└───────────┬─────────────────────────┘
            ↓
    ┌─────────────────┐
    │ Forest-Flow     │ → Hours 1, 2, 3, ..., N
    │ (Autoregressive)│
    └─────────────────┘
```

**Limitation:** Requires real patient data as input (privacy concerns)

### New Architecture (Fully Synthetic with Hour-0 Model)
```
    ┌──────────────┐
    │ Hour-0 Model │ → Demographics + Hour-0 vitals
    │ (IID)        │    (age, gender, hr_t0, bp_t0, ...)
    └──────┬───────┘
           ↓
    ┌─────────────────┐
    │ Forest-Flow     │ → Hours 1, 2, 3, ..., N
    │ (Autoregressive)│
    └─────────────────┘
```

**Advantages:**
- ✅ Fully synthetic (no real data needed)
- ✅ Privacy-preserving
- ✅ Unlimited generation capacity
- ✅ Controlled demographics distribution

### Quick Test (10 minutes)

For quick testing with reduced parameters:

1. Use CLI overrides (no script edits needed):
   ```bash
   uv run python scripts/02_train_model.py \
     --nt 10 \
     --n-noise 10 \
     --max-rows 5000 \
     --n-jobs 8
   ```

2. Run workflow:
   ```bash
   # Preprocess data
   uv run python scripts/01_preprocess_data.py

   # Fit preprocessor on FULL data (only once, ~2-5 min)
   uv run python scripts/01b_fit_preprocessor.py

   # Train on sample (uses pre-fitted preprocessor)
   uv run python scripts/02_train_model.py --nt 10 --n-noise 10 --max-rows 5000

   # Generate and evaluate
   uv run python scripts/03_generate_synthetic.py
   uv run python scripts/04_evaluate_quality.py
   ```

**Note**: Step 1b only needs to run once. After that, you can iterate on training (step 2) with different subsamples while maintaining correct scaling.

### Why Step 1b is Critical

**Problem**: If you fit the preprocessor on a training sample (e.g., 10k rows):
- ❌ Numeric range/statistic estimates are learned from an incomplete sample
- ❌ Rare categorical values are missed (not encoded)
- ❌ Downstream decode/rounding behavior can be less stable on unseen value regions
- ❌ Synthetic data quality is degraded

**Solution**: Fit preprocessor on FULL dataset once (step 1b):
- ✅ Correct min/max ranges from all 2.2M rows
- ✅ All categorical values encoded (even rare ones)
- ✅ Train on any subsample while maintaining correct scaling
- ✅ Better synthetic data quality

### Outputs

```
packages/data/processed/
  └── autoregressive_data.csv   # Preprocessed data (Step 1)

models/
  ├── preprocessor_full.pkl      # Pre-fitted preprocessor (Step 1b) ⭐️
  ├── forest_flow_model.pkl      # Trained model artifact (Step 2)
  └── preprocessor.pkl           # Runtime preprocessor + split metadata (Step 2)

results/
  ├── synthetic_patients.csv     # Generated data (Step 3)
  └── quality_metrics.txt        # Quality report (Step 4)
```

## 📚 Legacy Scripts (Preserved)

These scripts still work but are less organized:

### Original Demo Scripts
- `demo_autoregressive_forest_flow.py` - Complete training & generation demo
- `demo_iid_forest_flow.py` - IID sampling demo (deprecated)

### Specialized Scripts
- `preprocess_autoregressive_data.py` - Data preprocessing (now: `01_*.py`)
- `run_with_config.py` - Config-based training
- `evaluate_models.py` - Model evaluation

### Hyperparameter Sweep
- `run_sweep.py` - Full hyperparameter sweep with TSTR
- `plot_sweep_results.py` - Visualize sweep results

## 🔬 Advanced: Hyperparameter Sweep

For production-quality models, run a hyperparameter sweep to find optimal `nt` and `n_noise`:

```bash
# 1. Run sweep with explicit grid values (trains BOTH models)
uv run python scripts/run_sweep.py \
  --nt-values 1,3,5,7,9 \
  --noise-values 1,3,5,7,9,11,13,15 \
  --train-rows 50000

# 2. Visualize results
uv run python scripts/plot_sweep_results.py

# 3. Check outputs
ls results/sweep_*
# - sweep_results.csv       # Quantitative results (includes both models)
# - sweep_plots/            # Visualizations

ls models/sweep/
# - hour0_nt10_noise10.pkl  # Hour-0 models for each hyperparameter
# - hour0_nt15_noise15.pkl
# ...

# 4. Copy best hour-0 model to standard location (if needed)
cp models/sweep/hour0_nt15_noise15.pkl models/hour0_forest_flow.pkl
```

**🆕 Unified Training:**
The sweep automatically trains **BOTH models** with identical hyperparameters:
- Each `(nt, n_noise)` combination trains:
  1. Hour-0 IID model (unconditional generation)
  2. Autoregressive model (conditional generation)
- Results include training times for both models
- Hour-0 models saved separately for each configuration
- Select best configuration based on TSTR + quality metrics

The sweep will:
- 🆕 Train hour-0 IID model for each hyperparameter combination
- Train autoregressive model with same hyperparameters
- Evaluate with TSTR (Train on Synthetic, Test on Real)
- Compute quality metrics (KS, correlation, clinical validity)
- Generate plots to predict production performance
- Save all hour-0 models to `models/sweep/` for later comparison

## 📖 Script Details

### Hour-0 Model Scripts (NEW!)

#### 00_prepare_hour0_data.py

**What it does:**
- Extracts hour-0 patient states from flat table
- Filters to hours_in == 0 (or first hour per patient)
- Selects static features (demographics) + hour-0 vitals
- Handles missing values (median for numeric, mode for categorical)
- Keeps ID columns in CSV for bookkeeping; excludes them during model fitting

**Output:** `packages/data/processed/hour0_data.csv`

---

#### 00b_fit_hour0_preprocessor.py

**What it does:**
- Fits TabularPreprocessor on hour-0 data
- Learns min/max ranges for numeric features
- Encodes categorical values
- Saves preprocessor for hour-0 generation

**Output:** `models/hour0_preprocessor.pkl`

---

#### 00c_train_hour0_model.py

**What it does:**
- Trains IID (non-autoregressive) Forest-Flow model
- Generates initial patient states: demographics + hour-0 vitals
- No conditioning (fully unconditional generation)
- Model learns P(demographics, vitals_t0)

**Configuration:**
```python
nt = 50          # Flow time steps (same as autoregressive)
n_noise = 100    # Noise samples (critical for quality)
max_rows = None  # Use all data for best results
```

**Output:** `models/hour0_forest_flow.pkl`

---

#### 05_evaluate_hour0_model.py

**What it does:**
- Generates synthetic hour-0 samples
- Compares distributions (KS tests for numeric features)
- Compares correlations (MAE of correlation matrices)
- Checks clinical plausibility (valid ranges)
- Generates plots and quality report

**Output:**
- `results/hour0_quality_metrics.txt`
- `results/hour0_distributions.png`
- `results/hour0_correlations.png`

---

### Autoregressive Model Scripts

#### 01_preprocess_data.py

**What it does:**
- Loads MIMIC-III flat table
- Identifies static vs dynamic features
- Creates lagged features for autoregressive modeling
- Saves preprocessed data

**Configuration:**
```python
max_rows = None  # Set integer for testing, None for all data
```

**Output:** `packages/data/processed/autoregressive_data.csv`

---

#### 02_train_model.py

**What it does:**
- Loads preprocessed data
- Loads pre-fitted preprocessor from `models/preprocessor_full.pkl` (recommended)
- Trains selected backbone (`hs3f` default, or `forest-flow`)
- Saves model and preprocessor

**Configuration (CLI):**
```bash
uv run python scripts/02_train_model.py \
  --model-type hs3f \
  --nt 50 \
  --n-noise 100 \
  --max-rows 10000 \
  --n-jobs 4 \
  --batch-size 500
```

**Output:** `models/forest_flow_model.pkl`, `models/preprocessor.pkl`

---

#### 03_generate_synthetic.py

**What it does:**
- **Mode 1 (Hour-0)**: Generates fully synthetic patients using hour-0 model
  - Phase 1: Generate demographics + hour-0 vitals (IID)
  - Phase 2: Generate trajectories from hour-0 states (Autoregressive)
- **Mode 2 (Original)**: Uses real initial conditions for trajectories
  - Loads real patient data for initial states
  - Generates trajectories (Autoregressive)

**Usage:**
```bash
# Fully synthetic (Hour-0 model required)
uv run python scripts/03_generate_synthetic.py --use-hour0 --n-patients 100 --n-timesteps 48

# Using real initial conditions (original mode)
uv run python scripts/03_generate_synthetic.py --n-patients 100 --n-timesteps 48
```

**Configuration (command-line):**
```bash
--use-hour0        # Enable hour-0 model for fully synthetic generation
--n-patients 100   # Number of synthetic patients
--n-timesteps 10   # Trajectory length (hours)
```

**Output:** `results/synthetic_patients.csv`

---

### 04_evaluate_quality.py

**What it does:**
- Loads synthetic and real data
- Computes KS statistics (marginal distributions)
- Computes correlation Frobenius norm
- Checks clinical range violations
- Generates quality report

**Output:** `results/quality_metrics.txt`

---

## 🎯 Choosing Parameters

### Quick Testing (5 minutes)
```python
nt = 10
n_noise = 10
max_rows = 5000
```
Good for: Development, debugging, quick iterations

### Balanced (30 minutes)
```python
nt = 20
n_noise = 30
max_rows = 50000
```
Good for: Preliminary experiments, proof of concept

### Production (2+ hours)
```python
nt = 50
n_noise = 100
max_rows = None  # All data
```
Good for: Final models, publication, deployment

### Key Parameter Effects

**`nt` (flow time steps):**
- Higher = better quality, longer training
- Recommended: 50 for production, 10-20 for testing

**`n_noise` (noise samples):**
- Higher = better quality, much longer training
- Most critical parameter! (See sweep results)
- Recommended: 100 for production, 10-20 for testing

**`max_rows` (training data size):**
- More data = better model, but diminishing returns after 100k
- Recommended: All data for production, 5k-50k for testing

## 🔧 Troubleshooting

### Memory Issues
```bash
# Reduce training data size and parallelism
uv run python scripts/02_train_model.py --max-rows 5000 --n-jobs 2
```

### Slow Training
```bash
# Reduce flow discretization and noise multiplier
uv run python scripts/02_train_model.py --nt 10 --n-noise 10 --max-rows 5000
```

### Poor Quality
```bash
# Increase model capacity and use more training rows
uv run python scripts/02_train_model.py --nt 50 --n-noise 100 --max-rows 0
```

## 📞 Need Help?

1. Check `docs/SWEEP_ARCHITECTURE.md` for detailed sweep documentation
2. Check `README.md` and `docs/architecture/PROJECT_ARCHITECTURE.md` for package structure
3. Check `docs/guides/autoregressive-forest-flow.md` for theory

## ✨ New in Refactored Version

These numbered scripts use the new modular package structure:

```python
# New clean imports
from packages.data import TabularPreprocessor, prepare_autoregressive_data
from packages.models import ForestFlow, sample_trajectory
from packages.evaluation import compute_quality_metrics
```

All scripts are:
- ✅ Self-contained and documented
- ✅ Using professional package structure
- ✅ Configurable via CLI flags (with sensible defaults)
- ✅ Produce clear output messages
- ✅ Handle errors gracefully
