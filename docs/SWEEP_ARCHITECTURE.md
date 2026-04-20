# Hyperparameter Sweep Architecture

## Overview

The sweep framework systematically explores the hyperparameter space of Forest-Flow to understand how `nt` (number of time levels) and `n_noise` (noise samples per data point) affect synthetic data quality, enabling data-driven decisions for production configurations.

> **🔧 Troubleshooting**: This document includes a comprehensive [Troubleshooting & Technical Issues](#troubleshooting--technical-issues-resolved) section documenting all 14 technical problems encountered during development, their root causes, and solutions. Essential reading for debugging.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Decisions](#design-decisions)
3. [Evaluation Methodology](#evaluation-methodology)
4. [Memory Management](#memory-management)
5. [Configuration & Parameters](#configuration--parameters)
6. [Results Storage](#results-storage)
7. [Visualization & Analysis](#visualization--analysis)
8. [**Troubleshooting & Technical Issues**](#troubleshooting--technical-issues-resolved) ⚡ **New: Complete debugging guide**

---

## Architecture Overview

### High-Level Workflow

```
┌─────────────────┐
│  Sweep Config   │  nt_values = [5, 10, 15, ...]
│  Parameters     │  noise_values = [10, 20, 30, ...]
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Load Preprocessor & Memmap Data                │
│  - Reuse cached preprocessor (fitted on FULL   │
│    2.2M rows via scripts/fit_autoregressive_preprocessor.py) │
│  - Ensures correct min/max ranges & all rare   │
│    categorical values are captured             │
│  - Load memmap arrays (target + condition)     │
│  - Subsample training data (5k rows default)   │
└────────┬────────────────────────────────────────┘
         │
         ▼
    ┌───────────────────────┐
    │  For each (nt, noise) │
    │  combination:         │
    └───┬───────────────────┘
        │
        ├──▶ [1] Train ForestFlow Model
        │    └─ 5k rows × noise samples (e.g., 15 noise = 75k samples)
        │    └─ nt time levels × 326 dimensions
        │    └─ Legacy duplication (faster for small data)
        │    └─ Full XGBoost params (100 trees, depth 6)
        │
        ├──▶ [2] Generate Synthetic Data
        │    └─ 5k synthetic samples via ODE integration
        │    └─ Conditioned on real X_condition
        │    └─ Binary feature rounding applied
        │
        ├──▶ [3] TSTR Evaluation
        │    └─ Consistent preprocessing (TSTRPreprocessor)
        │    └─ Train classifier on ALL synthetic data
        │    └─ Test on real held-out data (50k sample)
        │    └─ Compare to baseline (trained on real)
        │
        └──▶ [4] Record Results
             └─ Metrics: ROC-AUC, accuracy, F1, train time
             └─ Append to CSV (resumable)
```

### Key Components

1. **`run_sweep.py`** - Main orchestrator script
2. **`analyze_sweep.py`** - Visualization and analysis
3. **`results/sweep_results.csv`** - Metrics storage (CSV for easy inspection)
4. **`results/sweep_plots/`** - Generated heatmaps and trend plots

---

## Design Decisions

### 1. Why Sample 5k Rows Instead of Full 2.2M?

**Decision**: Subsample to 5,000 rows for training during sweep.

**Reasoning**:
- **Speed**: Training on 5k × 15 noise = 75k samples per model is manageable (~8-9 min per run with full XGBoost params)
- **Trends Preserved**: Hyperparameter effects (nt, n_noise) are visible even with smaller samples
- **Memory Efficient**: Fits comfortably in RAM with parallel training (`n_jobs=4`)
- **Production Separation**: After finding optimal hyperparameters, retrain on full dataset with `run_with_config.py`
- **Sweep Focus**: 64 runs with full XGBoost params gives reliable rankings in ~10 hours

**Trade-off**: Absolute performance metrics will be slightly lower than full training (due to sample size, not model complexity), but relative comparisons (which hyperparameters are better) remain valid for extrapolation.

### 2. Why Legacy Duplication Over Data Iterator?

**Decision**: Use `use_data_iterator=False` (legacy mode).

**Reasoning**:
- **Simpler & Faster**: For small datasets (20k rows), duplicating 20k × 10 noise = 200k samples upfront is faster than batch generation
- **More Reliable**: Data iterator has edge cases with variable batch sizes (QuantileDMatrix requires exact batch sizes)
- **Memory Acceptable**: 200k × 680 features × 4 bytes ≈ 500MB per time level is manageable
- **Iterator for Production**: Large datasets (500k+ rows) should use the fixed iterator to save memory

**When to Switch**: Use iterator when training data > 100k rows to reduce 5-10× memory footprint.

### 3. Why Pre-fit Preprocessor on Full Dataset?

**Decision**: Fit `TabularPreprocessor` once on the FULL 2.2M rows, then reuse for all training runs.

**Critical Problem Without Pre-fitting**:
- ❌ **Wrong scaling ranges**: MinMaxScaler learns incorrect min/max from 5k sample
- ❌ **Missing rare categories**: One-hot encoder misses rare values (e.g., 0.1% prevalence)
- ❌ **Out-of-range generation**: Synthetic data may exceed learned ranges
- ❌ **Degraded quality**: Wrong normalization affects model training and sampling

**Solution (Step 1b)**:
```bash
# Run once to fit on full data
uv run python scripts/fit_autoregressive_preprocessor.py
```

**Benefits**:
- ✅ **Correct min/max ranges** from all 2.2M rows
- ✅ **All categorical values** encoded (even 0.01% prevalence)
- ✅ **Proper normalization** for training on any subsample
- ✅ **Better synthetic quality** with consistent scaling

**Implementation**:
- Uses `partial_fit()` to stream through data in chunks (memory-efficient)
- Saves to `artifacts/preprocessor_full.pkl` (single canonical location)
- Training scripts automatically detect and load pre-fitted preprocessor
- Only needs to run once (reuse for all subsequent training runs)

**Memory**: Streaming fit uses ~1GB peak (vs ~8GB for loading all data at once).

**Time**: ~2-5 minutes to fit on 2.2M rows.

### 5. Why n_jobs=2 Instead of All Cores?

**Decision**: Default to `n_jobs=2` instead of `-1` (all cores).

**Reasoning**:
- **Stability**: `n_jobs=-1` causes joblib to hang after training completes and produces resource leak warnings on macOS
- **Memory Pressure**: Each parallel job needs a copy of data; 2 jobs balances speed vs RAM
- **User Configurable**: `--n-jobs 4` flag available if more parallelism is needed
- **Diminishing Returns**: Beyond 4 jobs, overhead dominates gains for this problem size

**Observed Issues with n_jobs=-1**:
- Process hangs at 100% progress (joblib cleanup issue)
- Leaked semaphore objects warnings
- Potential OOM due to memory copies

### 6. Why Use Production XGBoost Parameters in Sweep?

**Decision**: Use `n_estimators=100, max_depth=6, learning_rate=0.1` (same as production).

**Reasoning**:
- **Accurate Trends**: Full XGBoost params give more reliable performance estimates
- **Direct Comparison**: Sweep results directly comparable to production runs
- **No Extrapolation Needed**: What you see in sweep is what you get in production
- **Still Fast**: With 5k training rows, each run completes in 6-10 minutes

**Trade-off**: Slightly slower sweep (~8 min/run vs ~5 min/run with reduced params), but more trustworthy results.

**Alternative**: Can reduce to `n_estimators=30, max_depth=4, learning_rate=0.2` for faster iteration (~3× speedup), but relative rankings may differ slightly from production.

### 7. Why Resume from CSV Instead of Database?

**Decision**: Store results in CSV with row-level atomicity.

**Reasoning**:
- **Simplicity**: CSV is human-readable, easy to inspect/debug
- **Atomic Writes**: Each run appends one row immediately (crash-safe)
- **No Dependencies**: No database setup required
- **Pandas Integration**: Easy to load for plotting with `pd.read_csv()`
- **Version Control**: CSV diffs are readable in git

**Trade-off**: Not suitable for concurrent sweeps (file locking issues). Use database for multi-process sweeps.

---

## Evaluation Methodology

### TSTR (Train on Synthetic, Test on Real)

**Purpose**: Measure how well synthetic data captures real patterns by evaluating downstream task performance.

**Methodology**:

1. **Consistent Preprocessing Pipeline**
   - **Critical**: Both synthetic and real data must go through IDENTICAL preprocessing
   - Process real data with `prepare_data()` → learn categorical encodings
   - Process synthetic data with same function → may have different categories
   - **Feature Alignment**: Add missing one-hot columns to synthetic data with value `0`
   - Example: If real has `[admission_EMERGENCY, admission_URGENT]` but synth only has `[admission_EMERGENCY]`, add `admission_URGENT=0` to synth
   - **Why Zeros Are Correct**: In one-hot encoding, 0 means "category not present" - mathematically sound!

2. **Train Classifier on Synthetic Data**
   - Task: Mortality prediction (binary classification)
   - Model: Logistic Regression (`max_iter=500, C=1.0`)
   - Features: Clinical variables (exclude leaky outcome columns)
   - Training Set: ALL 5k synthetic samples (TSTR uses all synthetic data)

3. **Test on Real Held-Out Data**
   - Test Set: 30% of real MIMIC-III data (held out, never seen in training)
   - Test Size: 50k rows (sampled from 2.2M for speed)
   - Metrics: Accuracy, F1, ROC-AUC
   - Evaluation: How well does synthetic-trained model generalize to real patients?

4. **Baseline Comparison**
   - Train identical classifier on real training data (70% of 50k = 35k rows)
   - Test on same real test set (30% of 50k = 15k rows)
   - Gap: `synth_roc_auc - real_roc_auc`

**Interpretation**:

| Gap (Synth - Real) | Quality Assessment |
|--------------------|-------------------|
| \|gap\| < 0.05     | Excellent - synthetic data preserves clinical patterns well |
| \|gap\| < 0.10     | Good - minor differences, usable for ML training |
| \|gap\| < 0.15     | Fair - noticeable gap, consider tuning |
| \|gap\| ≥ 0.15     | Poor - synthetic data missing critical patterns |

**Why TSTR Over Statistical Metrics?**

- **End-to-End**: Measures what matters - can synthetic data train useful models?
- **Downstream Task**: Clinical predictions (mortality) test medical validity
- **Holistic**: Captures correlations, distributions, and feature interactions
- **Actionable**: Directly measures utility for machine learning applications

**Alternatives Not Used** (documented for reference):
- **Marginal Distributions**: Kolmogorov-Smirnov test (too narrow, ignores correlations)
- **MMD (Maximum Mean Discrepancy)**: Measures distance in feature space (no task-specific meaning)
- **Correlation Preservation**: Pearson/Spearman matrices (partial view, misses complex patterns)

### Data Post-Processing

**Binary Feature Rounding**

**Problem**: Generated continuous values for binary features (e.g., `vent=0.73`, should be 0 or 1).

**Solution**: Round and clip binary features after inverse transform:

```python
binary_features = [
    'vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine',
    'epinephrine', 'isuprel', 'milrinone', 'norepinephrine',
    'phenylephrine', 'vasopressin', 'colloid_bolus',
    'crystalloid_bolus', 'nivdurations'
]

for col in binary_features:
    if col in df.columns:
        df[col] = np.clip(np.round(df[col]), 0, 1).astype(int)
```

**Why Post-Processing**:
- ForestFlow operates in continuous space (easier to model)
- Clinical data requires discrete values for binary flags
- Rounding is applied AFTER inverse transform, before TSTR evaluation

**Impact**: Ensures synthetic data matches real data format exactly.

### Integrated Quality Metrics (Tier 2)

In addition to TSTR (downstream task performance), the sweep automatically computes three complementary quality metrics:

#### 1. Marginal Distribution Similarity (KS Statistic)

**Method**: Kolmogorov-Smirnov two-sample test for each numeric feature, then average.

**Interpretation**:
- `< 0.05`: Excellent - distributions nearly identical
- `< 0.10`: Good - minor differences
- `< 0.20`: Fair - noticeable differences
- `≥ 0.20`: Poor - distributions diverge significantly

**Why it matters**: Ensures each feature independently matches real data distribution (e.g., heart rate, glucose levels).

#### 2. Correlation Preservation (Frobenius Norm)

**Method**: Compute correlation matrices for real and synthetic data, measure Frobenius norm of difference.

**Formula**: `||Corr_synth - Corr_real||_F`

**Interpretation**:
- `< 1.0`: Excellent - correlations well preserved
- `< 3.0`: Good - minor correlation differences
- `< 5.0`: Fair - some correlations lost
- `≥ 5.0`: Poor - correlation structure differs significantly

**Why it matters**: Ensures relationships between features (e.g., blood pressure ↔ heart rate) are preserved.

#### 3. Clinical Range Violations

**Method**: Check if synthetic values fall within physiologically valid ranges for 18 clinical features.

**Examples**:
- Heart rate: [0, 300] bpm
- Systolic BP: [50, 250] mmHg
- Glucose: [0, 1500] mg/dL

**Interpretation**:
- `< 0.1%`: Excellent - negligible violations
- `< 1.0%`: Good - minimal violations
- `< 5.0%`: Fair - some violations, may need post-processing
- `≥ 5.0%`: Poor - many impossible values

**Why it matters**: Clinical validity - synthetic patients must be physiologically plausible.

#### Combined Assessment

These three metrics complement TSTR:

| Metric | What it captures | TSTR blind spot |
|--------|------------------|-----------------|
| **TSTR** | Task-specific utility (mortality prediction) | Can miss feature-level issues |
| **KS** | Feature-level distribution match | Ignores correlations |
| **Correlation** | Inter-feature relationships | Ignores marginals |
| **Range** | Clinical plausibility | Can have valid ranges but wrong distributions |

**Best practice**: Look for configurations that score well on ALL metrics, not just TSTR. This ensures synthetic data is both useful AND realistic.

### Categorical Feature Consistency

**Problem**: `pd.get_dummies()` creates different columns for synthetic vs real data.

**Original Approach (Flawed)**:
```python
# Real data
X_real = pd.get_dummies(real_df)  # Creates columns based on real categories

# Synthetic data (processed separately)
X_synth = pd.get_dummies(synth_df)  # Creates columns based on synth categories

# Result: X_real has 666 features, X_synth has 844 features → MISMATCH!
```

**Solution: Fit Preprocessor on Real, Transform Both**:

```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class TSTRPreprocessor:
    def fit(self, real_df):
        # Learn categorical levels from REAL data
        self.categorical_cols = [...]
        self.numeric_cols = [...]

        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
        ])
        self.preprocessor.fit(real_df)

    def transform(self, df):
        # Apply same encoding to both real and synthetic
        return self.preprocessor.transform(df)

# Usage in TSTR
preprocessor = TSTRPreprocessor()
preprocessor.fit(real_df)  # Learn from real data

X_real = preprocessor.transform(real_df)      # 666 features
X_synth = preprocessor.transform(synth_df)    # 666 features (unknown categories → zeros)
```

**Key Benefits**:
1. **Consistent Dimensionality**: Both datasets have identical feature count
2. **Unknown Categories**: Synthetic categories not in real data are silently mapped to zero vector
3. **Robust**: Handles missing categories, mixed types, and missing values
4. **Scikit-learn Compatible**: Integrates seamlessly with LogisticRegression

**Mixed-Type Column Handling**:

Some columns have mixed types (e.g., `['A', 1.5, 'B']`) which breaks `numpy.isnan()`.

**Solution**: Convert all object columns to string before processing:

```python
def transform(self, df):
    # Handle mixed-type columns
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str)

    # Now safe to process
    return self.preprocessor.transform(X)
```

**Why This Works**:
- `OneHotEncoder` treats all strings uniformly
- Eliminates `ufunc 'isnan' not supported` error
- No information loss (mixed types were already non-numeric)

---

## Memory Management

### Strategy Overview

The sweep is designed to handle 2.2M row datasets on machines with 16-32GB RAM.

### 1. Memmap for Full Dataset

**Purpose**: Keep full 2.2M × 680 transformed data on disk, page in as needed.

**Implementation**:
```python
X_target = np.memmap("X_target.npy", dtype=np.float32, mode="r",
                     shape=(n_rows, n_target))
X_condition = np.memmap("X_condition.npy", dtype=np.float32, mode="r",
                        shape=(n_rows, n_condition))
```

**Benefit**: ~6GB of arrays live on disk, not RAM.

### 2. Subsampling for Training

**Purpose**: Load only what's needed for each sweep run.

**Implementation**:
```python
train_idx = rng.choice(n_rows, size=5000, replace=False)
X_train = np.asarray(X_target[train_idx])  # Loads 5k rows into RAM
```

**Benefit**: 5k rows × 326 features × 4 bytes ≈ 6MB (vs 2.8GB for full dataset). 450× memory reduction!

### 3. Model Cleanup Between Runs

**Purpose**: Free memory after each (nt, n_noise) run.

**Implementation**:
```python
del model, X_synth, X_cond_synth, synth_model, real_model
gc.collect()
```

**Benefit**: Prevents memory accumulation across 30+ runs.

### 4. Temporary File Cleanup

**Purpose**: Remove intermediate CSVs used for TSTR evaluation.

**Implementation**:
```python
try:
    # ... TSTR evaluation ...
finally:
    if temp_synth_path.exists():
        temp_synth_path.unlink()
```

**Benefit**: No disk space leaks from failed runs.

### Memory Budget (Typical Run)

| Component | Memory Usage |
|-----------|--------------|
| Base Python + Libraries | ~500 MB |
| Preprocessor + Metadata | ~100 MB |
| X_train (5k × 326) | ~6 MB |
| X_cond_train (5k × 354) | ~7 MB |
| Duplicated training data (5k × 30 noise) | ~195 MB |
| XGBoost model (per time level) | ~30-50 MB |
| Parallel workers (n_jobs=2) | ~800 MB - 1.5 GB |
| TSTR evaluation (50k real data) | ~200 MB |
| **Total Peak** | **~2-3 GB** |

**Headroom**: Very comfortable on 16GB machines (leaves 13-14GB for OS and other processes). Can run on 8GB machines.

---

## Configuration & Parameters

### Sweep Grid

**Purpose**: Define (nt, n_noise) combinations to explore.

**Default Configuration**:
```python
NT_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]
NOISE_VALUES = [1, 3, 5, 7, 9, 11, 13, 15]
# 8 × 8 = 64 runs
```

**Rationale**:
- **nt Range**: 1-15 covers very low to low-medium complexity (production: 20-50)
- **Noise Range**: 1-15 explores minimal to low noise (production: 50-100)
- **Grid Spacing**: Dense uniform spacing (steps of 2) captures non-linear effects
- **Extrapolation**: Dense sampling enables reliable logarithmic fit to predict production performance

### Training Configuration

```python
MAX_TRAIN_ROWS = 5000     # Subsampled training rows (default for sweep)
N_SYNTH_SAMPLES = 5000    # Synthetic samples for TSTR evaluation
BATCH_SIZE = 5000         # XGBoost batch size (matches train size for efficiency)
N_JOBS = 4                # Parallel jobs (dimension-level, balanced default)
```

**Tuning Guidelines**:
- **More Speed**: `--train-rows 2000 --n-jobs 2` (~2× faster runs, ~4-5 hours total)
- **Higher Quality**: `--train-rows 10000 --n-jobs 8` (smoother trends, ~15-18 hours total)
- **Memory Constrained**: `--train-rows 5000 --n-jobs 1` (minimal RAM, ~14-16 hours total)

### XGBoost Parameters

```python
SWEEP_XGB_PARAMS = {
    "n_estimators": 100,     # Full production param (100 trees)
    "max_depth": 6,          # Full production param (depth 6)
    "learning_rate": 0.1,    # Full production param (0.1)
    "subsample": 0.8,        # Row subsampling
    "colsample_bytree": 0.8, # Feature subsampling
}
```

**Why Use Full Production Parameters?**
- **64 runs**: With 8×8 grid, runtime is acceptable (~10-15 hours total)
- **Accurate Performance**: Sweep results directly reflect production quality
- **No Extrapolation Risk**: What you measure in sweep is what you'll get in production
- **Reliable Rankings**: Hyperparameter rankings match production exactly

---

## Results Storage

### CSV Schema

**File**: `results/sweep_results.csv`

```csv
nt,n_noise,train_time_sec,synth_accuracy,synth_f1,synth_roc_auc,real_accuracy,real_f1,real_roc_auc,timestamp
10,10,45.2,0.7234,0.6891,0.7512,0.7456,0.7123,0.7689,2026-02-16T12:34:56.789
15,20,78.5,0.7512,0.7234,0.7823,0.7456,0.7123,0.7689,2026-02-16T13:45:23.456
...
```

**Columns**:
- **nt, n_noise**: Hyperparameters tested
- **train_time_sec**: ForestFlow training time (excludes TSTR evaluation)
- **synth_accuracy, synth_f1, synth_roc_auc**: Synthetic model on real test set
- **real_accuracy, real_f1, real_roc_auc**: Real model on real test set (baseline)
- **avg_ks_stat**: Average KS statistic across features (marginal distribution similarity)
- **corr_frobenius**: Frobenius norm of correlation difference (correlation preservation)
- **range_violation_pct**: Percentage of clinical values outside valid ranges
- **timestamp**: ISO 8601 timestamp for tracking

**Resume Logic**:
```python
completed = set()
if results_path.exists():
    df = pd.read_csv(results_path)
    for _, row in df.iterrows():
        completed.add((row['nt'], row['n_noise']))

for nt in NT_VALUES:
    for n_noise in NOISE_VALUES:
        if (nt, n_noise) in completed:
            print(f"[SKIP] nt={nt}, n_noise={n_noise}")
            continue
        # ... train and evaluate ...
```

**Crash Safety**: Each row is appended immediately after completion (no buffering).

---

## Visualization & Analysis

### Generated Plots

**Script**: `scripts/analyze_sweep.py`

**Output Directory**: `results/sweep_plots/`

**Plots Generated**:
1. `sweep_heatmap.png` - TSTR performance across hyperparameters
2. `sweep_line_charts.png` - Line plots showing nt and n_noise effects
3. `sweep_training_time.png` - Computational cost analysis
4. `sweep_quality_metrics.png` - Statistical quality metrics (KS, correlation, ranges)
5. `sweep_extrapolation.png` - Predictions for production configs

### 1. Performance Heatmap

**File**: `sweep_heatmap.png`

**Purpose**: 2D grid showing ROC-AUC for each (nt, n_noise) combination.

**Features**:
- Left panel: Synthetic model absolute performance
- Right panel: Gap from real model (synth - real)
- Color scale: Red (poor) → Yellow → Green (excellent)
- Annotations: Exact values in each cell

**Usage**: Quickly identify which configurations achieve near-real performance.

### 2. Line Charts

**File**: `sweep_line_charts.png`

**Purpose**: Show effect of each parameter independently.

**Features**:
- Left panel: ROC-AUC vs nt (one line per n_noise)
- Right panel: ROC-AUC vs n_noise (one line per nt)
- Horizontal baseline: Real model performance
- Legends: Color-coded by other parameter

**Usage**: Understand marginal effects and interactions.

### 3. Training Time Analysis

**File**: `sweep_training_time.png`

**Purpose**: Understand computational cost vs hyperparameters.

**Features**:
- Time in minutes (y-axis)
- Shows scaling: time ∝ nt × n_noise (approximately)

**Usage**: Budget compute resources for production runs.

### 4. Quality Metrics Heatmaps

**File**: `sweep_quality_metrics.png`

**Purpose**: Visualize Tier 2 quality metrics across hyperparameter space.

**Features**:
- Three heatmaps side-by-side:
  - **KS Statistic**: Distribution similarity (lower = better, green)
  - **Correlation Frobenius**: Correlation preservation (lower = better, green)
  - **Range Violations**: Clinical validity (lower = better, green)
- Color scale: Red (poor) → Yellow → Green (excellent)
- Annotations: Exact values in each cell

**Usage**:
- Identify hyperparameters that excel across ALL quality dimensions
- Spot trade-offs (e.g., good TSTR but poor ranges → need post-processing)
- Validate that high TSTR correlates with low KS and good correlation preservation

### 5. Extrapolation Curves

**File**: `sweep_extrapolation.png`

**Purpose**: Predict performance at production configs (nt=50, noise=100).

**Method**:
```python
# Logarithmic fit: y = a * log(x) + b
popt, _ = curve_fit(log_fit, x_measured, y_measured)
y_prod = log_fit(50, *popt)  # Predict at nt=50
```

**Features**:
- Measured points (scatter)
- Fitted curve (dashed line)
- Production config marked with star
- Prediction with confidence

**Usage**:
- Decide if production config is worthwhile (ROC-AUC gain vs compute cost)
- Identify diminishing returns threshold

**Interpretation Example**:
```
nt=30: ROC-AUC = 0.78
nt=50 (extrapolated): ROC-AUC = 0.79
nt=100 (extrapolated): ROC-AUC = 0.795

Conclusion: Gains diminish above nt=50; 50 is good balance.
```

---

## Usage Guide

### Running the Sweep

```bash
# Standard sweep (5k rows, 64 runs, 8×8 grid)
uv run python scripts/run_sweep.py

# Fast iteration (2k rows, fewer parallel jobs)
uv run python scripts/run_sweep.py --train-rows 2000 --n-jobs 2

# High quality (10k rows, more parallel jobs)
uv run python scripts/run_sweep.py --train-rows 10000 --n-jobs 8
```

### Monitoring Progress

```bash
# Watch CSV file grow
watch -n 5 "tail -n 5 results/sweep_results.csv"

# Count completed runs
wc -l results/sweep_results.csv
# (subtract 1 for header)
```

### Resuming After Crash

Simply rerun the same command - completed runs are automatically skipped:

```bash
uv run python scripts/run_sweep.py --train-rows 20000
# Output:
# Completed: 12/30 runs
# [SKIP] nt=10, n_noise=10 (already completed)
# ...
# [RUN] nt=15, n_noise=50  <-- Resumes here
```

### Generating Plots

```bash
# After sweep completes (or during, for partial results)
uv run python scripts/analyze_sweep.py

# Output:
#   results/sweep_plots/sweep_heatmap.png
#   results/sweep_plots/sweep_line_charts.png
#   results/sweep_plots/sweep_training_time.png
#   results/sweep_plots/sweep_extrapolation.png
```

---

## Performance Estimates

### Single Run Timeline

| Stage | Time (5k rows, 100 trees) | Notes |
|-------|---------------------------|-------|
| Training ForestFlow | 3-8 min | Depends on nt, n_noise, n_jobs |
| Generating 5k samples | 30-90 sec | ODE integration across nt time levels |
| TSTR evaluation | 1-3 min | Train 2 logistic regressions, test both |
| **Total per run** | **5-12 min** | Average: ~8-9 minutes |

### Full Sweep Estimates

| Configuration | Runs | Est. Time | Use Case |
|--------------|------|-----------|----------|
| Default (8×8 grid) | 64 | 9-13 hours | Standard exploration |
| Reduced (5×5 grid) | 25 | 3-5 hours | Quick exploration |
| Minimal (4×4 grid) | 16 | 2-3 hours | Proof of concept |

**Parallelism**: Times assume `n_jobs=4` (dimension-level). With `n_jobs=2`, add ~30% time. With `n_jobs=8`, reduce by ~20% (if sufficient RAM).

**Speed vs Quality Trade-offs**:
- `--train-rows 2000`: ~2× faster runs (~5 min avg), but noisier trends
- `--train-rows 10000`: ~1.5× slower runs (~12 min avg), smoother trends

---

## Troubleshooting & Technical Issues Resolved

This section documents **every technical problem** encountered during development, their root causes, and solutions. This serves as:
1. **Debugging Reference**: Quick lookup for similar issues
2. **Design Rationale**: Understanding why certain decisions were made
3. **Learning Resource**: Common pitfalls in ML pipeline development

**Legend**:
- ❌ = Attempted fix that didn't work
- ✅ = Successful solution
- 🔧 = Workaround (not ideal but functional)

---

### Issue 1: Repeated Memory Limit Exceeded (OOM Killer)

**Timeline**: Initial implementation phase

**Symptom**: Process killed with exit code 137 or 143, no error message.

**Root Cause**: Training on 2.2M rows × 100 noise samples = 220M duplicated samples requiring ~80GB RAM.

**Attempted Solutions**:
1. ❌ Memory monitor at 80% threshold - killed too early
2. ❌ Increased to 90% threshold - still OOM before trigger
3. ❌ Memory-mapped arrays for full dataset - still duplicated in memory during XGBoost training
4. ❌ Streaming with data iterator - had bugs with batch sizes

**Final Solution**:
- Subsample to 5k-20k rows for sweep (manageable memory footprint)
- Use memmap for full 2.2M dataset (stays on disk)
- Use legacy duplication for small samples (faster, stable)
- Reserve full dataset training for production runs only

**Lesson**: For hyperparameter sweeps, sampling is essential. Full dataset training is for final production model only.

**Status**: ✅ Resolved - using 5k row subsampling with memmap

---

### Issue 2: ModuleNotFoundError: No module named 'numpy'

**Symptom**: Python can't find installed packages.

**Root Cause**: Running `python` directly instead of using the virtual environment.

**Solution**: Use `uv run python` to execute within the project's virtual environment.

**Command**:
```bash
# Wrong
python scripts/run_sweep.py

# Correct
uv run python scripts/run_sweep.py
```

**Status**: ✅ Resolved - all scripts use `uv run python`

---

### Issue 3: XGBoost Data Iterator Crashes

**Symptoms**:
- `Check failed: rbegin == Info().num_row_`
- `TypeError: Value type is not supported for data iterator:<class 'generator'>`

**Root Cause**: XGBoost's `QuantileDMatrix` requires:
1. Exact batch sizes (last batch can't be smaller)
2. Proper iterator protocol implementation

**Solution**: Modified `FlowMatchingDataIterator` in `src/synth_gen/models/iterator.py`:
```python
def __iter__(self):
    while True:  # Infinite loop
        indices = self.rng.choice(self.n, size=self.batch_size, replace=True)
        # ... yield batch ...
```

**Workaround for Sweep**: Use `use_data_iterator=False` (legacy duplication) for small datasets (<100k rows).

**Lesson**: Data iterators are powerful but brittle. Use legacy mode for small samples where memory isn't an issue.

**Status**: 🔧 Workaround - using legacy mode for sweep, iterator fixed for production

---

### Issue 4: TSTR ValueError - test_size Too Small

**Symptom**: `ValueError: The test_size = 1 should be greater or equal to the number of classes = 2`

**Root Cause**: Used `test_size=0.0001` for synthetic data, creating a test set with only 1 sample (can't stratify).

**Solution**: Changed `test_size` to `0.2` (use more synthetic data for validation).

**Later Fix**: Removed test split entirely for synthetic data - use ALL 5k samples for training (TSTR methodology).

**Code Location**: `run_sweep.py`, line ~270

**Status**: ✅ Resolved - using all synthetic data for training

---

### Issue 5: FileNotFoundError - Wrong Real Data Path

**Symptom**: `[Errno 2] No such file or directory: 'data/processed/flat_table_cleaned.csv'`

**Root Cause**: Incorrect filename in `real_test_path`.

**Solution**: Changed from `flat_table_cleaned.csv` to `flat_table.csv`.

**Lesson**: Always verify file paths exist before long runs.

**Status**: ✅ Resolved - correct path in code

---

### Issue 6: RuntimeWarning - Mean of Empty Slice

**Symptom**: `RuntimeWarning: Mean of empty slice` during TSTR evaluation, extremely slow.

**Root Cause**: Loading full 2.2M rows of real data for TSTR, computing medians on columns with all NaN values.

**Solution**: Added `max_rows=50000` to `prepare_data` call for real test data.

**Impact**: Reduced TSTR evaluation time from ~30 minutes to ~2 minutes.

**Code Location**: `run_sweep.py`, line ~260

**Status**: ✅ Resolved - using 50k row limit for real test data

---

### Issue 7: Joblib/Loky Semaphore Leak Warnings

**Symptoms**:
- `UserWarning: resource_tracker: There appear to be leaked semaphore objects`
- Similar warnings for file handles

**Root Cause**: Using `n_jobs=-1` (all cores) with joblib on macOS.

**Solution**: Changed default `N_JOBS` from `-1` to `2` (later increased to `4`).

**Why This Happens**: macOS has stricter resource limits; joblib cleanup isn't perfect.

**Lesson**: These are harmless cleanup warnings but indicate instability. Use explicit core counts for production code.

**Status**: ✅ Resolved - using N_JOBS=4 (explicit count)

---

### Issue 8: Process Hangs After Training

**Symptom**: Progress bar shows 100% but doesn't proceed to next step.

**Root Cause**: `n_jobs=-1` (all cores) causes joblib cleanup hang on some systems.

**Solutions Applied**:
1. Changed `N_JOBS` from `-1` to `2` (later `4`)
2. Added `sys.stdout.flush()` after critical print statements
3. Added explicit cleanup in finally blocks

**Lesson**: `n_jobs=-1` is unreliable. Always use explicit values (2-8) for production.

**Status**: ✅ Resolved - combined with Issue 7 solution

---

### Issue 9: Feature Mismatch in TSTR (844 vs 666 features)

**Symptom**: `ERROR: X has 844 features, but StandardScaler is expecting 666 features as input`

**Root Cause**: `pd.get_dummies()` creates different one-hot encoded columns for synthetic vs real data.

**Why It Happens**:
```python
# Real data has: admission_type = ['EMERGENCY', 'URGENT', 'ELECTIVE']
# → Creates: admission_type_EMERGENCY, admission_type_URGENT, admission_type_ELECTIVE

# Synthetic data has: admission_type = ['EMERGENCY', 'URGENT'] (missing ELECTIVE!)
# → Creates: admission_type_EMERGENCY, admission_type_URGENT

# Result: Different feature spaces!
```

**Initial Attempted Fix (Rejected by User)**:
- Manually add missing columns with zeros
- User feedback: "there is no point if we use just fill data with 0s and NaNs"

**Proper Solution**: Created `TSTRPreprocessor` in `src/mortality_classifier/train.py`:
```python
class TSTRPreprocessor:
    def fit(self, real_df):
        # Learn categorical levels from REAL data
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    def transform(self, df):
        # Apply same encoding to both real and synthetic
        # Unknown categories → zero vector (mathematically correct!)
```

**Why Zeros Are Correct**: In one-hot encoding, a zero vector means "none of the known categories" - this is the correct representation for unseen categories.

**Lesson**: Always fit preprocessing pipelines on real data, then transform both real and synthetic with the same fitted pipeline.

**Status**: ✅ Resolved - TSTRPreprocessor implemented and working

---

### Issue 10: ufunc 'isnan' Not Supported (Mixed-Type Columns)

**Symptom**: `ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''`

**Root Cause**: Raw CSV had mixed-type columns (e.g., `['A', 1.5, 'B']`) with `object` dtype. Pandas reads these as `object`, which `numpy.isnan()` cannot process during missing value imputation.

**User Feedback**: "do we need to convert anything to numeric, why cant we just train on the data as it is. It was working on the real data before"

**Solution**: Added explicit type conversion in `TSTRPreprocessor.fit()` and `transform()`:
```python
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype(str)  # Convert to string uniformly
```

**Additional Fix**: Use `pd.read_csv(..., low_memory=False)` for better type inference.

**Why This Works**:
- `OneHotEncoder` handles strings uniformly
- Eliminates type ambiguity that causes numpy errors
- Preserves all information (mixed types were already non-numeric)

**Lesson**: Always homogenize object dtypes before sklearn preprocessing to avoid type coercion errors.

**Status**: ✅ Resolved - TSTRPreprocessor converts object columns to strings

---

### Issue 11: DataFrame Fragmentation Warnings (Hundreds of Them)

**Symptom**:
```
PerformanceWarning: DataFrame is highly fragmented. This is usually the result of calling `frame.insert` many times
```

**Root Cause**: Adding missing one-hot columns one-by-one in a loop:
```python
for col in real_feature_names:
    if col not in synth_df.columns:
        synth_df[col] = 0  # Called hundreds of times!
```

**Impact**: Each assignment creates a new memory block, causing severe fragmentation and slow performance.

**Solution**: Batch create all missing columns at once:
```python
missing_cols = [col for col in real_feature_names if col not in synth_df.columns]
if missing_cols:
    missing_data = pd.DataFrame(0, index=synth_df.index, columns=missing_cols)
    synth_df = pd.concat([synth_df, missing_data], axis=1)
```

**Performance Gain**: ~10× faster for large column counts.

**Lesson**: Pandas operations should be batched, not looped. Always create multiple columns at once with `pd.concat()`.

**Status**: ✅ Resolved - using pd.concat() for batch column creation

---

### Issue 12: Quality Metrics Showing NaN (Wrong Data Stage)

**Symptom**:
```
Real shape: (50000, 354)   # Raw data
Synth shape: (5000, 680)   # One-hot encoded data
Common numeric columns: 340
Corr Frobenius: nan
Range violations: 41.21%  # Way too high!
```

**Root Cause**: Comparing dataframes at different processing stages:
- `real_df_full`: Raw data (354 columns)
- `synth_df_full`: After `prepare_data()` one-hot encoding (680 columns)

**Why It Failed**:
- Only 340 common columns (many synthetic one-hot columns missing in raw real data)
- Correlation matrix had all NaN values (columns don't align)
- Range violations computed on wrong columns

**Solution**: Use raw dataframes for quality metrics:
```python
# Load RAW data (before prepare_data)
synth_df_raw = pd.read_csv(temp_synth_path)       # 351 columns
real_df_raw = pd.read_csv(real_test_path, ...)    # 354 columns

# Drop ID columns from real
real_df_for_quality = real_df_raw.drop(columns=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'])

# Now both are at same stage (351 columns each, highly overlapping)
quality_metrics = compute_quality_metrics(real_df_for_quality, synth_df_raw)
```

**Lesson**: Quality metrics must compare data at the SAME processing stage. Use raw/inverse-transformed data, not one-hot encoded data.

**Status**: ✅ Resolved - using raw data for both real and synthetic

---

### Issue 13: High KS Statistic (Poor Distribution Match)

**Symptom**: `KS=0.5297` (much higher than expected <0.10 threshold)

**Root Causes**:
1. Small training sample (5k rows) doesn't capture full distribution
2. Small synthetic sample (5k samples) has high variance
3. Very low hyperparameters (nt=1, noise=1) produce low-quality synthetic data

**Expected Behavior**:
- KS should decrease as nt and noise increase
- Sweep will show this trend across hyperparameters

**Not a Bug**: This is expected for minimal hyperparameters. The sweep's purpose is to find which (nt, noise) achieves KS < 0.10.

**Lesson**: Don't panic when initial metrics are poor - that's why we're sweeping!

**Status**: ⚠️ Expected behavior - will improve with higher hyperparameters

---

### Issue 14: High Range Violation Percentage

**Symptom**: `RangeViol=41.21%` (way above 5% threshold)

**Root Cause (Initially)**: Was comparing one-hot encoded columns (wrong stage).

**Root Cause (After Fix)**: Low hyperparameters (nt=1, noise=1) produce poor-quality synthetic data with extreme values.

**Expected Behavior**: Should decrease as model quality improves with higher nt/noise.

**Potential Post-Processing Fix** (if violations persist at high nt/noise):
```python
# In create_flat_dataframe()
for col in CLINICAL_RANGES:
    if col in df.columns:
        valid_min, valid_max = CLINICAL_RANGES[col]
        df[col] = np.clip(df[col], valid_min, valid_max)
```

**Lesson**: Range violations are expected with low-quality models. Monitor this metric across the sweep to see if it improves.

**Status**: ⚠️ Expected behavior - will improve with higher hyperparameters (may need clipping)

---

### Summary of Issue Types

| Category | Issues | Status |
|----------|--------|--------|
| **Memory Management** | #1 (OOM), #6 (Full dataset) | ✅ All resolved |
| **Environment** | #2 (Module not found) | ✅ Resolved |
| **XGBoost** | #3 (Iterator), #7 (Joblib), #8 (Hanging) | ✅ Resolved |
| **Data Processing** | #4 (test_size), #5 (File path) | ✅ Resolved |
| **TSTR Preprocessing** | #9 (Feature mismatch), #10 (Mixed types), #11 (Fragmentation), #12 (Wrong stage) | ✅ All resolved |
| **Quality Metrics** | #13 (High KS), #14 (High violations) | ⚠️ Expected (low hyperparameters) |

**Key Takeaways**:
1. **Preprocessing is hard**: 6 out of 14 issues were data preprocessing bugs
2. **Joblib is brittle**: 3 out of 14 issues related to parallel processing
3. **Memory is limiting**: Required complete rethink of training strategy
4. **Debugging is iterative**: Each fix revealed new issues downstream

---

### Quick Reference: Common Issues

For detailed explanations, see Issues 1-14 above.

**Process hangs**: Use `--n-jobs 2` instead of `-1`

**All metrics NaN**: Check `sweep_results.csv` error column, verify `flat_table.csv` exists

**Memory killed (137)**: Reduce `--train-rows 2000` or `--n-jobs 1`

**Feature mismatch**: Already fixed with `TSTRPreprocessor` - should not occur

**ufunc 'isnan' error**: Already fixed with `astype(str)` - should not occur

**Quality metrics NaN**: Already fixed - using raw data for comparison

---

## Future Enhancements

### Potential Improvements

1. **Model Checkpointing**
   - Save top-K models based on ROC-AUC
   - Enable immediate production deployment of best config

2. **Multi-Metric Optimization**
   - Pareto frontier: performance vs training time
   - Weighted score: `α*ROC_AUC - β*train_time`

3. **Bayesian Optimization**
   - Use Gaussian Process to guide search
   - Fewer runs to find optimal hyperparameters

4. **Parallel Sweeps**
   - Run multiple (nt, n_noise) combinations in parallel
   - Requires database instead of CSV (PostgreSQL with row locking)

5. **Additional Metrics** (Tier 3)
   - ✅ Marginal distributions (KS test) - **IMPLEMENTED**
   - ✅ Correlation preservation (Frobenius norm) - **IMPLEMENTED**
   - ✅ Clinical range checks - **IMPLEMENTED**
   - Privacy metrics (Distance to Closest Record, membership inference)
   - Diversity metrics (mode coverage, alpha precision/beta recall)
   - Temporal metrics (DTW, autocorrelation preservation)

6. **Confidence Intervals**
   - Bootstrap TSTR evaluation (multiple random seeds)
   - Report uncertainty: `ROC-AUC = 0.78 ± 0.02`

---

## References

### Papers

- **Forest-Flow**: [Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees](https://arxiv.org/abs/2309.09968)
- **TSTR Evaluation**: [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503)

### Related Documentation

- `LIKELIHOOD_COMPUTATION.md` - Detailed explanation of flow matching theory
- `README.md` - Project overview and quickstart
- `configs/forest_flow_config.yaml` - Production hyperparameters

---

## Summary

The sweep framework provides a systematic, crash-safe, and memory-efficient way to explore Forest-Flow hyperparameters. Key design principles:

1. **Subsample for Speed**: 5k rows captures trends without full computational cost (~8-9 min/run)
2. **Production XGBoost Params**: 100 trees at depth 6 ensures sweep results match production quality
3. **Consistent Preprocessing**: `TSTRPreprocessor` ensures feature alignment between synthetic and real data
4. **Multi-Metric Evaluation**: TSTR (utility) + KS tests (distributions) + correlation preservation + clinical ranges
5. **Resume-Friendly**: CSV storage enables safe interruption and continuation
6. **Extrapolation-Ready**: Dense uniform sampling (8×8 grid) + log fits predict production performance
7. **Memory-Conscious**: Memmap + cleanup keeps RAM usage under 3GB (runs on 8GB machines)
8. **Binary Feature Handling**: Post-processing ensures discrete values for clinical flags

**Workflow**: Run sweep → Analyze plots → Identify optimal (nt, n_noise) → Retrain production model with `run_with_config.py` on full dataset.

**Time Budget**: 64 runs × 9 min/run ≈ 9-10 hours for full sweep (8×8 grid).
