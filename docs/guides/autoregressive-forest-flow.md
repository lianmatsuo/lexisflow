# Autoregressive Forest-Flow: Complete Guide

**Quick Navigation:**
- [Quick Start](#quick-start)
- [What Changed](#what-changed)
- [Core Concepts](#core-concepts)
- [Implementation Details](#implementation-details)
- [Working with MIMIC-III Data](#working-with-mimic-iii-data)
- [Complete Workflow](#complete-workflow)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

---

## Quick Start

### 5-Minute Example

```python
from src.forest_flow import (
    ForestFlow, TabularPreprocessor,
    prepare_autoregressive_data,
    sample_trajectory,
    split_static_dynamic
)
import pandas as pd

# 1. Load data and identify static vs dynamic columns
df = pd.read_csv("data/processed/flat_table.csv")
static_cols, dynamic_cols = split_static_dynamic(df)

# 2. Create autoregressive dataset with lag features
df_ar, target_cols, condition_cols = prepare_autoregressive_data(
    df, 'subject_id', 'hours_in', static_cols=static_cols, lag=1
)

# 3. Preprocess
preprocessor = TabularPreprocessor(numeric_cols=..., categorical_cols=...)
preprocessor.fit(df_ar[target_cols + condition_cols])
X_all = preprocessor.transform(df_ar[target_cols + condition_cols])

X_target = X_all[:, :len(target_cols)]
X_condition = X_all[:, len(target_cols):]

# 4. Train conditional model
model = ForestFlow(nt=50, n_noise=100)
model.fit(X_target, X_condition=X_condition)

# 5. Generate trajectories
trajectories = sample_trajectory(
    model, preprocessor,
    n_trajectories=100, max_hours=48,
    target_cols=target_cols, condition_cols=condition_cols
)
```

### Run Demo Scripts

```bash
# Quick test (30 seconds)
python examples/autoregressive_quickstart.py

# Full MIMIC-III demo
python scripts/demo_autoregressive_forest_flow.py
```

---

## What Changed

### Before: IID Mode
- Learns: `P(X)` - independent samples
- Generates: Random rows with no temporal relationship
- Use case: Single snapshots, cross-sectional data

### After: Autoregressive Mode
- Learns: `P(X_t | X_{t-1}, Static_Features)` - conditional sequences
- Generates: Complete patient trajectories with temporal coherence
- Use case: Time series, patient trajectories over hours/days

### New Features

1. **Data Preparation**
   - `prepare_autoregressive_data()` - Creates lag features
   - `load_static_columns()` - Loads MIMIC-III static columns
   - `split_static_dynamic()` - Automatically splits columns

2. **Conditional Training**
   - `ForestFlow.fit(X_target, X_condition)` - Trains with conditioning
   - Noise applied only to targets, conditions kept clean

3. **Trajectory Sampling**
   - `sample_trajectory()` - Generates full patient sequences
   - Each timestep conditions on previous timestep
   - Variable-length with discharge detection

---

## Core Concepts

### 1. Two Types of "Time"

**Critical: These are completely different!**

#### Flow Time (t ∈ [0, 1])
- Internal diffusion/flow parameter
- Used during ODE integration
- Discretized into `nt` levels (e.g., 0.02, 0.04, ..., 1.0)
- **NOT** related to patient timeline

#### Patient Time (hours)
- Real temporal sequence: Hour 0, 1, 2, 3...
- Used in autoregressive generation
- Each hour conditions on previous hour

**Variable naming convention:**
- `t`, `t_index`, `t_levels` → Flow time
- `hour`, `hours_in`, `timestep` → Patient time

### 2. Conditioning Strategy

**IID Mode (Original):**
```python
X_t = (1 - t) * Z + t * X      # All features noisy
Y_t = X - Z
model.fit(X_t, Y_t)
```

**Conditional Mode (Autoregressive):**
```python
X_t_noisy = (1 - t) * Z + t * X_target     # Only targets noisy
X_t_input = [X_t_noisy, X_condition]        # Append clean condition
Y_t = X_target - Z

model.fit(X_t_input, Y_t)
```

**Why keep conditions clean?**
1. Represents known information (history/static features)
2. Improves training stability
3. Theoretically correct for conditional flow matching
4. Better trajectory coherence

### 3. Static vs Dynamic Features

**Static Features** (28 columns in MIMIC-III):
- Patient characteristics that don't change over time
- Examples: age, gender, ethnicity, admission_type
- Loaded from `data/processed/static_data.csv`
- Carried through entire trajectory

**Dynamic Features** (varies by preprocessing):
- Time-varying measurements
- Examples: heart rate, blood pressure, lab values, medications
- Change at each timestep
- These are the targets we generate

### 4. The First Timestep Problem

**Challenge:** Hour 0 has no history (no X_{-1}).

**Solutions:**

1. **Special fill value** (default):
   ```python
   prepare_autoregressive_data(..., fill_strategy='special', fill_value=-1.0)
   ```
   Lag columns filled with -1 for first timestep.

2. **Mean imputation**:
   ```python
   prepare_autoregressive_data(..., fill_strategy='mean')
   ```
   Use column means for missing lags.

3. **Separate initial model** (recommended for production):
   ```python
   # Train IID model for Hour 0 only
   df_hour0 = df[df['hours_in'] == 0]
   model_initial = ForestFlow(...)
   model_initial.fit(X_hour0)

   # Sample initial states
   X_0 = model_initial.sample(n_trajectories)

   # Use in trajectory generation
   trajectories = sample_trajectory(..., initial_state=X_0)
   ```

---

## Implementation Details

### Architecture Changes

#### 1. Data Preprocessing (`src/forest_flow/preprocessing.py`)

**New Function: `prepare_autoregressive_data()`**

Transforms flat time series into autoregressive format:

**Input:**
```
subject_id, hours_in, hr, bp, age
1,          0,        80, 120, 65
1,          1,        82, 118, 65
1,          2,        79, 115, 65
```

**Output:**
```
hours_in, hr, bp, age, hr_lag1, bp_lag1
0,        80, 120, 65, -1,      -1       # no history
1,        82, 118, 65, 80,      120      # condition on t=0
2,        79, 115, 65, 82,      118      # condition on t=1
```

**Parameters:**
- `id_col`: Patient ID column (e.g., 'subject_id')
- `time_col`: Time column (e.g., 'hours_in')
- `static_cols`: List of static feature names
- `lag`: Number of timesteps to lag (default=1)
- `fill_strategy`: How to handle first timestep ('special', 'mean', 'zero')
- `fill_value`: Value for special fill (default=-1.0)

**Returns:**
- `df_autoregressive`: DataFrame with lag columns
- `target_cols`: List of target column names
- `condition_cols`: List of condition column names (static + lags)

#### 2. Model Training (`src/forest_flow/model.py`)

**Modified: `FlowMatchingDataIterator`**
- Now accepts `X_condition` parameter
- Noise applied only to `X_target`
- Returns `[Noisy_Target, Clean_Condition]` as input

**Modified: `ForestFlow.fit()`**
```python
model.fit(X_target, X_condition=X_condition)
```
- `X_target`: Features to generate (dynamic features)
- `X_condition`: Conditioning features (static + lag)
- Automatically detects conditional vs IID mode

**Modified: `ForestFlow.sample()`**
```python
X_synth = model.sample(n_samples, X_condition=X_condition)
```
- Requires condition if model trained conditionally
- Validates condition shape
- Concatenates condition at each flow time step

#### 3. Trajectory Sampling (`src/forest_flow/sampling.py`)

**Main Function: `sample_trajectory()`**

**Algorithm:**
```
1. Generate initial state (Hour 0)
   - Condition = [Static, DummyLag=-1]
   - X_0 = model.sample(1, X_condition)

2. For hour = 1 to max_hours:
   - Condition = [Static, X_{hour-1}]
   - X_hour = model.sample(1, X_condition)
   - Check discharge flag (optional)
   - If discharged, break

3. Return list of trajectory DataFrames
```

**Parameters:**
- `model`: Fitted conditional ForestFlow
- `preprocessor`: Fitted TabularPreprocessor
- `static_features`: Static features for each trajectory (n_traj, n_static)
- `n_trajectories`: Number of trajectories to generate
- `max_hours`: Maximum timesteps per trajectory
- `target_cols`: List of target column names
- `condition_cols`: List of condition column names
- `initial_strategy`: 'random' or 'zero'
- `discharge_col`: Optional discharge indicator column
- `discharge_threshold`: Threshold for early stopping
- `random_state`: Random seed

**Returns:**
- List of DataFrames, one per trajectory
- Each has columns: [timestep, target_features...]

---

## Working with MIMIC-III Data

### Understanding static_data.csv

Your MIMIC-III data has **28 static columns** in `data/processed/static_data.csv`:

#### Demographics (3)
- `gender`: M/F
- `ethnicity`: WHITE, BLACK, ASIAN, HISPANIC, etc.
- `age`: Age at admission (float)

#### Admission Information (4)
- `insurance`: Medicare, Medicaid, Private, Government
- `diagnosis_at_admission`: Primary diagnosis text
- `admission_type`: EMERGENCY, ELECTIVE, URGENT
- `first_careunit`: Initial ICU (MICU, SICU, CCU, etc.)

#### Outcomes (4)
- `mort_icu`: ICU mortality (0/1)
- `mort_hosp`: Hospital mortality (0/1)
- `hospital_expire_flag`: Expired during hospitalization (0/1)
- `readmission_30`: 30-day readmission (0/1)

#### Code Status (9)
- `fullcode_first`, `fullcode`: Full code status
- `dnr_first`, `dnr`: Do not resuscitate
- `cmo_first`, `cmo`, `cmo_last`: Comfort measures only
- `dnr_first_charttime`, `timecmo_chart`: Timestamps

#### Timing (8)
- `admittime`, `intime`, `outtime`, `dischtime`, `deathtime`
- `los_icu`: Length of stay in ICU (days)

#### Discharge (2)
- `dischtime`: Discharge timestamp
- `discharge_location`: HOME, SNF, REHAB, DEAD/EXPIRED, etc.

#### Other (1)
- `hospstay_seq`: Hospital stay sequence number

### Using Utility Functions

#### Load Static Columns
```python
from src.forest_flow import load_static_columns

static_cols = load_static_columns()
# Returns: ['gender', 'ethnicity', 'age', 'insurance', ...]
print(f"Found {len(static_cols)} static columns")
```

#### Split Static vs Dynamic
```python
from src.forest_flow import split_static_dynamic

df = pd.read_csv("data/processed/flat_table.csv")
static_cols, dynamic_cols = split_static_dynamic(df)

print(f"Static: {len(static_cols)}")
print(f"Dynamic: {len(dynamic_cols)}")
```

#### Get Column Information
```python
from src.forest_flow import get_mimic_column_info

info = get_mimic_column_info()
print("Demographics:", info['static']['demographics'])
# ['gender', 'ethnicity', 'age']

print("Admission:", info['static']['admission_info'])
# ['insurance', 'diagnosis_at_admission', 'admission_type', 'first_careunit']
```

### Handling Categorical Encoding

Static categorical columns get one-hot encoded, which increases dimensionality:

**Original:** `gender` (1 column)
**After encoding:** `gender_M`, `gender_F` (2 columns)

When extracting static features for sampling, calculate preprocessed dimensions:

```python
n_static_preprocessed = 0
for col in static_cols:
    if col in numeric_cols:
        n_static_preprocessed += 1
    elif col in categorical_cols:
        prefix = f"{col}_"
        n_static_preprocessed += len([
            c for c in preprocessor.dummy_columns
            if c.startswith(prefix)
        ])

# Extract static features
static_features = X_condition[:n_trajectories, :n_static_preprocessed]
```

---

## Complete Workflow

### Step-by-Step with MIMIC-III Data

```python
import pandas as pd
from tqdm import tqdm
from src.forest_flow import (
    ForestFlow,
    TabularPreprocessor,
    prepare_autoregressive_data,
    sample_trajectory,
    split_static_dynamic,
)

# ===================================================================
# Step 1: Load data and identify column types
# ===================================================================
print("Step 1: Loading data...")
df = pd.read_csv("data/processed/flat_table.csv")

# Automatically identify static vs dynamic
static_cols, dynamic_cols = split_static_dynamic(df)
print(f"  Static: {len(static_cols)} columns")
print(f"  Dynamic: {len(dynamic_cols)} columns")

# ===================================================================
# Step 2: Create autoregressive dataset
# ===================================================================
print("\nStep 2: Preparing autoregressive data...")
df_ar, target_cols, condition_cols = prepare_autoregressive_data(
    df,
    id_col='subject_id',
    time_col='hours_in',
    static_cols=static_cols,
    lag=1,
    fill_strategy='special',
    fill_value=-1.0
)

print(f"  Target columns: {len(target_cols)} (dynamic features)")
print(f"  Condition columns: {len(condition_cols)} (static + lag)")

# ===================================================================
# Step 3: Identify preprocessing column types
# ===================================================================
print("\nStep 3: Identifying column types...")
numeric_cols = []
categorical_cols = []
int_cols = []

all_cols = target_cols + condition_cols

for col in tqdm(all_cols, desc="  Analyzing columns"):
    if col not in df_ar.columns:
        continue

    if df_ar[col].dtype == "object":
        if df_ar[col].nunique() < 50:
            categorical_cols.append(col)
    else:
        numeric_cols.append(col)
        if df_ar[col].nunique() <= 10:
            int_cols.append(col)

print(f"  Numeric: {len(numeric_cols)}")
print(f"  Categorical: {len(categorical_cols)}")

# ===================================================================
# Step 4: Preprocess
# ===================================================================
print("\nStep 4: Preprocessing...")
preprocessor = TabularPreprocessor(
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    int_cols=int_cols,
    max_missing_ratio=None,  # XGBoost handles NaNs
)

preprocessor.fit(df_ar[all_cols])
X_all = preprocessor.transform(df_ar[all_cols])

# Split into target and condition
n_target = len(target_cols)
X_target = X_all[:, :n_target]
X_condition = X_all[:, n_target:]

print(f"  X_target shape: {X_target.shape}")
print(f"  X_condition shape: {X_condition.shape}")

# ===================================================================
# Step 5: Train conditional model
# ===================================================================
print("\nStep 5: Training conditional Forest-Flow...")
model = ForestFlow(
    nt=50,
    n_noise=100,
    n_jobs=-1,
    use_data_iterator=True,
    random_state=42
)

model.fit(X_target, X_condition=X_condition)
print("  Training complete!")

# ===================================================================
# Step 6: Prepare static features for sampling
# ===================================================================
print("\nStep 6: Preparing for sampling...")

# Calculate static feature dimensions after preprocessing
n_static_preprocessed = 0
for col in static_cols:
    if col in numeric_cols:
        n_static_preprocessed += 1
    elif col in categorical_cols:
        prefix = f"{col}_"
        n_static_preprocessed += len([
            c for c in preprocessor.dummy_columns
            if c.startswith(prefix)
        ])

print(f"  Static features (preprocessed): {n_static_preprocessed} dims")

# Extract static features from existing data or create new
n_trajectories = 100
static_features = X_condition[:n_trajectories, :n_static_preprocessed]

# ===================================================================
# Step 7: Generate trajectories
# ===================================================================
print("\nStep 7: Generating trajectories...")
trajectories = sample_trajectory(
    model=model,
    preprocessor=preprocessor,
    static_features=static_features,
    n_trajectories=n_trajectories,
    max_hours=48,
    target_cols=target_cols,
    condition_cols=condition_cols,
    initial_strategy='random',
    discharge_col=None,  # Optional: 'discharged' if you have it
    random_state=42,
    verbose=True
)

print(f"\n  Generated {len(trajectories)} trajectories")
print(f"  Trajectory lengths: {[len(t) for t in trajectories[:5]]}")

# ===================================================================
# Step 8: Inspect and save results
# ===================================================================
print("\nStep 8: Saving results...")

# View first trajectory
print("\nFirst trajectory (first 10 hours, first 5 columns):")
print(trajectories[0].head(10).iloc[:, :5])

# Convert to flat format
from src.forest_flow import prepare_training_data_from_trajectories
df_synth = prepare_training_data_from_trajectories(trajectories)

# Save
df_synth.to_csv("results/synthetic_trajectories.csv", index=False)
print(f"\nSaved {len(df_synth)} records to results/synthetic_trajectories.csv")
```

---

## API Reference

### Data Preparation

#### `prepare_autoregressive_data(df, id_col, time_col, static_cols, lag=1, fill_strategy='special', fill_value=-1.0)`

Creates lagged features for autoregressive modeling.

**Returns:** `(df_autoregressive, target_cols, condition_cols)`

#### `load_static_columns(static_data_path='data/processed/static_data.csv')`

Loads list of static column names from MIMIC-III static_data.csv.

**Returns:** `List[str]`

#### `split_static_dynamic(df, static_data_path='data/processed/static_data.csv')`

Splits DataFrame columns into static and dynamic lists.

**Returns:** `(static_cols, dynamic_cols)`

#### `get_mimic_column_info(static_data_path='data/processed/static_data.csv')`

Returns categorized dictionary of MIMIC-III columns.

**Returns:** `Dict[str, Any]`

### Model Training

#### `ForestFlow(nt=50, n_noise=100, n_jobs=-1, use_data_iterator=True, ...)`

Flow-matching generative model with conditional support.

**Methods:**
- `fit(X_target, X_condition=None)` - Train model
- `sample(n_samples, X_condition=None, random_state=None)` - Generate samples

### Trajectory Generation

#### `sample_trajectory(model, preprocessor, static_features=None, n_trajectories=1, max_hours=48, ...)`

Generates autoregressive patient trajectories.

**Returns:** `List[pd.DataFrame]` - One DataFrame per trajectory

#### `prepare_training_data_from_trajectories(trajectories, id_col='trajectory_id', time_col='timestep')`

Converts trajectory list back to flat DataFrame.

**Returns:** `pd.DataFrame`

---

## Troubleshooting

### Error: "Model was trained conditionally, must provide X_condition"

**Cause:** Trying to sample without condition data from conditional model.

**Fix:**
```python
X_synth = model.sample(n_samples, X_condition=X_condition)
```

### Error: "X_condition shape mismatch: expected (n, d), got (n, k)"

**Cause:** Wrong number of condition features.

**Fix:** Properly calculate static feature dimensions after preprocessing:
```python
n_static_preprocessed = sum([
    1 if col in numeric_cols else
    len([c for c in preprocessor.dummy_columns if c.startswith(f"{col}_")])
    for col in static_cols
])
static_features = X_condition[:n_samples, :n_static_preprocessed]
```

### Trajectories look incoherent/jumpy

**Possible causes & solutions:**

1. **Too few flow time steps**
   - Solution: Increase `nt` (try 50-100)

2. **Too few noise samples**
   - Solution: Increase `n_noise` (try 100-500)

3. **Missing static features**
   - Solution: Ensure all static columns from static_data.csv are included

4. **Wrong lag features**
   - Solution: Verify condition_cols include lag columns

5. **Poor initial state**
   - Solution: Train separate model for Hour 0

### First timestep looks unrealistic

**Cause:** No history at t=0, model trained on fill value (-1).

**Fix:** Train separate model for initial state:
```python
# Filter to hour 0 only
df_hour0 = df[df['hours_in'] == 0]

# Train IID model
model_initial = ForestFlow(...)
model_initial.fit(X_hour0)

# Sample initial states
X_0 = model_initial.sample(n_trajectories)

# Use in trajectory generation
trajectories = sample_trajectory(..., initial_state=X_0)
```

### Memory errors during training

**Solutions:**

1. Use data iterator mode (default):
   ```python
   ForestFlow(..., use_data_iterator=True)
   ```

2. Reduce batch size:
   ```python
   ForestFlow(..., batch_size=500)
   ```

3. Reduce n_noise:
   ```python
   ForestFlow(..., n_noise=50)
   ```

4. Train on subset of data:
   ```python
   df = pd.read_csv("data/processed/flat_table.csv", nrows=10000)
   ```

### Slow training

**Solutions:**

1. Enable parallel training:
   ```python
   ForestFlow(..., n_jobs=-1)
   ```

2. Use data iterator:
   ```python
   ForestFlow(..., use_data_iterator=True)
   ```

3. Reduce parameters for testing:
   ```python
   ForestFlow(nt=10, n_noise=10)  # Quick test
   ```

4. Reduce XGBoost complexity:
   ```python
   xgb_params = {
       'n_estimators': 50,  # Default 100
       'max_depth': 4,       # Default 6
   }
   ForestFlow(..., xgb_params=xgb_params)
   ```

---

## Examples

### Example 1: Quick Test with Synthetic Data

```python
# Create toy dataset
import numpy as np
import pandas as pd

data = []
for pid in range(10):
    age = np.random.randint(30, 80)
    hr_base = np.random.normal(80, 10)

    for hour in range(5):
        hr = hr_base + np.random.normal(0, 3)
        data.append({
            'patient_id': pid,
            'hour': hour,
            'heart_rate': hr,
            'age': age
        })

df = pd.DataFrame(data)

# Train and sample
from src.forest_flow import *

df_ar, target, condition = prepare_autoregressive_data(
    df, 'patient_id', 'hour', static_cols=['age'], lag=1
)

preprocessor = TabularPreprocessor(
    numeric_cols=['heart_rate', 'age', 'heart_rate_lag1'],
    categorical_cols=[]
)
preprocessor.fit(df_ar[target + condition])
X = preprocessor.transform(df_ar[target + condition])

X_target = X[:, :1]
X_condition = X[:, 1:]

model = ForestFlow(nt=5, n_noise=5)
model.fit(X_target, X_condition)

trajectories = sample_trajectory(
    model, preprocessor,
    n_trajectories=3, max_hours=10,
    target_cols=target, condition_cols=condition
)

print(trajectories[0])
```

### Example 2: Training on MIMIC-III Subset

```python
from src.forest_flow import *

# Load subset
df = pd.read_csv("data/processed/flat_table.csv", nrows=5000)

# Auto-detect columns
static_cols, dynamic_cols = split_static_dynamic(df)

# Prepare
df_ar, target, condition = prepare_autoregressive_data(
    df, 'subject_id', 'hours_in', static_cols, lag=1
)

# Quick column type detection
numeric = [c for c in target + condition if df_ar[c].dtype != 'object']
categorical = [c for c in target + condition if df_ar[c].dtype == 'object']

# Preprocess
preprocessor = TabularPreprocessor(numeric, categorical)
preprocessor.fit(df_ar[target + condition])
X = preprocessor.transform(df_ar[target + condition])

X_target = X[:, :len(target)]
X_condition = X[:, len(target):]

# Train
model = ForestFlow(nt=10, n_noise=10, n_jobs=-1)
model.fit(X_target, X_condition)

# Sample
trajectories = sample_trajectory(
    model, preprocessor,
    n_trajectories=10, max_hours=24,
    target_cols=target, condition_cols=condition
)
```

### Example 3: Production Settings

```python
# High-quality settings for production
model = ForestFlow(
    nt=100,              # More flow time levels
    n_noise=500,         # More noise samples
    n_jobs=-1,           # Parallel training
    use_data_iterator=True,  # Memory efficient
    batch_size=1000,
    xgb_params={
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
    },
    random_state=42
)

# Train on full data
model.fit(X_target, X_condition)

# Generate many trajectories
trajectories = sample_trajectory(
    model, preprocessor,
    n_trajectories=1000,
    max_hours=72,
    target_cols=target_cols,
    condition_cols=condition_cols,
    random_state=42,
    verbose=True
)
```

---

## Performance Tips

1. **Use data iterator mode** (default)
   - Avoids n_noise duplication
   - Much more memory efficient

2. **Parallel training** (`n_jobs=-1`)
   - Trains multiple flow time levels simultaneously

3. **Batch size tuning**
   - Larger = faster but more memory
   - Default 1000 works well

4. **Start small, scale up**
   - Test with nt=5, n_noise=5
   - Production: nt=100, n_noise=500

5. **Profile your code**
   - Most time in XGBoost training
   - Consider reducing max_depth or n_estimators for speed

---

## Validation Checklist

Before running production experiments, verify:

- [ ] Static columns loaded from static_data.csv
- [ ] Lag columns created correctly
- [ ] First timestep has fill value (-1 or mean)
- [ ] Condition columns = static + lag
- [ ] Target columns = dynamic only
- [ ] Static feature dimensions calculated correctly
- [ ] Model trains without errors
- [ ] Trajectories have reasonable values
- [ ] Trajectories show temporal coherence
- [ ] No linter errors

---

## File Structure

```
synth-gen/
├── src/forest_flow/
│   ├── model.py              # ✓ Conditional training
│   ├── preprocessing.py      # ✓ prepare_autoregressive_data()
│   ├── sampling.py           # ✓ sample_trajectory()
│   ├── utils.py              # ✓ MIMIC-III helpers
│   └── __init__.py           # ✓ Exports all functions
├── examples/
│   └── autoregressive_quickstart.py   # Quick demo
├── scripts/
│   └── demo_autoregressive_forest_flow.py    # Full MIMIC-III demo
├── data/processed/
│   ├── static_data.csv       # 28 static columns
│   └── flat_table.csv        # Time series data
└── docs/guides/autoregressive-forest-flow.md  # This file
```

---

## References

1. **Original Paper**: "Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees" (Fonseca & Tabak, 2024)
   - https://arxiv.org/abs/2309.09968

2. **Flow Matching**: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
   - https://arxiv.org/abs/2210.02747

3. **MIMIC-III**: Johnson et al., "MIMIC-III, a freely accessible critical care database"
   - https://physionet.org/content/mimiciii/

---

## Summary

✅ **Implementation Complete**

Your Forest-Flow now generates realistic temporal patient trajectories by learning:
```
P(X_t | X_{t-1}, Static_Features)
```

**Key achievements:**
- Autoregressive trajectory generation
- Proper static column handling (28 columns from static_data.csv)
- Conditional training with clean history
- Variable-length trajectories
- Complete documentation and examples

**Next steps:**
1. Run `python examples/autoregressive_quickstart.py`
2. Try full demo: `python scripts/demo_autoregressive_forest_flow.py`
3. Scale up parameters for production (nt=100, n_noise=500)
4. Consider training separate model for initial state (Hour 0)

For questions or issues, refer to the Troubleshooting section above.
