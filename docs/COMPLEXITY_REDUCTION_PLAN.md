# Complexity Reduction Plan

## Problem Statement

The codebase has accumulated unnecessary complexity through:
1. **Tuple column names** like `('Heart Rate', 'mean')` requiring special handling everywhere
2. **Multiple data formats** (CSV, Parquet, memmap) with different code paths
3. **Inconsistent feature identification** (string vs tuple column matching)
4. **Redundant preprocessor saving** (multiple locations, compatibility layers)
5. **Complex sweep architecture** (memmap, column split files, multiple caches)

This complexity makes the code harder to maintain, test, and extend.

## Proposed Simplifications

---

## 1. Standardize Column Names to Strings

### Current Problem
```python
# Tuple columns from MIMIC-III aggregation
('Heart Rate', 'mean')
('Heart Rate', 'std')
('Temperature', 'min')
...

# Requires special handling everywhere
def is_lagged(col):
    if isinstance(col, str):
        return col.endswith('_lag1')
    elif isinstance(col, tuple):
        return col[0].endswith('_lag1')  # Awkward!
```

### Proposed Solution
Flatten tuple columns to strings **at the earliest stage** (01_preprocess_data.py):

```python
# Convert during initial preprocessing
df.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
              for col in df.columns]

# Result:
'Heart_Rate_mean'
'Heart_Rate_std'
'Temperature_min'
```

### Benefits
- ✅ Simpler column matching (just `col.endswith('_lag1')`)
- ✅ Easier to work with in pandas (no special indexing)
- ✅ More readable in CSV files
- ✅ Fewer edge cases in code

### Implementation
**File:** `scripts/01_preprocess_data.py`

Add after loading data:
```python
def flatten_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tuple columns to flat strings."""
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            new_cols.append('_'.join(str(x) for x in col))
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df

# Apply
df = pd.read_csv(input_path)
df = flatten_column_names(df)
```

### Backward Compatibility
- Update all `known_binary_features` lists
- Update clinical range dictionaries
- Run tests to verify no breakage

**Estimated effort:** ~2-3 hours

---

## 2. Remove Parquet/Memmap Code Paths

### Current Problem
Multiple code paths for different formats:
```python
if use_parquet:
    df = pd.read_parquet(path)
else:
    df = pd.read_csv(path)
```

### Proposed Solution
**Single format: CSV** (as user requested)

Remove all:
- `use_parquet` flags
- `if use_parquet:` branches
- Memmap logic in `run_sweep.py`
- Parquet-specific dependencies

### Benefits
- ✅ Single code path (easier to test)
- ✅ Fewer dependencies (no pyarrow for parquet)
- ✅ Simpler logic in all scripts

### Implementation
Already mostly complete from Task 1! Just need cleanup:
- Remove unused imports (`pyarrow.parquet`)
- Remove old memmap directory references
- Simplify sweep script (already updated)

**Estimated effort:** ~1 hour

---

## 3. Unify Preprocessor Storage

### Current Problem
```python
# Multiple save locations for "compatibility"
preprocessor_output_models = "models/preprocessor_full.pkl"
preprocessor_output_sweep = "packages/data/preprocessed/preprocessor_fitted.pkl"
```

### Proposed Solution
**Single canonical location:** `models/preprocessor_full.pkl`

All scripts read from same location:
```python
# scripts/02_train_model.py
preprocessor_path = "models/preprocessor_full.pkl"

# scripts/run_sweep.py
preprocessor_path = "models/preprocessor_full.pkl"

# scripts/03_generate_synthetic.py
preprocessor_path = "models/preprocessor_full.pkl"
```

### Benefits
- ✅ Single source of truth
- ✅ No confusion about which version to use
- ✅ Easier to track model artifacts

### Implementation
Already complete from Task 1!

**Estimated effort:** ✅ Done

---

## 4. Simplify Column Split Metadata

### Current Problem
```python
# Separate text file with custom parsing
split_path = data_dir / "column_split.txt"
with open(split_path) as f:
    lines = f.read().strip().split("\n")
    split_info = dict(line.split("=") for line in lines)
n_target = int(split_info["n_target"])
```

### Proposed Solution
**Include in preprocessor pickle:**

```python
# 01b_fit_preprocessor.py
preprocessor_data = {
    'preprocessor': preprocessor,
    'target_cols': target_cols,
    'condition_cols': condition_cols,
    'n_target': len(target_cols),  # Already included!
    'all_cols': all_cols,
    ...
}
```

Then loading is simpler:
```python
# run_sweep.py
with open(preprocessor_path, 'rb') as f:
    data = pickle.load(f)

preprocessor = data['preprocessor']
n_target = data['n_target']
target_cols = data['target_cols']
```

### Benefits
- ✅ Single file to manage (preprocessor.pkl)
- ✅ Type-safe (no string parsing)
- ✅ Atomic updates (can't have mismatched metadata)

### Implementation
Already complete from Task 1!

**Estimated effort:** ✅ Done

---

## 5. Consolidate Binary Feature Detection

### Current Problem
Binary feature logic scattered across scripts:
- `01b_fit_preprocessor.py` has hardcoded list
- Auto-detection logic duplicated
- `is_lagged()` helper defined locally

### Proposed Solution
**Centralize in utils module:**

```python
# packages/data/feature_utils.py
KNOWN_BINARY_FEATURES = [
    'vent', 'vaso', 'sedation', 'adenosine', 'dobutamine',
    'dopamine', 'epinephrine', 'isuprel', 'milrinone',
    'norepinephrine', 'phenylephrine', 'vasopressin',
    'colloid_bolus', 'crystalloid_bolus', 'nivdurations',
]

def is_binary_feature(col: str, df: pd.DataFrame, strict: bool = True) -> bool:
    """
    Identify binary features.

    Args:
        col: Column name (string only, no tuples)
        df: DataFrame sample to check
        strict: If True, require integer dtype AND only {0,1} values

    Returns:
        True if binary feature
    """
    # Check known list
    base_name = col.replace('_lag1', '')
    if base_name in KNOWN_BINARY_FEATURES:
        return True

    # Auto-detect
    if strict and col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_integer_dtype(df[col]):
                unique_vals = df[col].dropna().unique()
                if set(unique_vals).issubset({0, 1}):
                    return True

    return False


def identify_feature_types(
    df: pd.DataFrame,
    exclude_cols: list[str] = None,
) -> dict[str, list[str]]:
    """
    Identify numeric, binary, and categorical columns.

    Returns:
        {
            'numeric': [...],
            'binary': [...],
            'categorical': [...],
            'int': [...],
        }
    """
    ...
```

### Benefits
- ✅ Single source of truth for feature logic
- ✅ Reusable across scripts
- ✅ Easier to test in isolation
- ✅ Consistent behavior everywhere

**Estimated effort:** ~2-3 hours

---

## 6. Simplify Configuration

### Current Problem
- `configs/forest_flow_config.yaml` exists but is barely used
- Most parameters hardcoded in scripts
- Inconsistent parameter passing

### Proposed Solution
**Use YAML config consistently:**

```yaml
# configs/pipeline_config.yaml
data:
  input_path: "packages/data/processed/flat_table.csv"
  output_path: "packages/data/processed/autoregressive_data.csv"

preprocessing:
  chunk_size: 100000
  lag: 1

model:
  nt: 50
  n_noise: 100
  n_jobs: 8
  max_train_rows: 10000  # null for all

generation:
  n_samples: 1000
  max_hours: 48

sweep:
  nt_values: [10, 25, 50]
  noise_values: [10, 50, 100]
```

Then load in scripts:
```python
import yaml

with open('configs/pipeline_config.yaml') as f:
    config = yaml.safe_load(f)

nt = config['model']['nt']
n_noise = config['model']['n_noise']
```

### Benefits
- ✅ Single place to change parameters
- ✅ Easier to version control configs
- ✅ No need to edit Python code for experiments

**Estimated effort:** ~2-4 hours

---

## 7. Remove Legacy Files & Imports

### Current Problem
- Old `src/forest_flow/` directory deleted but references remain
- Multiple documentation files (some outdated)
- Unused imports and helper functions

### Proposed Solution
**Clean sweep:**

1. Remove from git:
```bash
git rm -r src/forest_flow/  # Already done
git rm docs/api/          # Outdated API docs
git rm scripts/demo_*.py  # Old demos
```

2. Remove legacy imports:
```python
# Search for any remaining absolute imports
rg "from synth_gen\." --type py
rg "from forest_flow\." --type py
```

3. Archive old documentation:
```
docs/
├── archive/           # NEW: Move old docs here
│   ├── api/
│   └── LIKELIHOOD_COMPUTATION.md
└── current/           # Only actively maintained docs
    ├── HOUR_0_MODEL_PLAN.md
    ├── ADDITIONAL_TSTR_TASKS.md
    └── ...
```

### Benefits
- ✅ Less confusion about what's current
- ✅ Cleaner codebase
- ✅ Faster searches (fewer files)

**Estimated effort:** ~1-2 hours

---

## Implementation Priority

### Phase 1: Quick Wins (Total: ~4-6 hours)
1. ✅ Remove Parquet code paths (mostly done)
2. ✅ Unify preprocessor storage (done)
3. 🟢 Clean up legacy files
4. 🟢 Remove unused imports

### Phase 2: Structural Improvements (Total: ~6-8 hours)
5. 🟡 Flatten tuple column names
6. 🟡 Centralize feature detection utilities
7. 🟡 Add feature_utils.py module

### Phase 3: Configuration (Total: ~2-4 hours)
8. 🟡 Migrate to YAML-based configuration
9. 🟡 Update all scripts to use config

## Testing Strategy

After each simplification:
1. Run existing tests: `uv run pytest`
2. Run end-to-end pipeline: `01_preprocess → 01b_fit → 02_train`
3. Verify outputs match previous format
4. Check no regressions in metrics

## Success Criteria

- ✅ All scripts use consistent paths
- ✅ No tuple column handling code
- ✅ Single format (CSV only)
- ✅ Single preprocessor location
- ✅ Centralized feature detection
- ✅ Configuration file driven
- ✅ All tests pass
- ✅ Documentation matches code

## Long-Term Benefits

1. **Maintainability:** 30-40% less code to maintain
2. **Onboarding:** Easier for new contributors to understand
3. **Testing:** Simpler logic → easier to test
4. **Performance:** Fewer branches → slightly faster
5. **Reliability:** Fewer edge cases → fewer bugs

## Risk Assessment

**Low risk** for most changes:
- Path standardization (✅ done, already working)
- Remove unused code (safe if tests pass)
- Centralize utilities (doesn't change behavior)

**Medium risk:**
- Flatten tuple columns (affects entire pipeline)
  - Mitigation: Do in separate branch, test thoroughly

**High risk:** None identified

## Conclusion

These simplifications will significantly improve code quality and maintainability without changing model behavior or performance. The changes are largely mechanical and can be tested systematically.

**For thesis submission:**
- Focus on Phase 1 (quick wins)
- Defer Phase 2-3 to post-thesis

**For production system:**
- Implement all phases
- Set up CI/CD to prevent complexity creep
