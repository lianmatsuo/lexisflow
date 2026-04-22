# Data Processing (Implemented)

This document describes only the dataset preparation that is implemented in the
current scripts.

## Supported Dataset Pipelines

- MIMIC path:
  - `scripts/mimic/prepare_hour0.py`
  - `scripts/mimic/prepare_autoregressive.py`
- Challenge 2012 path:
  - `scripts/challenge2012/prepare_hour0.py`
  - `scripts/challenge2012/prepare_autoregressive.py`

Both are orchestrated by:

```bash
uv run python scripts/run_sweep.py --dataset <mimic|challenge2012>
```

## MIMIC Processing

### Input

- `data/processed/flat_table.csv`

### Hour-0 Preparation

`scripts/mimic/prepare_hour0.py`:

1. flatten column names
2. optionally drop cached fully-missing columns
3. drop configured datetime columns (`DEFAULT_DATETIME_COLUMNS`)
4. apply shared feature-pruning policy
5. keep hour-0 rows (`hours_in == 0`, fallback first row per patient)
6. detect static/dynamic columns via `split_static_dynamic`
7. impute missing values
8. optional diagnosis bucketing
9. write `data/processed/hour0_data.csv`

### Autoregressive Preparation

`scripts/mimic/prepare_autoregressive.py`:

1. find fully-missing columns (DuckDB scan + cache file)
2. drop datetime columns
3. detect static/dynamic groups
4. build lag-1 autoregressive pairs via `prepare_autoregressive_data`
5. split patients into train/test/holdout (80/10/10, patient-disjoint)
6. write:
   - `data/processed/autoregressive_data.csv` (train split)
   - `data/processed/real_test.csv`
   - `data/processed/real_holdout.csv`

## Challenge 2012 Processing

### Inputs

- `data/challenge2012/raw/set-a/*.txt`
- `data/challenge2012/raw/Outcomes-a.txt`

### Autoregressive Preparation

`scripts/challenge2012/prepare_autoregressive.py`:

1. parse per-patient text files into 48-hour grid
2. keep static fields (`Age`, `Gender`, `Height`, `Weight`, `ICUType`)
3. drop configured low-coverage variables
4. impute dynamic signals (ffill/bfill within patient, then global median)
5. build lag-1 autoregressive pairs via shared helper
6. attach labels (`hospital_expire_flag`, `los_icu`)
7. split patients into train/test/holdout (80/10/10)
8. write:
   - `data/challenge2012/processed/autoregressive_data.csv`
   - `data/challenge2012/processed/real_test.csv`
   - `data/challenge2012/processed/real_holdout.csv`

## Shared Preprocessor Fitting

After preparation, both datasets run:

- `fit_hour0_preprocessor.py`
- `fit_autoregressive_preprocessor.py`

Those artifacts are consumed by training/sweep stages.
