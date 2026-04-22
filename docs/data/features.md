# Feature Semantics (Implemented)

This page documents how features are handled in current preprocessing code.

## Static vs Dynamic Split

The split is determined by code, not hardcoded markdown lists:

- helper: `lexisflow.data.split_static_dynamic`
- callers:
  - `scripts/mimic/prepare_hour0.py`
  - `scripts/mimic/prepare_autoregressive.py`

## Autoregressive Layout

Autoregressive rows are built with lag-1 pairs using:

- `lexisflow.data.prepare_autoregressive_data`

Conceptually:

- targets: current-step dynamic features
- conditions: static features + lagged dynamic features (`*_lag1`)

This structure is used by sweep training in `src/lexisflow/sweep/training.py`.

## IDs and Time Columns

Core grouping columns:

- `subject_id`
- `hours_in`

MIMIC also carries additional IDs where available
(`hadm_id`, `icustay_id`) in processed CSVs.

ID columns are preserved in CSV artifacts for grouping and bookkeeping and are
not treated as predictive targets.

## Datetime and Pruning Policy

MIMIC preprocessing drops configured datetime columns:

- constant list: `lexisflow.data.DEFAULT_DATETIME_COLUMNS`

MIMIC preprocessing also applies shared feature pruning:

- function: `lexisflow.data.columns_to_drop_default_feature_pruning`

## Binary/Categorical/Numeric Handling

Feature typing utilities live in:

- `src/lexisflow/data/feature_utils.py`

Transformers and model preprocessing are implemented in:

- `src/lexisflow/data/transformers.py`

## Outcome Labels Used by Evaluation

The current TSTR framework evaluates:

- mortality (`hospital_expire_flag`)
- ICU LOS (`los_icu` converted to categories in the evaluation stack)

Implementation:

- `src/lexisflow/evaluation/tstr_framework.py`
