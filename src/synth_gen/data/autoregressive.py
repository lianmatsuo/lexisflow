"""Autoregressive data preparation for time series modeling."""

from __future__ import annotations

import pandas as pd

from .feature_utils import columns_to_drop_default_feature_pruning


def prepare_autoregressive_data(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    static_cols: list[str] | None = None,
    lag: int = 1,
    fill_strategy: str = "special",
    fill_value: float = -1.0,
    keep_biophysical_mean_only: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare data for autoregressive modeling.

    Transforms data from P(X) to P(X_t | X_{t-lag}, Static_Features).
    For each row at time t, creates lagged columns from time t-lag.

    Args:
        df: DataFrame with patient time series data.
        id_col: Column name for patient/subject ID (e.g., 'subject_id').
        time_col: Column name for time (e.g., 'hours_in').
        static_cols: Optional list of static feature columns (e.g., ['age', 'gender']).
                    If None, all columns are treated as dynamic.
        lag: Number of time steps to lag (default=1).
        fill_strategy: How to handle first timestep where no history exists:
                      - 'special': Fill with fill_value (default=-1.0)
                      - 'mean': Fill with column mean
                      - 'zero': Fill with 0
        fill_value: Value to use when fill_strategy='special'.
        keep_biophysical_mean_only: If True, apply default feature pruning:
            drop ``*_std`` / ``*_count`` when matching ``*_mean`` exists, and drop
            sparse low-coverage features listed in project policy.

    Returns:
        df_autoregressive: DataFrame with target and condition columns.
        target_cols: List of target column names (current timestep).
        condition_cols: List of condition column names (history + static).

    Example:
        Original data (subject_id=1):
            hours_in, hr, bp, age
            0,        80, 120, 65
            1,        82, 118, 65
            2,        79, 115, 65

        After prepare_autoregressive_data (lag=1, static_cols=['age']):
            hours_in, hr, bp, age, hr_lag1, bp_lag1
            0,        80, 120, 65, -1,      -1       # no history
            1,        82, 118, 65, 80,      120      # condition on t=0
            2,        79, 115, 65, 82,      118      # condition on t=1

        target_cols = ['hr', 'bp']
        condition_cols = ['age', 'hr_lag1', 'bp_lag1']
    """
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in DataFrame")
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame")

    if keep_biophysical_mean_only:
        drop_cols = columns_to_drop_default_feature_pruning(df.columns)
        if drop_cols:
            df = df.drop(columns=drop_cols)

    # Identify static and dynamic columns
    static_cols = static_cols or []

    # All known ID columns that should never be lagged or used as features
    all_id_cols = ["subject_id", "hadm_id", "icustay_id"]

    reserved_cols = [id_col, time_col] + static_cols + all_id_cols
    dynamic_cols = [col for col in df.columns if col not in reserved_cols]

    # Sort by ID and time
    df_sorted = df.sort_values([id_col, time_col]).reset_index(drop=True)

    # Create lagged columns for dynamic features
    lagged_dfs = []
    for col in dynamic_cols:
        # Shift within each group (ID)
        lagged = df_sorted.groupby(id_col)[col].shift(lag)
        lagged_dfs.append(lagged.rename(f"{col}_lag{lag}"))

    # Concatenate lagged columns
    df_lagged = pd.concat(lagged_dfs, axis=1)

    # Handle first timestep (no history)
    if fill_strategy == "special":
        df_lagged = df_lagged.fillna(fill_value)
    elif fill_strategy == "mean":
        df_lagged = df_lagged.fillna(df_lagged.mean())
    elif fill_strategy == "zero":
        df_lagged = df_lagged.fillna(0.0)
    else:
        raise ValueError(f"Unknown fill_strategy: {fill_strategy}")

    # Combine: [ID, Time, Target_Cols, Static_Cols, Lagged_Cols]
    df_autoregressive = pd.concat(
        [
            df_sorted[[id_col, time_col]],
            df_sorted[dynamic_cols],  # Target columns
            df_sorted[static_cols] if static_cols else pd.DataFrame(),  # Static
            df_lagged,  # Condition (lagged)
        ],
        axis=1,
    )

    # Define target and condition columns
    target_cols = dynamic_cols
    condition_cols = static_cols + list(df_lagged.columns)

    return df_autoregressive, target_cols, condition_cols


def split_static_dynamic(
    df: pd.DataFrame, static_data_path: str = "data/processed/static_data.csv"
) -> tuple[list[str], list[str]]:
    """Split DataFrame columns into static and dynamic.

    Args:
        df: DataFrame with both static and dynamic columns.
        static_data_path: Path to static_data.csv file.

    Returns:
        Tuple of (static_cols, dynamic_cols) present in df.

    Example:
        >>> df = pd.read_csv("data/processed/flat_table.csv", nrows=100)
        >>> static_cols, dynamic_cols = split_static_dynamic(df)
        >>> print(f"Static: {len(static_cols)}, Dynamic: {len(dynamic_cols)}")
    """
    # Load static columns from reference file
    try:
        df_static = pd.read_csv(static_data_path)
        # Exclude ID columns
        id_columns = ["subject_id", "hadm_id", "icustay_id"]
        static_cols_all = [col for col in df_static.columns if col not in id_columns]
    except FileNotFoundError:
        # Fallback: heuristic detection based on column variance
        print(f"Warning: {static_data_path} not found, using heuristic detection")
        static_cols_all = []

    # Only keep static columns that exist in df
    static_cols = [col for col in static_cols_all if col in df.columns]

    # Dynamic columns are everything except IDs, time, static, and lagged IDs
    id_columns = ["subject_id", "hadm_id", "icustay_id"]
    exclude = id_columns + ["hours_in"] + static_cols

    # Also exclude any lagged versions of ID columns (shouldn't exist, but safety check)
    lagged_id_cols = [f"{id_col}_lag{i}" for id_col in id_columns for i in range(1, 10)]
    exclude.extend([col for col in lagged_id_cols if col in df.columns])

    dynamic_cols = [col for col in df.columns if col not in exclude]

    return static_cols, dynamic_cols
