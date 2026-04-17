"""Data loading utilities."""

from __future__ import annotations

import pandas as pd
from pathlib import Path


def load_csv(
    path: str | Path, nrows: int | None = None, low_memory: bool = False, **kwargs
) -> pd.DataFrame:
    """Load CSV file with sensible defaults.

    Args:
        path: Path to CSV file.
        nrows: Optional number of rows to read.
        low_memory: Use low memory mode for large files.
        **kwargs: Additional arguments passed to pd.read_csv.

    Returns:
        DataFrame with loaded data.
    """
    return pd.read_csv(path, nrows=nrows, low_memory=low_memory, **kwargs)


def load_parquet(
    path: str | Path, columns: list[str] | None = None, **kwargs
) -> pd.DataFrame:
    """Load Parquet file with sensible defaults.

    Args:
        path: Path to Parquet file.
        columns: Optional list of columns to load.
        **kwargs: Additional arguments passed to pd.read_parquet.

    Returns:
        DataFrame with loaded data.
    """
    return pd.read_parquet(path, columns=columns, **kwargs)


def load_mimic_flat_table(
    path: str = "data/processed/flat_table.csv",
    nrows: int | None = None,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load MIMIC-III flat table with common defaults.

    Args:
        path: Path to flat_table.csv.
        nrows: Optional number of rows to read.
        exclude_cols: Optional list of columns to exclude.

    Returns:
        DataFrame with MIMIC-III time series data.
    """
    df = load_csv(path, nrows=nrows, low_memory=False)

    if exclude_cols:
        df = df.drop(
            columns=[c for c in exclude_cols if c in df.columns], errors="ignore"
        )

    return df
