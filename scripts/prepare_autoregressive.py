#!/usr/bin/env python3
"""Prepare autoregressive training data from the flat MIMIC table.

This script transforms the flat MIMIC-III table into autoregressive format
suitable for conditional flow matching models.

Steps:
1. Identify and drop 100% missing columns (using DuckDB to scan entire file)
2. Drop datetime columns (timestamps that shouldn't be features)
3. Apply shared feature-pruning policy (biophysical *_std/*_count + sparse removals)
4. Create autoregressive features with lag=1

Usage:
    uv run python scripts/prepare_autoregressive.py

Output:
    data/processed/autoregressive_data.csv
"""

import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd
import duckdb
from tqdm import tqdm

from synth_gen.data import (
    columns_to_drop_default_feature_pruning,
    DEFAULT_DATETIME_COLUMNS,
    prepare_autoregressive_data,
    split_static_dynamic,
    flatten_column_names,
    normalize_column_name_list,
)


def identify_missing_columns_duckdb(csv_path: str) -> list[str]:
    """Use DuckDB to identify columns that are 100% missing across entire dataset.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of column names that are completely missing
    """
    print("  Using DuckDB to scan entire file for 100% missing columns...")
    print("  (This may take several minutes for large files)")

    con = duckdb.connect(database=":memory:")

    # Get column names
    print("  Loading column list...")
    result = con.execute(f"""
        SELECT * FROM read_csv_auto('{csv_path}', header=true, sample_size=100)
        LIMIT 0
    """).df()
    all_columns = result.columns.tolist()

    # Get total row count
    print("  Counting total rows...")
    total_rows = con.execute(f"""
        SELECT COUNT(*) as count
        FROM read_csv_auto('{csv_path}', header=true, sample_size=-1)
    """).fetchone()[0]

    print(f"  Total rows: {total_rows:,}, Total columns: {len(all_columns)}")
    print(
        "\n  Checking each column for missing values (this will take several minutes)..."
    )

    # Check each column with progress bar
    completely_missing = []

    pbar = tqdm(
        all_columns,
        desc="  Scanning columns",
        unit="col",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for col in pbar:
        # Escape column name properly for DuckDB
        escaped_col = f'"{col}"'

        try:
            query = f"""
                SELECT COUNT(*) as non_null_count
                FROM read_csv_auto('{csv_path}', header=true, sample_size=-1)
                WHERE {escaped_col} IS NOT NULL
                  AND CAST({escaped_col} AS VARCHAR) != ''
            """
            non_null_count = con.execute(query).fetchone()[0]

            if non_null_count == 0:
                completely_missing.append(col)
                pbar.set_postfix({"found_missing": len(completely_missing)})

        except Exception:
            # If error checking, assume not missing
            continue

    con.close()
    print(f"\n  ✓ Found {len(completely_missing)} columns with 100% missing values")

    return completely_missing


def main():
    print("=" * 70)
    print("01: DATA PREPROCESSING")
    print("=" * 70)

    # Configuration
    input_path = "data/processed/flat_table.csv"
    output_path = "data/processed/autoregressive_data.csv"
    missing_cols_cache = "data/processed/completely_missing_columns.txt"
    max_rows = None  # Set to integer for quick testing, None for all data

    # Check input exists
    if not Path(input_path).exists():
        print(f"\nError: {input_path} not found!")
        print("Please ensure MIMIC-III data is properly preprocessed.")
        sys.exit(1)

    # Step 1: Identify 100% missing columns using DuckDB
    print("\nSTEP 1: Identifying 100% missing columns...")
    if Path(missing_cols_cache).exists():
        print(f"  Loading cached results from {missing_cols_cache}")
        with open(missing_cols_cache, "r") as f:
            cached_raw = [line.strip() for line in f if line.strip()]
        completely_missing_cols = normalize_column_name_list(cached_raw)
        print(
            f"  Found {len(completely_missing_cols)} completely missing columns (cached)"
        )
    else:
        completely_missing_cols = normalize_column_name_list(
            identify_missing_columns_duckdb(input_path)
        )
        # Save cache
        with open(missing_cols_cache, "w") as f:
            for col in completely_missing_cols:
                f.write(f"{col}\n")
        print(f"  Saved cache to {missing_cols_cache}")

    # Step 2: Define datetime columns to drop
    print("\nSTEP 2: Identifying datetime columns to drop...")
    datetime_columns = list(DEFAULT_DATETIME_COLUMNS)
    print(f"  Datetime columns: {datetime_columns}")

    # Step 3: Load data and drop columns
    print(f"\nSTEP 3: Loading data from {input_path}...")
    print("  (This may take a few minutes for large files)")
    if max_rows:
        print(f"  Testing mode: loading only {max_rows:,} rows")
        df = pd.read_csv(input_path, nrows=max_rows, low_memory=False)
    else:
        # Show file size
        file_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.1f} MB")
        df = pd.read_csv(input_path, low_memory=False)

    print(f"  ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Flatten any tuple column names to strings
    print("\n  Flattening column names (tuple → string)...")
    df = flatten_column_names(df)
    print("  ✓ Column names standardized")

    # Drop 100% missing columns
    cols_to_drop_missing = [col for col in completely_missing_cols if col in df.columns]
    if cols_to_drop_missing:
        print(
            f"\n  Dropping {len(cols_to_drop_missing)} columns with 100% missing values..."
        )
        df = df.drop(columns=cols_to_drop_missing)
        print(
            f"  Shape after dropping missing: {df.shape[0]:,} rows × {df.shape[1]} columns"
        )

    # Drop datetime columns
    cols_to_drop_datetime = [col for col in datetime_columns if col in df.columns]
    if cols_to_drop_datetime:
        print(f"\n  Dropping {len(cols_to_drop_datetime)} datetime columns...")
        print(f"  Columns: {cols_to_drop_datetime}")
        df = df.drop(columns=cols_to_drop_datetime)
        print(
            f"  Shape after dropping datetime: {df.shape[0]:,} rows × {df.shape[1]} columns"
        )

    # Step 4: Detect static and dynamic columns
    print("\nSTEP 4: Detecting static and dynamic features...")
    static_cols, dynamic_cols = split_static_dynamic(df)
    print(f"  Static features: {len(static_cols)}")
    print(f"  Dynamic features: {len(dynamic_cols)}")

    # Step 5: Prepare autoregressive data
    print("\nSTEP 5: Creating autoregressive features (lag=1)...")
    print(
        f"  Processing {df.shape[0]:,} rows across {df['subject_id'].nunique()} patients..."
    )
    prune_cols = columns_to_drop_default_feature_pruning(df.columns)
    print(f"  Shared feature-pruning policy will drop {len(prune_cols)} input columns")

    start_time = time.time()

    df_ar, target_cols, condition_cols = prepare_autoregressive_data(
        df,
        id_col="subject_id",
        time_col="hours_in",
        static_cols=static_cols,
        lag=1,
        fill_strategy="special",
        fill_value=-1.0,
    )

    elapsed = time.time() - start_time
    print(f"  ✓ Completed in {elapsed:.1f} seconds")
    print(f"  Target features: {len(target_cols)}")
    print(f"  Condition features: {len(condition_cols)}")
    print(f"  Output shape: {df_ar.shape[0]:,} rows × {df_ar.shape[1]} columns")

    # Step 6: Patient-level train / test / holdout split then save
    # 80% train, 10% quality+TSTR test, 10% privacy holdout — patient-disjoint.
    # Training CSV only contains train-split patients so quality/privacy metrics
    # computed against the test/holdout CSVs cannot leak training rows.
    print("\nSTEP 6: Patient-level train/test/holdout split (80/10/10)...")
    subjects = df_ar["subject_id"].unique()
    rng_split = np.random.default_rng(0)
    rng_split.shuffle(subjects)
    n_subjects = len(subjects)
    n_test = max(1, int(0.10 * n_subjects))
    n_holdout = max(1, int(0.10 * n_subjects))
    test_subjects = set(subjects[:n_test].tolist())
    holdout_subjects = set(subjects[n_test : n_test + n_holdout].tolist())
    train_subjects = set(subjects[n_test + n_holdout :].tolist())

    df_train = df_ar[df_ar["subject_id"].isin(train_subjects)]
    df_test = df_ar[df_ar["subject_id"].isin(test_subjects)]
    df_holdout = df_ar[df_ar["subject_id"].isin(holdout_subjects)]

    print(
        f"  Patients: {len(train_subjects):,} train / {len(test_subjects):,} test / {len(holdout_subjects):,} holdout"
    )
    print(
        f"  Rows:     {len(df_train):,} train / {len(df_test):,} test / {len(df_holdout):,} holdout"
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    real_test_path = "data/processed/real_test.csv"
    real_holdout_path = "data/processed/real_holdout.csv"

    start_time = time.time()
    df_train.to_csv(output_path, index=False)
    df_test.to_csv(real_test_path, index=False)
    df_holdout.to_csv(real_holdout_path, index=False)
    elapsed = time.time() - start_time

    print(f"  ✓ Saved in {elapsed:.1f} seconds")
    print(f"    {output_path} ({Path(output_path).stat().st_size / 1024**2:.1f} MB)")
    print(
        f"    {real_test_path} ({Path(real_test_path).stat().st_size / 1024**2:.1f} MB)"
    )
    print(
        f"    {real_holdout_path} ({Path(real_holdout_path).stat().st_size / 1024**2:.1f} MB)"
    )

    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(
        f"Train:   {output_path}  ({len(df_train):,} rows, {len(train_subjects):,} patients)"
    )
    print(
        f"Test:    {real_test_path}  ({len(df_test):,} rows, {len(test_subjects):,} patients)"
    )
    print(
        f"Holdout: {real_holdout_path}  ({len(df_holdout):,} rows, {len(holdout_subjects):,} patients)"
    )
    print("\nColumns dropped:")
    print(f"  - 100% missing: {len(cols_to_drop_missing)}")
    print(f"  - Datetime: {len(cols_to_drop_datetime)}")
    print(
        f"  - Total dropped: {len(cols_to_drop_missing) + len(cols_to_drop_datetime)}"
    )
    print("\n" + "=" * 70)
    print(
        "Next step: Run scripts/fit_autoregressive_preprocessor.py to fit preprocessor on full data"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
