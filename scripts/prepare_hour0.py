#!/usr/bin/env python3
"""Prepare hour-0 training rows from the flat MIMIC table for IID generation.

This script prepares training data for the Hour-0 model, which generates
demographics + initial vital signs. These outputs are used as inputs for the
autoregressive Forest-Flow model.

Steps:
1. Flatten column names and drop cached 100%-missing columns
2. Drop datetime columns (shared list with autoregressive prep)
3. Apply shared feature pruning policy from synth_gen.data
4. Filter to hours_in == 0 (or first available hour per patient)
5. Select static + hour-0 vital columns
6. Drop ID columns from modeling features (subject_id, hadm_id, icustay_id kept in CSV for training filters)
7. Handle missing values (median for vitals, mode for demographics)
8. Bucket rare diagnosis_at_admission classes into OTHER (configurable)
9. Save as hour0_data.csv

Usage:
    uv run python scripts/prepare_hour0.py

Input:
    data/processed/flat_table.csv

Output:
    data/processed/hour0_data.csv
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

from synth_gen.data import (
    columns_to_drop_default_feature_pruning,
    DEFAULT_DATETIME_COLUMNS,
    flatten_column_names,
    normalize_column_name_list,
    split_static_dynamic,
)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare hour-0 data for IID generation"
    )
    parser.add_argument(
        "--diagnosis-min-count",
        type=int,
        default=50,
        help=(
            "Minimum frequency to keep a diagnosis_at_admission category. "
            "Categories below this count are mapped to OTHER (default: 50)."
        ),
    )
    parser.add_argument(
        "--diagnosis-other-label",
        type=str,
        default="OTHER",
        help="Label used for rare diagnosis_at_admission categories (default: OTHER).",
    )
    parser.add_argument(
        "--disable-diagnosis-bucketing",
        action="store_true",
        help="Disable rare-category bucketing for diagnosis_at_admission.",
    )
    parser.add_argument(
        "--keep-biophysical-aggregates",
        action="store_true",
        help=(
            "Keep pruned biophysical aggregates and sparse features "
            "(default: apply shared pruning policy used by autoregressive prep)."
        ),
    )
    args = parser.parse_args()

    print("=" * 70)
    print("00: PREPARE HOUR-0 DATA")
    print("=" * 70)

    # Configuration
    input_path = "data/processed/flat_table.csv"
    output_path = "data/processed/hour0_data.csv"
    missing_cols_cache = "data/processed/completely_missing_columns.txt"

    # Check if input exists
    if not Path(input_path).exists():
        print(f"\nError: {input_path} not found!")
        print("Please ensure flat_table.csv exists in data/processed/")
        sys.exit(1)

    # Load flat table
    print(f"\nLoading flat table from {input_path}...")
    print("  (This may take a moment for large files)")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Flatten any tuple column names to strings
    print("\n  Flattening column names (tuple → string)...")
    df = flatten_column_names(df)
    print("  ✓ Column names standardized")

    # Align with prepare_autoregressive.py: drop cached 100%-missing columns.
    if Path(missing_cols_cache).exists():
        print(f"\n  Loading cached 100%-missing columns from {missing_cols_cache}...")
        with open(missing_cols_cache, "r") as f:
            cached_raw = [line.strip() for line in f if line.strip()]
        cached_missing_cols = normalize_column_name_list(cached_raw)
        cols_to_drop_missing = [col for col in cached_missing_cols if col in df.columns]
        if cols_to_drop_missing:
            print(
                f"  Dropping {len(cols_to_drop_missing)} cached fully-missing columns..."
            )
            df = df.drop(columns=cols_to_drop_missing)
            print(f"  ✓ Shape after drop: {df.shape[0]:,} rows × {df.shape[1]} columns")
        else:
            print("  ✓ Cached missing columns not present after flattening; skipping.")
    else:
        print(
            f"\n  Note: {missing_cols_cache} not found; skipping explicit 100%-missing drop."
        )

    # Align with prepare_autoregressive.py: drop shared datetime columns upfront.
    datetime_cols_present = [
        col for col in DEFAULT_DATETIME_COLUMNS if col in df.columns
    ]
    if datetime_cols_present:
        print(f"\n  Dropping {len(datetime_cols_present)} datetime columns...")
        df = df.drop(columns=datetime_cols_present)
        print(f"  ✓ Shape after drop: {df.shape[0]:,} rows × {df.shape[1]} columns")

    if not args.keep_biophysical_aggregates:
        drop_bio = columns_to_drop_default_feature_pruning(df.columns)
        if drop_bio:
            print(
                f"\n  Dropping {len(drop_bio)} columns from shared pruning policy "
                "(biophysical *_std/*_count + sparse removals; use "
                "--keep-biophysical-aggregates to retain them)..."
            )
            df = df.drop(columns=drop_bio)
            print(f"  ✓ Shape after drop: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Check for hours_in column
    if "hours_in" not in df.columns:
        print("\nError: 'hours_in' column not found!")
        print(f"Available columns: {list(df.columns)[:10]}...")
        sys.exit(1)

    # Filter to hour 0 (or first hour for each patient)
    print("\nFiltering to hour-0 data...")
    print(f"  Hours_in range: {df['hours_in'].min()} to {df['hours_in'].max()}")

    # Try hour 0 first
    df_hour0 = df[df["hours_in"] == 0].copy()

    if len(df_hour0) == 0:
        print("  ⚠️  No hour 0 data found, using first hour for each patient...")
        # Get first hour for each patient
        df_hour0 = (
            df.sort_values("hours_in").groupby("subject_id").first().reset_index()
        )
    else:
        print(f"  ✓ Found {len(df_hour0):,} hour-0 records")

    # Identify static and dynamic columns
    print("\nIdentifying feature columns...")
    static_cols, dynamic_cols = split_static_dynamic(df_hour0)

    print(f"  Static features: {len(static_cols)}")
    print(f"  Dynamic features: {len(dynamic_cols)}")

    # For hour-0 model: we want static + current hour vitals (no lag)
    # Keep ID columns in CSV for bookkeeping/debugging (will be filtered during training)
    # Exclude timestamp columns (not useful for modeling)
    exclude_columns = list(DEFAULT_DATETIME_COLUMNS)

    # Select features to keep (including IDs for consistency with autoregressive data)
    id_columns = ["subject_id", "hadm_id", "icustay_id", "hours_in"]
    hour0_features = static_cols + dynamic_cols
    hour0_features = [
        col
        for col in hour0_features
        if col in df_hour0.columns and col not in exclude_columns
    ]

    # Add ID columns (if present)
    for id_col in id_columns:
        if id_col in df_hour0.columns and id_col not in hour0_features:
            hour0_features.append(id_col)

    print("\nSelecting hour-0 features:")
    print(f"  Total columns (including IDs): {len(hour0_features)}")
    print(
        f"    - ID columns: {len([c for c in id_columns if c in hour0_features])} (kept for bookkeeping)"
    )
    print(
        f"    - Static (demographics): {len([c for c in static_cols if c in hour0_features])}"
    )
    print(
        f"    - Dynamic (vitals): {len([c for c in dynamic_cols if c in hour0_features])}"
    )
    print(
        "  Note: IDs kept in CSV but filtered during training (consistent with autoregressive)"
    )

    # Create hour-0 dataset
    df_hour0_clean = df_hour0[hour0_features].copy()

    # Handle missing values
    print("\nHandling missing values...")

    # Count missing before
    total_values = df_hour0_clean.shape[0] * df_hour0_clean.shape[1]
    missing_before = df_hour0_clean.isna().sum().sum()
    missing_pct = (missing_before / total_values) * 100

    print(
        f"  Missing values before: {missing_before:,} / {total_values:,} ({missing_pct:.2f}%)"
    )

    # Strategy:
    # - Numeric features: median imputation
    # - Categorical features: mode imputation or 'Unknown'

    for col in tqdm(hour0_features, desc="  Imputing columns", unit="col"):
        if col not in df_hour0_clean.columns:
            continue

        missing_count = df_hour0_clean[col].isna().sum()
        if missing_count == 0:
            continue

        dtype = df_hour0_clean[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            # Median imputation for numeric
            median_val = df_hour0_clean[col].median()
            df_hour0_clean[col] = df_hour0_clean[col].fillna(median_val)
        else:
            # Mode imputation for categorical
            if df_hour0_clean[col].mode().empty:
                # If no mode (all missing), use 'Unknown'
                df_hour0_clean[col] = df_hour0_clean[col].fillna("Unknown")
            else:
                mode_val = df_hour0_clean[col].mode()[0]
                df_hour0_clean[col] = df_hour0_clean[col].fillna(mode_val)

    # Count missing after
    missing_after = df_hour0_clean.isna().sum().sum()
    print(f"  Missing values after: {missing_after:,} / {total_values:,}")

    if missing_after > 0:
        print(f"  ⚠️  Warning: {missing_after} missing values remain")
        # Show which columns still have missing
        missing_cols = df_hour0_clean.columns[df_hour0_clean.isna().any()].tolist()
        print(f"     Columns with missing: {missing_cols[:10]}...")
    else:
        print("  ✓ All missing values handled!")

    # Bucket rare diagnosis classes to reduce extreme cardinality in hour-0 HS3F.
    # This significantly reduces training time for the categorical classifier head.
    diagnosis_col_name = "diagnosis_at_admission"
    if not args.disable_diagnosis_bucketing:
        if diagnosis_col_name in df_hour0_clean.columns:
            print("\nBucketing rare diagnosis_at_admission categories...")
            diagnosis_series = (
                df_hour0_clean[diagnosis_col_name]
                .astype(str)
                .str.strip()
                .replace("", "Unknown")
            )
            before_nunique = diagnosis_series.nunique(dropna=False)
            value_counts = diagnosis_series.value_counts(dropna=False)
            rare_labels = value_counts[value_counts < args.diagnosis_min_count].index
            rare_mask = diagnosis_series.isin(rare_labels)
            rare_rows = int(rare_mask.sum())

            if len(rare_labels) > 0:
                df_hour0_clean[diagnosis_col_name] = diagnosis_series.where(
                    ~rare_mask, args.diagnosis_other_label
                )
                after_nunique = df_hour0_clean[diagnosis_col_name].nunique(dropna=False)
                print(
                    "  ✓ diagnosis_at_admission unique values: "
                    f"{before_nunique:,} -> {after_nunique:,}"
                )
                print(
                    "  ✓ Bucketed rare labels: "
                    f"{len(rare_labels):,} labels across {rare_rows:,} rows "
                    f"(min_count={args.diagnosis_min_count}, other={args.diagnosis_other_label})"
                )
            else:
                print(
                    "  ✓ No diagnosis categories below min_count="
                    f"{args.diagnosis_min_count}; no bucketing applied"
                )
        else:
            print(
                "\nSkipping diagnosis bucketing: "
                "'diagnosis_at_admission' column not present"
            )
    else:
        print("\nDiagnosis bucketing disabled by flag.")

    # Display statistics
    print("\nHour-0 dataset statistics:")
    print(f"  Patients: {len(df_hour0_clean):,}")
    print(f"  Features: {len(hour0_features)}")
    print(f"  Total data points: {total_values:,}")

    # Show sample of first few columns
    print("\nSample data (first 5 rows, first 5 columns):")
    print(df_hour0_clean.iloc[:5, :5].to_string())

    # Save hour-0 data
    print(f"\nSaving hour-0 data to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_hour0_clean.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {len(df_hour0_clean):,} rows × {len(hour0_features)} columns")

    print("\n✅ Hour-0 data preparation complete!")
    print(f"   Output: {output_path}")
    print(f"   Patients: {len(df_hour0_clean):,}")
    print(f"   Features: {len(hour0_features)}")
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Run scripts/fit_hour0_preprocessor.py to fit preprocessor")
    print("  2. Run scripts/train_hour0.py to train Hour-0 model")
    print("=" * 70)


if __name__ == "__main__":
    main()
