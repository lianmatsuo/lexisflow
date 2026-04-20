#!/usr/bin/env python3
"""Fit the TabularPreprocessor on the full autoregressive dataset.

This script fits the preprocessor on the entire autoregressive dataset to ensure
it learns the correct min/max ranges and sees all categorical values. This is
critical for proper scaling and encoding, especially for rare categories.

The preprocessor is fit once on all data, saved, and reused during training.
This allows training on subsamples while maintaining correct data ranges.

Usage:
    uv run python scripts/fit_autoregressive_preprocessor.py

Input:
    data/processed/autoregressive_data.csv

Output:
    models/preprocessor_full.pkl
"""

from pathlib import Path
import pickle

import pandas as pd
from tqdm import tqdm

from synth_gen.data import (
    TabularPreprocessor,
    split_static_dynamic,
    is_lagged,
    identify_feature_types,
)


def main():
    print("=" * 70)
    print("01b: FIT PREPROCESSOR ON FULL DATASET")
    print("=" * 70)

    # Configuration
    input_path = "data/processed/autoregressive_data.csv"

    # Save to models directory
    preprocessor_output = "artifacts/preprocessor_full.pkl"
    column_split_output = "artifacts/column_split.txt"

    chunk_size = 100000  # Process in chunks for memory efficiency

    # Get total number of rows for progress tracking
    print("\nCounting total rows...")
    print("  (This may take a moment for large CSV files)")
    with open(input_path, "r") as f:
        total_rows = sum(1 for _ in f) - 1  # -1 for header

    print(f"  Total rows: {total_rows:,}")

    # Load first chunk to identify columns
    print("\nLoading first chunk to identify columns...")
    df_first = pd.read_csv(input_path, nrows=chunk_size, low_memory=False)

    print(f"  Columns: {df_first.shape[1]}")

    # Identify static and dynamic columns
    print("\nIdentifying feature columns...")
    static_cols, dynamic_cols = split_static_dynamic(df_first)

    # Separate current timestep (target) from lagged (condition)
    # Target: columns without _lag1 suffix
    # Condition: static + columns with _lag1 suffix
    target_cols = [col for col in dynamic_cols if not is_lagged(col)]
    lagged_cols = [col for col in dynamic_cols if is_lagged(col)]
    condition_cols = static_cols + lagged_cols

    # All unique columns (no duplicates)
    all_cols = target_cols + condition_cols

    print(f"  Target features (current timestep): {len(target_cols)}")
    print(f"  Condition features (history + static): {len(condition_cols)}")
    print(f"    - Static: {len(static_cols)}")
    print(f"    - Lagged: {len(lagged_cols)}")
    print(f"  Total unique features: {len(all_cols)}")

    # Use centralized feature type identification
    print("\nIdentifying feature types using centralized detection...")
    feature_types = identify_feature_types(df_first, columns=all_cols)

    numeric_cols = feature_types["numeric"]
    binary_cols = feature_types["binary"]
    categorical_cols = feature_types["categorical"]
    int_cols = feature_types["int"]

    print("\nColumn types:")
    print(f"  Numeric (continuous): {len(numeric_cols)}")
    print(f"  Binary (0/1 flags): {len(binary_cols)}")
    print(f"  Categorical (multi-class): {len(categorical_cols)}")
    print(f"  Integer (subset of numeric): {len(int_cols)}")

    # Initialize preprocessor
    print("\nInitializing preprocessor...")
    print("  Configuration:")
    print(f"    - Numeric (continuous): {len(numeric_cols)} features")
    print(f"    - Binary (discrete 0/1): {len(binary_cols)} features")
    print(f"    - Categorical (strings): {len(categorical_cols)} features")

    preprocessor = TabularPreprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        int_cols=int_cols,
        binary_cols=binary_cols,
    )

    # Fit preprocessor on full data using streaming
    print(f"\nFitting preprocessor on FULL dataset ({total_rows:,} rows)...")
    print(f"  Processing in chunks of {chunk_size:,} rows")
    print("  This ensures correct min/max ranges and captures all categories")

    # CSV: Stream in chunks
    n_chunks = (total_rows + chunk_size - 1) // chunk_size

    # Stream through all data - partial_fit handles both categorical and numeric
    print("\n  Streaming through data...")
    print("  - Collecting categorical values for label encoding")
    print("  - Fitting scaler on numeric/binary features")
    chunk_iter = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False)
    for chunk in tqdm(
        chunk_iter, total=n_chunks, desc="  Fitting preprocessor", unit="chunk"
    ):
        preprocessor.partial_fit(chunk[all_cols])

    # Finalize to build label encoding mappings
    preprocessor.finalize_fit()

    # Report categorical statistics
    if categorical_cols:
        print("\n  Categorical encoding summary:")
        total_categories = 0
        for col in categorical_cols:
            n_cats = len(preprocessor.category_mappings.get(col, {}))
            total_categories += n_cats
            print(f"    {col}: {n_cats} unique values")
        print(f"  Total unique categories: {total_categories}")
        print(f"  Encoded as {len(categorical_cols)} integer columns (label encoding)")
        print(f"    vs. {total_categories} dummy columns with one-hot encoding")

    print("\n  ✓ Preprocessor fitted on full dataset!")

    # Build explicit transformed-space indices for target/condition splits.
    transformed_cols = (
        list(preprocessor.numeric_cols)
        + list(getattr(preprocessor, "binary_cols", []))
        + list(preprocessor.categorical_cols)
    )
    transformed_index = {col: idx for idx, col in enumerate(transformed_cols)}
    target_indices = [transformed_index[col] for col in target_cols]
    condition_indices = [transformed_index[col] for col in condition_cols]
    if target_indices != list(range(len(target_indices))):
        print(
            "  Note: target features are non-contiguous in transformed space; "
            "saving explicit index mapping."
        )

    # Display fitted ranges for key features (sanity check)
    print("\nFitted preprocessor statistics:")
    print(f"  Total input features: {len(all_cols)}")
    print(f"  Total output features: {preprocessor.n_features}")
    print(f"  Samples processed: {preprocessor._n_samples_seen:,}")

    # Show sample value ranges (for verification)
    print("\n  Sample numeric feature ranges (from data):")
    sample_df = pd.read_csv(input_path, nrows=1000)
    for col in numeric_cols[:5]:
        if col in sample_df.columns:
            min_val = sample_df[col].min()
            max_val = sample_df[col].max()
            print(f"    {col}: [{min_val:.2f}, {max_val:.2f}]")

    # Prepare metadata to save
    preprocessor_data = {
        "preprocessor": preprocessor,
        "target_cols": target_cols,
        "condition_cols": condition_cols,
        "n_target": len(target_cols),
        "target_indices": target_indices,
        "condition_indices": condition_indices,
        "all_cols": all_cols,
        "numeric_cols": numeric_cols,
        "binary_cols": binary_cols,
        "categorical_cols": categorical_cols,
        "int_cols": int_cols,
    }

    # Save to models/ directory
    print(f"\nSaving fitted preprocessor to {preprocessor_output}...")
    Path(preprocessor_output).parent.mkdir(parents=True, exist_ok=True)

    with open(preprocessor_output, "wb") as f:
        pickle.dump(preprocessor_data, f)

    print(f"  ✓ Saved to {preprocessor_output}")

    # Save column split info
    print(f"\nSaving column split info to {column_split_output}...")
    with open(column_split_output, "w") as f:
        f.write(f"n_target={len(target_cols)}\n")
        f.write(f"n_condition={len(condition_cols)}\n")

    print(f"  ✓ Saved to {column_split_output}")

    print("\n✅ Preprocessor fitting complete!")
    print(f"   Fitted on: {total_rows:,} rows (FULL dataset)")
    print(f"   Input features: {len(all_cols)}")
    print(f"     - Numeric (continuous): {len(numeric_cols)}")
    print(f"     - Binary (discrete 0/1): {len(binary_cols)}")
    print(f"     - Categorical (strings): {len(categorical_cols)}")
    print(f"   Output features: {preprocessor.n_features} (after encoding)")
    print("\n   ✅ CRITICAL BENEFITS:")
    print("   • Preprocessor has CORRECT min/max ranges from all data")
    print("   • All rare categorical values are encoded")
    print("   • Binary features auto-round to {0, 1} on inverse transform")
    print("   • Training on subsamples maintains proper scaling!")
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Run scripts/train_autoregressive.py to train Forest-Flow")
    print("     (it will automatically use this pre-fitted preprocessor)")
    print("  2. Run run_sweep.py for hyperparameter search")
    print("     (it will use the same preprocessor)")
    print("=" * 70)


if __name__ == "__main__":
    main()
