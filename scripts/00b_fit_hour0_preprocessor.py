#!/usr/bin/env python3
"""00b: Fit Hour-0 Preprocessor - Fit TabularPreprocessor for hour-0 features.

This script fits a preprocessor specifically for hour-0 data (demographics + initial vitals).
This is separate from the autoregressive preprocessor because the feature set is different
(no lagged features, just static + current timestep).

Note: The hour0_data.csv contains ID columns (subject_id, hadm_id, icustay_id, hours_in)
for bookkeeping and debugging, but these are filtered out during training as they
should not be used as model features. This is consistent with the autoregressive
pipeline where IDs are kept in the CSV but excluded via split_static_dynamic().

Usage:
    uv run python scripts/00b_fit_hour0_preprocessor.py

Input:
    data/processed/hour0_data.csv

Output:
    models/hour0_preprocessor.pkl
"""

import sys
from pathlib import Path
import pickle

import pandas as pd

from packages.data import TabularPreprocessor, identify_feature_types


def main():
    print("=" * 70)
    print("00b: FIT HOUR-0 PREPROCESSOR")
    print("=" * 70)

    # Configuration
    input_path = "data/processed/hour0_data.csv"
    preprocessor_output = "artifacts/hour0_preprocessor.pkl"

    # Check if input exists
    if not Path(input_path).exists():
        print(f"\nError: {input_path} not found!")
        print("Run 00_prepare_hour0_data.py first to prepare hour-0 data.")
        sys.exit(1)

    # Load hour-0 data
    print(f"\nLoading hour-0 data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Filter out ID columns (kept in CSV for bookkeeping, not used as features)
    id_columns = ["subject_id", "hadm_id", "icustay_id", "hours_in"]
    all_cols = [col for col in df.columns if col not in id_columns]

    print(f"\nFeatures to preprocess: {len(all_cols)}")
    print(
        f"  (Filtered out {len([c for c in id_columns if c in df.columns])} ID columns)"
    )

    # Use centralized feature type identification
    print("\nClassifying column types using centralized detection...")
    feature_types = identify_feature_types(df, columns=all_cols)

    numeric_cols = feature_types["numeric"]
    binary_cols = feature_types["binary"]
    categorical_cols = feature_types["categorical"]
    int_cols = feature_types["int"]

    print("\nColumn type distribution:")
    print(f"  Numeric (continuous): {len(numeric_cols)}")
    print(f"  Binary (0/1 flags): {len(binary_cols)}")
    print(f"  Categorical (multi-class): {len(categorical_cols)}")
    print(f"  Integer (subset of numeric): {len(int_cols)}")

    # Show sample columns
    if numeric_cols:
        print(f"\n  Sample numeric: {numeric_cols[:5]}")
    if binary_cols:
        print(f"  Sample binary: {binary_cols[:5]}")
    if categorical_cols:
        print(f"  Sample categorical: {categorical_cols[:5]}")

    # Initialize preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = TabularPreprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        int_cols=int_cols,
        binary_cols=binary_cols,
    )

    # Fit preprocessor on all hour-0 data
    print(f"\nFitting preprocessor on hour-0 data ({len(df):,} patients)...")
    print("  This learns correct min/max ranges and categorical encodings")

    preprocessor.fit(df[all_cols])

    print("\n  ✓ Preprocessor fitted!")

    # Display fitted statistics
    print("\nFitted preprocessor statistics:")
    print(f"  Input features: {len(all_cols)}")
    print(f"  Output features: {preprocessor.n_features}")
    print(f"  Samples seen: {preprocessor._n_samples_seen:,}")

    # Show sample value ranges (for verification)
    print("\n  Sample numeric feature ranges from data:")
    for col in numeric_cols[:5]:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"    {col}: [{min_val:.2f}, {max_val:.2f}]")

    if categorical_cols and preprocessor.category_mappings:
        total_categories = sum(
            len(mapping) for mapping in preprocessor.category_mappings.values()
        )
        print(
            f"\n  Categorical encodings: {total_categories} unique values → {len(categorical_cols)} integer columns"
        )

    # Test transform/inverse transform
    print("\nTesting transform/inverse transform...")
    X_transformed = preprocessor.transform(df[all_cols].iloc[:5])
    X_inverse = preprocessor.inverse_transform(X_transformed)

    print(f"  Original shape: {df[all_cols].iloc[:5].shape}")
    print(f"  Transformed shape: {X_transformed.shape}")
    print(f"  Inverse shape: {X_inverse.shape}")

    # Check reconstruction quality
    original_sample = df[all_cols].iloc[:5]
    max_diff = 0
    for col in numeric_cols[:5]:
        if col in original_sample.columns and col in X_inverse.columns:
            diff = abs(original_sample[col] - X_inverse[col]).max()
            max_diff = max(max_diff, diff)

    print(f"  Max reconstruction error (numeric): {max_diff:.6f}")
    if max_diff < 0.01:
        print("  ✓ Reconstruction looks good!")
    else:
        print("  ⚠️  Warning: High reconstruction error")

    # Prepare metadata to save
    preprocessor_data = {
        "preprocessor": preprocessor,
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

    print("\n✅ Hour-0 preprocessor fitting complete!")
    print(f"   Input features: {len(all_cols)}")
    print(f"     - Numeric (continuous): {len(numeric_cols)}")
    print(f"     - Binary (discrete 0/1): {len(binary_cols)}")
    print(f"     - Categorical (strings): {len(categorical_cols)}")
    print(f"   Output features: {preprocessor.n_features} (after encoding)")
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Run 00c_train_hour0_model.py to train Hour-0 Forest-Flow")
    print("=" * 70)


if __name__ == "__main__":
    main()
