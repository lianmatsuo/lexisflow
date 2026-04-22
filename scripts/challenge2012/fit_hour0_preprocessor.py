#!/usr/bin/env python3
"""Fit the TabularPreprocessor for Challenge 2012 hour-0 features.

Output:
    artifacts/challenge2012/hour0_preprocessor.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

from lexisflow.data import TabularPreprocessor, identify_feature_types


# Only identifier columns are excluded. Labels (hospital_expire_flag, los_icu)
# are retained as features so they survive the inverse transform in downstream
# synthetic DataFrames (required by MortalityTask / LOSTask).
EXCLUDE_COLS = {"subject_id", "hours_in"}
EXCLUDE_NOISY_COLS = {
    "Temp",
    "PaO2",
    "PaCO2",
    "ALP",
    "SaO2",
    "Glucose",
    "Platelets",
    "Height",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="data/challenge2012/processed/hour0_data.csv",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/challenge2012/hour0_preprocessor.pkl",
    )
    args = parser.parse_args()

    if not Path(args.input_path).exists():
        print(
            f"ERROR: {args.input_path} not found. Run prepare_hour0.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 70)
    print("CHALLENGE 2012 — FIT HOUR-0 PREPROCESSOR")
    print("=" * 70)

    df = pd.read_csv(args.input_path, low_memory=False)
    print(f"\nLoaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    feature_cols = [
        c for c in df.columns if c not in EXCLUDE_COLS and c not in EXCLUDE_NOISY_COLS
    ]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    feature_types = identify_feature_types(df, columns=feature_cols)
    numeric_cols = feature_types["numeric"]
    binary_cols = feature_types["binary"]
    categorical_cols = feature_types["categorical"]
    int_cols = feature_types["int"]

    print(
        f"\n  Numeric: {len(numeric_cols)}  Binary: {len(binary_cols)}  "
        f"Categorical: {len(categorical_cols)}  Int: {len(int_cols)}"
    )

    preprocessor = TabularPreprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        int_cols=int_cols,
        binary_cols=binary_cols,
    )
    preprocessor.fit(df[feature_cols])
    print(f"\n  Fitted. Output features: {preprocessor.n_features}")

    # Reconstruction sanity check
    X = preprocessor.transform(df[feature_cols].iloc[:5])
    inv = preprocessor.inverse_transform(X)
    max_diff = max(
        (
            abs(df[feature_cols].iloc[:5][c] - inv[c]).max()
            for c in numeric_cols[:5]
            if c in inv.columns
        ),
        default=0.0,
    )
    print(f"  Reconstruction max error (sample): {max_diff:.6f}")

    payload = {
        "preprocessor": preprocessor,
        "all_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "binary_cols": binary_cols,
        "categorical_cols": categorical_cols,
        "int_cols": int_cols,
    }
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n✓ Saved to {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
