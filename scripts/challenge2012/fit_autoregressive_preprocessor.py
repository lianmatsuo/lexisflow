#!/usr/bin/env python3
"""Fit the TabularPreprocessor on the Challenge 2012 autoregressive dataset.

Output:
    artifacts/challenge2012/autoregressive_preprocessor.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

from lexisflow.data import TabularPreprocessor, identify_feature_types, is_lagged


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
# Labels are included as static-like condition columns so they propagate into
# the inverse-transformed synthetic DataFrame (mirrors MIMIC behaviour — see
# MortalityTask / LOSTask which read these columns).
STATIC_COLS = [
    "Age",
    "Gender",
    "Weight",
    "ICUType",
    "hospital_expire_flag",
    "los_icu",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="data/challenge2012/processed/autoregressive_data.csv",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/challenge2012/autoregressive_preprocessor.pkl",
    )
    args = parser.parse_args()

    if not Path(args.input_path).exists():
        print(
            f"ERROR: {args.input_path} not found. Run prepare_autoregressive.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 70)
    print("CHALLENGE 2012 — FIT AUTOREGRESSIVE PREPROCESSOR")
    print("=" * 70)

    df = pd.read_csv(args.input_path, low_memory=False)
    print(f"\nLoaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    feature_cols = [
        c for c in df.columns if c not in EXCLUDE_COLS and c not in EXCLUDE_NOISY_COLS
    ]
    static_cols = [c for c in STATIC_COLS if c in feature_cols]
    lagged_cols = [c for c in feature_cols if is_lagged(c)]
    target_cols = [c for c in feature_cols if c not in static_cols and not is_lagged(c)]
    condition_cols = static_cols + lagged_cols
    all_cols = target_cols + condition_cols

    print(
        f"\n  Static: {len(static_cols)}  Target: {len(target_cols)}  "
        f"Lagged: {len(lagged_cols)}  Condition total: {len(condition_cols)}"
    )
    print(f"  Total unique feature columns: {len(all_cols)}")

    feature_types = identify_feature_types(df, columns=all_cols)
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
    preprocessor.fit(df[all_cols])
    print(f"\n  Fitted. Output features: {preprocessor.n_features}")

    # Build target/condition index maps in transformed space (mirrors MIMIC script)
    transformed_cols = (
        list(preprocessor.numeric_cols)
        + list(getattr(preprocessor, "binary_cols", []))
        + list(preprocessor.categorical_cols)
    )
    transformed_index = {c: i for i, c in enumerate(transformed_cols)}
    target_indices = [transformed_index[c] for c in target_cols]
    condition_indices = [transformed_index[c] for c in condition_cols]

    payload = {
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
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n✓ Saved to {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
