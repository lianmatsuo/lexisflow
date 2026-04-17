#!/usr/bin/env python3
"""00c: Train Hour-0 Model - Train IID Forest-Flow for initial patient states.

This script trains an IID (non-autoregressive) Forest-Flow model to generate
hour-0 patient states: demographics + initial vital signs.

The model is fully unconditional (no conditioning on history, since there is no history).
It learns P(demographics, vitals_t0) directly from the hour-0 data distribution.

Usage:
    uv run python scripts/00c_train_hour0_model.py

Input:
    data/processed/hour0_data.csv
    models/hour0_preprocessor.pkl

Output:
    models/hour0_forest_flow.pkl
"""

import sys
from pathlib import Path
import pickle
import time
import argparse

import pandas as pd
import numpy as np

from packages.models import ForestFlow, HS3F


def main():
    parser = argparse.ArgumentParser(description="Train hour-0 IID generator")
    parser.add_argument(
        "--model-type",
        choices=["forest-flow", "hs3f"],
        default="hs3f",
        help="Generator backbone to train (default: hs3f)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("00c: TRAIN HOUR-0 MODEL (IID GENERATOR)")
    print("=" * 70)
    print(f"  Model type: {args.model_type}")

    # Configuration
    input_path = "data/processed/hour0_data.csv"
    preprocessor_path = "artifacts/hour0_preprocessor.pkl"
    model_output = "artifacts/hour0_forest_flow.pkl"

    # Model hyperparameters
    nt = 10  # Flow time steps (same as autoregressive model)
    n_noise = 10  # Noise samples (critical for quality)
    n_jobs = 4  # Parallel training
    batch_size = 5000  # Batch size for data iterator (memory efficient)
    max_rows = None  # Use all data (or set to e.g., 5000 for quick test)

    # Check if inputs exist
    if not Path(input_path).exists():
        print(f"\nError: {input_path} not found!")
        print("Run 00_prepare_hour0_data.py first.")
        sys.exit(1)

    if not Path(preprocessor_path).exists():
        print(f"\nError: {preprocessor_path} not found!")
        print("Run 00b_fit_hour0_preprocessor.py first.")
        sys.exit(1)

    # Load preprocessor
    print(f"\nLoading preprocessor from {preprocessor_path}...")
    with open(preprocessor_path, "rb") as f:
        preprocessor_data = pickle.load(f)

    preprocessor = preprocessor_data["preprocessor"]
    all_cols = preprocessor_data["all_cols"]

    print("  ✓ Preprocessor loaded")
    print(f"  Input features: {len(all_cols)}")
    print(f"  Output features: {preprocessor.n_features} (after encoding)")

    # Load hour-0 data
    print(f"\nLoading hour-0 data from {input_path}...")
    if max_rows:
        print(f"  (Using {max_rows:,} rows for training)")
        df = pd.read_csv(input_path, nrows=max_rows, low_memory=False)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Transform data
    print("\nTransforming data with preprocessor...")
    X = preprocessor.transform(df[all_cols])
    print(f"  Transformed shape: {X.shape}")
    print(f"  Features: {X.shape[1]}")

    # Train IID generator (no conditioning)
    print("\nTraining IID generator for hour-0 generation...")
    print("  Configuration:")
    print(f"    - nt (time steps): {nt}")
    print(f"    - n_noise (samples): {n_noise}")
    print(f"    - n_jobs (parallel): {n_jobs}")
    print(f"    - batch_size: {batch_size} (data iterator enabled)")
    print(f"    - Training samples: {X.shape[0]:,}")
    print("    - Mode: IID (no conditioning)")
    print("  This may take several minutes...")

    start_time = time.time()

    common_xgb = {
        "max_depth": 6,
        "n_estimators": 100,
        "learning_rate": 0.1,
    }
    if args.model_type == "hs3f":
        model = HS3F(
            nt=nt,
            n_noise=n_noise,
            n_jobs=n_jobs,
            random_state=42,
            solver="rk4",
            xgb_params=common_xgb,
        )
    else:
        print("\n  Using XGBoost data iterator (memory efficient, no duplication)")
        model = ForestFlow(
            nt=nt,
            n_noise=n_noise,
            n_jobs=n_jobs,
            random_state=42,
            use_data_iterator=True,  # Use XGBoost 2.0+ data iterator (avoids n_noise duplication)
            batch_size=batch_size,  # Process data in batches for memory efficiency
            xgb_params=common_xgb,
        )

    # Get feature types for XGBoost categorical support
    feature_types = preprocessor.get_feature_types()
    print("\n  Feature types for XGBoost:")
    print(f"    Quantitative: {feature_types.count('q')}")
    print(f"    Categorical: {feature_types.count('c')}")

    # For IID generation: X = X, X_condition = None
    print("\n  Training model (IID mode: no conditioning)...")
    model.fit(X, X_condition=None, feature_types=feature_types)

    elapsed = time.time() - start_time
    print(f"\n  ✓ Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Test generation
    print("\nTesting generation...")
    print("  Generating 5 synthetic hour-0 samples...")

    X_synthetic = model.sample(n_samples=5, X_condition=None)
    print(f"  Generated shape: {X_synthetic.shape}")

    # Inverse transform to original scale
    df_synthetic = preprocessor.inverse_transform(X_synthetic)
    print(f"  Inverse transformed shape: {df_synthetic.shape}")

    # Show sample of generated data
    print("\n  Sample generated hour-0 states (first 3 rows, first 5 columns):")
    print(df_synthetic.iloc[:3, :5].to_string())

    # Basic sanity checks
    print("\n  Sanity checks:")

    # Check for NaN
    nan_count = df_synthetic.isna().sum().sum()
    if nan_count > 0:
        print(f"    ⚠️  Warning: {nan_count} NaN values in generated data")
    else:
        print("    ✓ No NaN values")

    # Check for infinite values
    inf_count = np.isinf(df_synthetic.select_dtypes(include=[np.number]).values).sum()
    if inf_count > 0:
        print(f"    ⚠️  Warning: {inf_count} infinite values in generated data")
    else:
        print("    ✓ No infinite values")

    # Check numeric ranges (compare to training data)
    numeric_cols = preprocessor_data.get("numeric_cols", [])
    if numeric_cols:
        sample_col = numeric_cols[0]
        if sample_col in df.columns and sample_col in df_synthetic.columns:
            train_min = df[sample_col].min()
            train_max = df[sample_col].max()
            gen_min = df_synthetic[sample_col].min()
            gen_max = df_synthetic[sample_col].max()
            print(f"    {sample_col} range:")
            print(f"      Training: [{train_min:.2f}, {train_max:.2f}]")
            print(f"      Generated: [{gen_min:.2f}, {gen_max:.2f}]")

    # Save model with metadata
    print(f"\nSaving trained model to {model_output}...")
    Path(model_output).parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "model": model,
        "model_type": args.model_type,
        "preprocessor_path": preprocessor_path,
        "all_cols": all_cols,
        "n_features": preprocessor.n_features,
        "training_samples": X.shape[0],
        "nt": nt,
        "n_noise": n_noise,
        "training_time_seconds": elapsed,
    }

    with open(model_output, "wb") as f:
        pickle.dump(model_data, f)

    print(f"  ✓ Saved to {model_output}")

    # Show model size
    model_size_mb = Path(model_output).stat().st_size / (1024 * 1024)
    print(f"  Model size: {model_size_mb:.1f} MB")

    print("\n✅ Hour-0 model training complete!")
    print(f"   Model: {args.model_type} (IID)")
    print(f"   Training samples: {X.shape[0]:,}")
    print(f"   Input features: {len(all_cols)}")
    print(f"   Encoded features: {X.shape[1]}")
    print(f"   Training time: {elapsed:.1f}s")
    print(f"   Output: {model_output}")
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Run 05_evaluate_hour0_model.py to evaluate hour-0 quality")
    print("  2. Run 03_generate_synthetic.py --use-hour0 for fully synthetic rollouts")
    print("=" * 70)


if __name__ == "__main__":
    main()
