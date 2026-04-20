#!/usr/bin/env python3
"""Train the autoregressive generative model (HS3F or Forest-Flow) on preprocessed data.

Usage:
    uv run python scripts/train_autoregressive.py

Output:
    models/forest_flow_model.pkl
    models/preprocessor.pkl
"""

import sys
from pathlib import Path
import pickle
import argparse

import pandas as pd

from synth_gen.data import TabularPreprocessor, split_static_dynamic
from synth_gen.models import ForestFlow, HS3F


def main():
    parser = argparse.ArgumentParser(
        description="Train generative model on autoregressive data"
    )
    parser.add_argument(
        "--model-type",
        choices=["forest-flow", "hs3f"],
        default="hs3f",
        help="Generator backbone to train (default: hs3f)",
    )
    parser.add_argument(
        "--nt",
        type=int,
        default=50,
        help="Flow time levels (default: 50)",
    )
    parser.add_argument(
        "--n-noise",
        type=int,
        default=100,
        help="Noise multipliers per row during training (default: 100)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10000,
        help="Training rows to load from autoregressive CSV; use 0 for all rows (default: 10000)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Parallel jobs for XGBoost-backed training (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for data iterator mode (default: 500)",
    )
    parser.add_argument(
        "--disable-data-iterator",
        action="store_true",
        help="Disable ForestFlow data iterator and materialize duplicated training matrix",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("02: MODEL TRAINING")
    print("=" * 70)
    print(f"  Model type: {args.model_type}")

    # Configuration
    input_path = "data/processed/autoregressive_data.csv"
    model_output = "artifacts/forest_flow_model.pkl"
    preprocessor_output = "artifacts/preprocessor.pkl"
    preprocessor_full_path = (
        "artifacts/preprocessor_full.pkl"  # Pre-fitted on full data
    )

    # Model hyperparameters
    nt = args.nt
    n_noise = args.n_noise
    batch_size = args.batch_size
    max_rows = None if args.max_rows <= 0 else args.max_rows
    n_jobs = args.n_jobs
    use_data_iterator = not args.disable_data_iterator

    # Check if input exists
    if not Path(input_path).exists():
        print(f"\nError: {input_path} not found!")
        print("Run scripts/prepare_autoregressive.py first to prepare data.")
        sys.exit(1)

    # Load autoregressive data
    print(f"\nLoading autoregressive data from {input_path}...")
    if max_rows:
        print(f"  (Using {max_rows:,} rows for training)")
        df = pd.read_csv(input_path, nrows=max_rows, low_memory=False)
    else:
        df = pd.read_csv(input_path, low_memory=False)

    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("  Training config:")
    print(f"    nt={nt}, n_noise={n_noise}, n_jobs={n_jobs}, batch_size={batch_size}")
    print(f"    data_iterator={'on' if use_data_iterator else 'off'}")

    # Check for pre-fitted preprocessor
    use_prefitted = Path(preprocessor_full_path).exists()

    if use_prefitted:
        print(f"\n✓ Found pre-fitted preprocessor: {preprocessor_full_path}")
        print("  Loading preprocessor fitted on FULL dataset...")
        with open(preprocessor_full_path, "rb") as f:
            preprocessor_data = pickle.load(f)

        preprocessor = preprocessor_data["preprocessor"]
        target_cols = preprocessor_data["target_cols"]
        condition_cols = preprocessor_data["condition_cols"]
        target_indices = preprocessor_data.get("target_indices")
        condition_indices = preprocessor_data.get("condition_indices")
        all_cols = preprocessor_data["all_cols"]

        print(f"  ✓ Preprocessor fitted on {preprocessor._n_samples_seen:,} samples")
        print(f"  Target features: {len(target_cols)}")
        print(f"  Condition features: {len(condition_cols)}")

        # Transform training data
        print("\nTransforming training data...")
        X = preprocessor.transform(df[all_cols])
        print(f"  Input features: {len(all_cols)}")
        print(f"  Output features: {X.shape[1]} (preprocessed)")

    else:
        print("\n⚠️  Warning: Pre-fitted preprocessor not found!")
        print(f"   Looking for: {preprocessor_full_path}")
        print("   Fitting preprocessor on training sample instead (NOT RECOMMENDED)")
        print(
            "   → Run scripts/fit_autoregressive_preprocessor.py first to fit on full data"
        )
        print()

        # Identify columns
        print("Identifying feature columns...")
        static_cols, dynamic_cols = split_static_dynamic(df)

        # Autoregressive columns: dynamic features are targets, lagged versions are conditions
        target_cols = dynamic_cols
        condition_cols = static_cols + [
            col for col in df.columns if col.endswith("_lag1")
        ]
        all_cols = target_cols + condition_cols

        print(f"  Target features: {len(target_cols)}")
        print(f"  Condition features: {len(condition_cols)}")

        # Identify numeric vs categorical columns
        numeric_cols = []
        categorical_cols = []
        int_cols = []

        for col in all_cols:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                numeric_cols.append(col)
                if pd.api.types.is_integer_dtype(dtype):
                    int_cols.append(col)
            else:
                categorical_cols.append(col)

        # Fit preprocessor on sample (suboptimal)
        print("\nFitting preprocessor on SAMPLE (may miss rare values)...")
        preprocessor = TabularPreprocessor(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            int_cols=int_cols,
        )
        preprocessor.fit(df[all_cols])
        n_target = len(target_cols)
        target_indices = None
        condition_indices = None

        print(f"  Input features: {len(all_cols)}")
        print(f"  Output features: {preprocessor.n_features} (preprocessed)")

        # Transform
        X = preprocessor.transform(df[all_cols])

    # Train Forest-Flow
    print(f"\nTraining model (nt={nt}, n_noise={n_noise})...")
    print("  This may take several minutes...")
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
        if use_data_iterator:
            print("  Using XGBoost data iterator (memory efficient, no duplication)")
        else:
            print("  Data iterator disabled (materialized training matrix)")
        model = ForestFlow(
            nt=nt,
            n_noise=n_noise,
            n_jobs=n_jobs,  # Parallel training
            random_state=42,
            use_data_iterator=use_data_iterator,  # Use XGBoost 2.0+ data iterator (avoids n_noise duplication)
            batch_size=batch_size,  # Process data in batches for memory efficiency
            xgb_params=common_xgb,
        )

    # Build index-based target/condition split in transformed feature space.
    if target_indices is None or condition_indices is None:
        target_indices, condition_indices = preprocessor.split_indices(
            target_cols, condition_cols
        )
    n_target = len(target_indices)
    X_target = X[:, target_indices]
    X_cond = X[:, condition_indices]

    # Get feature types for XGBoost categorical support
    full_feature_types = preprocessor.get_feature_types()
    feature_types = [
        full_feature_types[i] for i in (target_indices + condition_indices)
    ]
    print("\n  Feature types for XGBoost categorical support:")
    print(f"    Total features: {len(feature_types)}")
    print(f"    Quantitative: {feature_types.count('q')}")
    print(f"    Categorical: {feature_types.count('c')}")

    # Note: ForestFlow.fit() expects X (target) and X_condition (optional)
    model.fit(X_target, X_condition=X_cond, feature_types=feature_types)

    print("  ✓ Training complete!")

    # Save model and preprocessor
    print(f"\nSaving model to {model_output}...")
    Path(model_output).parent.mkdir(parents=True, exist_ok=True)

    with open(model_output, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "model_type": args.model_type,
            },
            f,
        )

    print(f"Saving preprocessor to {preprocessor_output}...")
    with open(preprocessor_output, "wb") as f:
        pickle.dump(
            {
                "preprocessor": preprocessor,
                "target_cols": target_cols,
                "condition_cols": condition_cols,
                "n_target": n_target,
                "target_indices": target_indices,
                "condition_indices": condition_indices,
            },
            f,
        )

    print("\n✅ Training complete!")
    print(f"   Model: {model_output}")
    print(f"   Preprocessor: {preprocessor_output}")
    print("\n" + "=" * 70)
    print("Next step: Run scripts/generate.py to generate data")
    print("=" * 70)


if __name__ == "__main__":
    main()
