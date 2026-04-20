#!/usr/bin/env python3
"""Generate synthetic ICU patient trajectories from trained models.

This script uses trained Forest-Flow models to generate synthetic ICU patient data.
It supports two modes:

1. Hour-0 Model Mode (Fully Synthetic):
   - Uses hour-0 model to generate initial patient states (demographics + vitals)
   - Uses autoregressive model to generate trajectories from hour-0 states
   - No real data needed as input (fully synthetic)

2. Real Initial Conditions Mode (Original):
   - Uses real patient data for initial conditions
   - Uses autoregressive model to generate trajectories

Usage:
    # Fully synthetic (with hour-0 model)
    uv run python scripts/generate.py --use-hour0

    # Using real initial conditions (original)
    uv run python scripts/generate.py

Output:
    results/synthetic_patients.csv
"""

import sys
from pathlib import Path
import pickle
import argparse

import pandas as pd

from synth_gen.models import sample_trajectory


def load_model_artifact(path: str):
    """Load model artifact from pickle supporting old/new formats."""
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact.get("model_type", "forest-flow")
    return artifact, "forest-flow"


def generate_hour0_states(hour0_model, hour0_preprocessor, n_patients):
    """Generate initial patient states using hour-0 model.

    Args:
        hour0_model: Trained hour-0 Forest-Flow model
        hour0_preprocessor: Hour-0 preprocessor
        n_patients: Number of patients to generate

    Returns:
        DataFrame with demographics + hour-0 vitals
    """
    print(f"\n  Generating {n_patients} hour-0 patient states...")

    # Sample from hour-0 model (IID)
    X_hour0 = hour0_model.sample(n_samples=n_patients, X_condition=None)

    # Inverse transform to original scale
    df_hour0 = hour0_preprocessor.inverse_transform(X_hour0)

    return df_hour0


def convert_hour0_to_autoregressive_conditions(df_hour0, target_cols, condition_cols):
    """Convert hour-0 states to autoregressive initial conditions.

    This extracts static features and creates lag1 features from hour-0 vitals.

    Args:
        df_hour0: Hour-0 states (demographics + vitals)
        target_cols: Target columns for autoregressive model
        condition_cols: Condition columns for autoregressive model

    Returns:
        DataFrame with proper autoregressive format (static + lag1 features)
    """
    print("\n  Converting hour-0 states to autoregressive format...")

    # Identify static vs lagged columns
    static_cols = [col for col in condition_cols if not col.endswith("_lag1")]

    # Create initial conditions DataFrame
    initial_conditions = pd.DataFrame()

    # Add static features (keep as-is)
    for col in static_cols:
        if col in df_hour0.columns:
            initial_conditions[col] = df_hour0[col]

    # Create lag1 features from hour-0 vitals
    for col in target_cols:
        lag_col = f"{col}_lag1"
        if lag_col in condition_cols:
            if col in df_hour0.columns:
                # Hour-0 value becomes lag1 for hour-1
                initial_conditions[lag_col] = df_hour0[col]
            else:
                # If feature not in hour-0 data, use -1 (missing)
                initial_conditions[lag_col] = -1.0

    # Add synthetic subject IDs
    initial_conditions["subject_id"] = [f"synth_{i}" for i in range(len(df_hour0))]

    print(f"    ✓ Created {len(initial_conditions)} initial conditions")
    print(f"    Static features: {len(static_cols)}")
    print(f"    Lag1 features: {sum(1 for c in condition_cols if c.endswith('_lag1'))}")

    return initial_conditions


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient trajectories"
    )
    parser.add_argument(
        "--use-hour0",
        action="store_true",
        help="Use hour-0 model for fully synthetic generation (default: use real initial conditions)",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=100,
        help="Number of synthetic patients to generate (default: 100)",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=10,
        help="Trajectory length in hours (default: 10)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("03: GENERATE SYNTHETIC DATA")
    if args.use_hour0:
        print("    MODE: Fully Synthetic (Hour-0 Model)")
    else:
        print("    MODE: Real Initial Conditions")
    print("=" * 70)

    # Configuration
    model_path = "artifacts/forest_flow_model.pkl"
    preprocessor_path = "artifacts/preprocessor.pkl"
    hour0_model_path = "artifacts/hour0_forest_flow.pkl"
    hour0_preprocessor_path = "artifacts/hour0_preprocessor.pkl"
    real_data_path = "data/processed/autoregressive_data.csv"
    output_path = "results/synthetic_patients.csv"

    n_patients = args.n_patients
    n_timesteps = args.n_timesteps
    nt_sample = 50  # Sampling discretization (higher = better quality)

    # Check inputs exist
    if not Path(model_path).exists():
        print(f"\nError: {model_path} not found!")
        print("Run scripts/train_autoregressive.py first to train a model.")
        sys.exit(1)

    if not Path(preprocessor_path).exists():
        print(f"\nError: {preprocessor_path} not found!")
        print("Run scripts/train_autoregressive.py first to train a model.")
        sys.exit(1)

    # Load autoregressive model and preprocessor
    print(f"\nLoading autoregressive model from {model_path}...")
    model, model_type = load_model_artifact(model_path)
    print(f"  Loaded model type: {model_type}")

    print(f"Loading autoregressive preprocessor from {preprocessor_path}...")
    with open(preprocessor_path, "rb") as f:
        prep_data = pickle.load(f)
        preprocessor = prep_data["preprocessor"]
        target_cols = prep_data["target_cols"]
        condition_cols = prep_data["condition_cols"]

    print(f"  Target features: {len(target_cols)}")
    print(f"  Condition features: {len(condition_cols)}")

    # Generate or load initial conditions
    if args.use_hour0:
        # Hour-0 model mode: Fully synthetic
        print(f"\n{'='*70}")
        print("PHASE 1: GENERATE HOUR-0 STATES")
        print(f"{'='*70}")

        # Check hour-0 model exists
        if not Path(hour0_model_path).exists():
            print(f"\nError: {hour0_model_path} not found!")
            print("Run scripts/train_hour0.py first to train hour-0 model.")
            print("Or omit '--use-hour0' to use real initial conditions.")
            sys.exit(1)

        if not Path(hour0_preprocessor_path).exists():
            print(f"\nError: {hour0_preprocessor_path} not found!")
            sys.exit(1)

        # Load hour-0 model and preprocessor
        print(f"\nLoading hour-0 model from {hour0_model_path}...")
        hour0_model, hour0_model_type = load_model_artifact(hour0_model_path)
        print(f"  Loaded hour-0 model type: {hour0_model_type}")

        print(f"Loading hour-0 preprocessor from {hour0_preprocessor_path}...")
        with open(hour0_preprocessor_path, "rb") as f:
            hour0_prep_data = pickle.load(f)
            hour0_preprocessor = hour0_prep_data["preprocessor"]

        # Generate hour-0 states
        df_hour0 = generate_hour0_states(hour0_model, hour0_preprocessor, n_patients)

        # Convert to autoregressive format
        initial_conditions = convert_hour0_to_autoregressive_conditions(
            df_hour0, target_cols, condition_cols
        )

        print(
            f"\n  ✓ Generated {len(initial_conditions)} fully synthetic initial conditions"
        )

    else:
        # Original mode: Use real initial conditions
        print("\nLoading real data for initial conditions...")
        df_real = pd.read_csv(real_data_path, nrows=n_patients)

        # Extract initial conditions (first timestep for each patient)
        initial_conditions = df_real.groupby("subject_id").first().reset_index()
        initial_conditions = initial_conditions.head(n_patients)

        print(f"  Using {len(initial_conditions)} real initial conditions")

    # Generate synthetic trajectories
    print(f"\n{'='*70}")
    print("PHASE 2: GENERATE AUTOREGRESSIVE TRAJECTORIES")
    print(f"{'='*70}")
    print(f"\nGenerating {n_patients} synthetic patient trajectories...")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Sampling resolution: nt={nt_sample}")
    print("  This may take a few minutes...")

    synth_df = sample_trajectory(
        model=model,
        preprocessor=preprocessor,
        initial_conditions=initial_conditions,
        n_timesteps=n_timesteps,
        target_cols=target_cols,
        condition_cols=condition_cols,
        nt=nt_sample,
        static_cols=[col for col in condition_cols if not col.endswith("_lag1")],
        id_col="subject_id",
        time_col="hours_in",
    )

    print(f"  Generated: {synth_df.shape[0]:,} rows × {synth_df.shape[1]} columns")
    print("  ✓ Binary features automatically rounded to {0, 1} by preprocessor")

    # Basic statistics
    print("\nSynthetic data summary:")
    print(f"  Unique patients: {synth_df['subject_id'].nunique()}")
    print(
        f"  Average trajectory length: {synth_df.groupby('subject_id').size().mean():.1f} timesteps"
    )
    print(f"  Total observations: {len(synth_df):,}")

    # Save synthetic data
    print(f"\nSaving synthetic data to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(output_path, index=False)

    print("\n✅ Generation complete!")
    if args.use_hour0:
        print("   Mode: Fully Synthetic (Hour-0 + Autoregressive)")
    else:
        print("   Mode: Real Initial Conditions + Autoregressive")
    print(f"   Output: {output_path}")
    print(f"   Shape: {synth_df.shape}")
    print("\n" + "=" * 70)
    print("Next step: evaluate synthetic data quality (see src/synth_gen/evaluation)")
    print("=" * 70)


if __name__ == "__main__":
    main()
