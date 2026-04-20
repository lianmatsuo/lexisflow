"""Autoregressive trajectory sampling for Forest-Flow.

This module implements trajectory generation for time series data,
where each timestep is generated conditionally on the previous timestep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable
from tqdm import tqdm

from .forest_flow import ForestFlow
from ..data.transformers import TabularPreprocessor


def sample_trajectory(
    model: ForestFlow,
    preprocessor: TabularPreprocessor,
    static_features: np.ndarray | None = None,
    n_trajectories: int = 1,
    max_hours: int = 48,
    target_cols: list[str] | None = None,
    condition_cols: list[str] | None = None,
    static_cols: list[str] | None = None,
    initial_state: np.ndarray | None = None,
    initial_strategy: str = "random",
    discharge_col: str | None = None,
    discharge_threshold: float = 0.5,
    random_state: int | None = None,
    verbose: bool = True,
) -> list[pd.DataFrame]:
    """Generate autoregressive trajectories (patient time series).

    Args:
        model: Fitted conditional ForestFlow model.
        preprocessor: Fitted TabularPreprocessor (must include all columns).
        static_features: Optional static features of shape (n_trajectories, n_static).
                        If None and model is conditional, zeros are used.
        n_trajectories: Number of independent trajectories to generate.
        max_hours: Maximum number of timesteps (patient hours) to generate.
        target_cols: List of target column names (dynamic features).
        condition_cols: List of condition column names (static + lag features).
        static_cols: List of static column names (subset of condition_cols).
                    These will be included in output and remain constant per trajectory.
        initial_state: Optional initial states of shape (n_trajectories, d_target).
                      If None, generated based on initial_strategy.
        initial_strategy: How to generate initial state if not provided:
                         - 'random': Sample from model without conditioning
                         - 'zero': Use zeros
        discharge_col: Optional column name for binary discharge indicator.
                      If provided, trajectory stops when value > discharge_threshold.
        discharge_threshold: Threshold for discharge (default 0.5).
        random_state: Random seed for reproducibility.
        verbose: If True, show progress bars.

    Returns:
        List of DataFrames, one per trajectory. Each DataFrame has columns:
        [timestep, static_cols..., target_cols...] where timestep is patient time (0, 1, 2, ...).
        Static columns remain constant within each trajectory.

    Example:
        >>> # Train conditional model
        >>> df_ar, target_cols, condition_cols = prepare_autoregressive_data(
        ...     df, 'subject_id', 'hours_in', static_cols=['age', 'gender']
        ... )
        >>> preprocessor.fit(df_ar)
        >>> X_target = preprocessor.transform(df_ar[target_cols])
        >>> X_condition = preprocessor.transform(df_ar[condition_cols])
        >>> model.fit(X_target, X_condition)
        >>>
        >>> # Generate trajectories
        >>> trajectories = sample_trajectory(
        ...     model, preprocessor,
        ...     n_trajectories=10,
        ...     max_hours=48,
        ...     target_cols=target_cols,
        ...     condition_cols=condition_cols,
        ...     static_cols=['age', 'gender']
        ... )
    """
    if not model.is_conditional_:
        raise ValueError("Model must be trained conditionally for trajectory sampling")

    rng = np.random.default_rng(random_state)

    # Initialize static features if not provided
    if static_features is None and model.d_condition_ > 0:
        # Use zeros for static features
        n_static = len([c for c in condition_cols if not c.endswith("_lag1")])
        static_features = np.zeros((n_trajectories, n_static), dtype=np.float32)

    trajectories = []

    iterator = range(n_trajectories)
    if verbose:
        iterator = tqdm(iterator, desc="Generating trajectories", unit="traj")

    for traj_idx in iterator:
        trajectory_steps = []

        # Step 0: Generate or use initial state
        if initial_state is not None:
            X_t = initial_state[traj_idx : traj_idx + 1].copy()
        elif initial_strategy == "random":
            # For conditional models, we need to provide dummy condition
            # This generates from P(X_0 | history=-1, static)
            if model.is_conditional_:
                # Use dummy lag features (all -1 or zeros) for initial state
                if static_features is not None:
                    static_part = static_features[traj_idx : traj_idx + 1]
                else:
                    static_part = np.zeros((1, 0), dtype=np.float32)

                # Dummy lag features (no history at t=0)
                n_lag_features = model.d_condition_ - static_part.shape[1]
                dummy_lag = np.full((1, n_lag_features), -1.0, dtype=np.float32)

                X_condition_init = np.concatenate([static_part, dummy_lag], axis=1)
                X_t = model.sample(
                    1, X_condition=X_condition_init, random_state=rng.integers(0, 2**31)
                )
            else:
                # IID model: sample without condition
                X_t = model.sample(1, random_state=rng.integers(0, 2**31))
        elif initial_strategy == "zero":
            X_t = np.zeros((1, model.d_target_), dtype=np.float32)
        else:
            raise ValueError(f"Unknown initial_strategy: {initial_strategy}")

        trajectory_steps.append(X_t.copy())

        # Steps 1 to max_hours: Autoregressive generation
        for hour in range(1, max_hours):
            # Prepare condition vector: [Static_Features, Lagged_Features]
            if static_features is not None:
                static_part = static_features[traj_idx : traj_idx + 1]
            else:
                static_part = np.empty((1, 0), dtype=np.float32)

            # Previous state becomes the lag features
            lagged_part = X_t  # Shape: (1, d_target)

            # Concatenate: [Static, Lag]
            X_condition = np.concatenate([static_part, lagged_part], axis=1)

            # Generate next state conditionally
            X_t = model.sample(
                1, X_condition=X_condition, random_state=rng.integers(0, 2**31)
            )

            trajectory_steps.append(X_t.copy())

            # Check for discharge if applicable
            if discharge_col is not None and target_cols is not None:
                # Find discharge column index
                try:
                    discharge_idx = target_cols.index(discharge_col)
                    if X_t[0, discharge_idx] > discharge_threshold:
                        if verbose:
                            tqdm.write(
                                f"  Trajectory {traj_idx} discharged at hour {hour}"
                            )
                        break
                except ValueError:
                    pass  # discharge_col not in target_cols

        # Stack trajectory: shape (n_steps, d_target)
        trajectory_arr = np.vstack(trajectory_steps)

        # Inverse transform: We need to handle this carefully
        # The preprocessor was fit on all columns (target + condition)
        # But trajectory_arr only contains target features
        # We need to reconstruct the full feature space for inverse_transform

        # Solution: Pad with zeros for condition columns, then take only target columns
        n_condition_features = preprocessor.n_features - trajectory_arr.shape[1]
        if n_condition_features > 0:
            # Pad with condition features (use zeros as placeholder)
            padding = np.zeros(
                (trajectory_arr.shape[0], n_condition_features), dtype=np.float32
            )
            full_arr = np.hstack([trajectory_arr, padding])
        else:
            full_arr = trajectory_arr

        # Inverse transform
        full_df = preprocessor.inverse_transform(full_arr)

        # Extract target columns
        if target_cols is not None:
            trajectory_df = full_df[target_cols].copy()
        else:
            trajectory_df = full_df

        # Note: Binary features are automatically rounded by preprocessor.inverse_transform()
        # if they were specified in binary_cols during preprocessor initialization

        # Add static features (these remain constant across all timesteps)
        if static_cols is not None and static_features is not None:
            # Inverse transform static features for this trajectory
            # We need to create a full feature array with static + dummy lag features
            static_arr = static_features[traj_idx : traj_idx + 1]

            # Determine how many lag features to pad
            n_lag_features = (
                preprocessor.n_features - static_arr.shape[1] - trajectory_arr.shape[1]
            )

            if n_lag_features > 0:
                lag_padding = np.zeros((1, n_lag_features), dtype=np.float32)
                target_padding = np.zeros(
                    (1, trajectory_arr.shape[1]), dtype=np.float32
                )
                static_full = np.hstack([target_padding, static_arr, lag_padding])
            else:
                target_padding = np.zeros(
                    (1, trajectory_arr.shape[1]), dtype=np.float32
                )
                static_full = np.hstack([target_padding, static_arr])

            # Inverse transform to get original static values
            static_df = preprocessor.inverse_transform(static_full)

            # Extract only the static columns
            static_values_df = static_df[static_cols]

            # Replicate static values for all timesteps
            for col in static_cols:
                trajectory_df[col] = static_values_df[col].iloc[0]

        # Add timestep column at the beginning
        trajectory_df.insert(0, "timestep", range(len(trajectory_df)))

        trajectories.append(trajectory_df)

    return trajectories


def sample_trajectory_with_initial_sampling(
    model: ForestFlow,
    preprocessor: TabularPreprocessor,
    initial_sampler: Callable[[int, int], np.ndarray],
    n_trajectories: int = 1,
    max_hours: int = 48,
    target_cols: list[str] | None = None,
    static_cols: list[str] | None = None,
    n_static_dims: int = 0,
    discharge_col: str | None = None,
    discharge_threshold: float = 0.5,
    random_state: int | None = None,
    verbose: bool = True,
) -> list[pd.DataFrame]:
    """Generate trajectories with custom initial state sampler.

    This variant allows full control over initial state generation,
    useful for training a separate model for admission states.

    Args:
        model: Fitted conditional ForestFlow model for P(X_t | X_{t-1}, Static).
        preprocessor: Fitted TabularPreprocessor.
        initial_sampler: Function that takes (n_samples, random_seed) and returns
                        initial states of shape (n_samples, d_target + d_static).
        n_trajectories: Number of trajectories to generate.
        max_hours: Maximum timesteps to generate.
        target_cols: List of target column names.
        static_cols: List of static column names.
        n_static_dims: Number of static feature dimensions.
        discharge_col: Optional discharge indicator column.
        discharge_threshold: Threshold for discharge.
        random_state: Random seed.
        verbose: Show progress.

    Returns:
        List of trajectory DataFrames.
    """
    rng = np.random.default_rng(random_state)

    # Sample initial states using custom sampler
    initial_full = initial_sampler(n_trajectories, rng.integers(0, 2**31))

    # Split into static and initial dynamic
    if n_static_dims > 0:
        static_features = initial_full[:, :n_static_dims]
        initial_dynamic = initial_full[:, n_static_dims:]
    else:
        static_features = None
        initial_dynamic = initial_full

    # Use main trajectory sampler
    return sample_trajectory(
        model=model,
        preprocessor=preprocessor,
        static_features=static_features,
        n_trajectories=n_trajectories,
        max_hours=max_hours,
        target_cols=target_cols,
        condition_cols=None,  # Not needed when providing initial_state
        initial_state=initial_dynamic,
        discharge_col=discharge_col,
        discharge_threshold=discharge_threshold,
        random_state=random_state,
        verbose=verbose,
    )


def prepare_training_data_from_trajectories(
    trajectories: list[pd.DataFrame],
    id_col: str = "trajectory_id",
    time_col: str = "timestep",
) -> pd.DataFrame:
    """Convert trajectory list back to flat DataFrame for analysis.

    Args:
        trajectories: List of trajectory DataFrames from sample_trajectory.
        id_col: Name for trajectory ID column.
        time_col: Name for time column (should already exist as 'timestep').

    Returns:
        Concatenated DataFrame with trajectory IDs.
    """
    dfs = []
    for i, traj_df in enumerate(trajectories):
        df_copy = traj_df.copy()
        df_copy[id_col] = i
        dfs.append(df_copy)

    return pd.concat(dfs, ignore_index=True)
