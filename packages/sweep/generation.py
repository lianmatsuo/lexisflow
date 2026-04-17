"""Autoregressive trajectory sampling + inverse-transform helpers.

The sweep and backbone-comparison drivers share a single generation path that
produces fixed-length ICU trajectories (``SEQUENCE_TIMESTEPS`` hours each) and
emits subject / hour columns needed by downstream TSTR and trajectory metrics.

``create_flat_dataframe`` inverse-transforms the target+condition pair back to
a patient-row DataFrame so quality/privacy metrics can operate on the raw
feature space.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.data import TabularPreprocessor

from .config import SEQUENCE_TIMESTEPS


# Real clinical binary features that should be rounded after inverse_transform.
BINARY_OUTPUT_COLUMNS = (
    "vent",
    "vaso",
    "adenosine",
    "dobutamine",
    "dopamine",
    "epinephrine",
    "isuprel",
    "milrinone",
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
    "colloid_bolus",
    "crystalloid_bolus",
    "nivdurations",
)

# ID columns dropped before quality/privacy comparisons so they do not dominate
# distance-based metrics or get treated as features by DOMIAS-like probes.
ID_COLUMNS_TO_DROP = ("subject_id", "hadm_id", "icustay_id", "hours_in")


def generate_synthetic_data(
    model: object,
    X_cond_sample: np.ndarray,
    n_samples: int,
    trajectory_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate fixed-length autoregressive trajectories with subject/hour IDs.

    All trajectories are ``SEQUENCE_TIMESTEPS`` hours long. The standalone
    sampling module (``packages/models/sampling.py``) supports early stopping
    via a discharge-column threshold, but no per-hour discharge indicator
    exists in the current feature set, so the sweep uses a simpler fixed-window
    loop.
    """
    n_condition = X_cond_sample.shape[1]
    n_target = getattr(model, "d_target_", None)
    if n_target is None:
        raise ValueError("Model missing d_target_ metadata required for AR generation")
    n_static = n_condition - int(n_target)
    if n_static < 0:
        raise ValueError(
            f"Condition width {n_condition} is smaller than target width {n_target}"
        )

    n_trajectories = max(1, int(np.ceil(n_samples / SEQUENCE_TIMESTEPS)))
    rng = np.random.default_rng(int(trajectory_seed))
    if n_static > 0:
        static_source = X_cond_sample[:, :n_static]
        idx = rng.choice(len(static_source), size=n_trajectories, replace=True)
        static_batch = static_source[idx]
    else:
        static_batch = np.zeros((n_trajectories, 0), dtype=np.float32)

    uses_row_conditioning = bool(getattr(model, "uses_row_conditioning_", True))
    print(
        "    Generating autoregressive trajectories "
        f"({n_trajectories:,} patients × {SEQUENCE_TIMESTEPS} timesteps)..."
    )

    if not uses_row_conditioning:
        return _generate_unconditional_baseline(
            model=model,
            rng=rng,
            static_batch=static_batch,
            n_trajectories=n_trajectories,
            n_target=int(n_target),
        )

    return _generate_row_conditioned(
        model=model,
        rng=rng,
        static_batch=static_batch,
        n_trajectories=n_trajectories,
        n_target=int(n_target),
    )


def _generate_unconditional_baseline(
    *,
    model: object,
    rng: np.random.Generator,
    static_batch: np.ndarray,
    n_trajectories: int,
    n_target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast path for unconditional baselines (e.g., CTGANAdapter)."""
    total_rows = n_trajectories * SEQUENCE_TIMESTEPS
    print(f"    Baseline fast path: single-pass sampling of {total_rows:,} rows...")
    X_target_full = model.sample(
        total_rows,
        random_state=int(rng.integers(0, 2**31 - 1)),
    ).astype(np.float32, copy=False)
    expected_shape = (total_rows, n_target)
    if X_target_full.shape != expected_shape:
        raise ValueError(
            "Unexpected target shape from unconditional baseline sampling: "
            f"got {X_target_full.shape}, expected {expected_shape}"
        )

    target_cube = X_target_full.reshape(SEQUENCE_TIMESTEPS, n_trajectories, n_target)
    cond_steps: list[np.ndarray] = []
    patient_ids: list[np.ndarray] = []
    timesteps: list[np.ndarray] = []
    lagged = np.full((n_trajectories, n_target), -1.0, dtype=np.float32)
    for hour in range(SEQUENCE_TIMESTEPS):
        X_cond = np.hstack([static_batch, lagged]).astype(np.float32, copy=False)
        cond_steps.append(X_cond)
        patient_ids.append(np.arange(n_trajectories, dtype=np.int64))
        timesteps.append(np.full(n_trajectories, hour, dtype=np.int64))
        lagged = target_cube[hour].astype(np.float32, copy=False)

    X_cond_full = np.vstack(cond_steps)
    subject_id = np.concatenate(patient_ids)
    hours_in = np.concatenate(timesteps)
    print("    Generation complete!")
    return X_target_full, X_cond_full, subject_id, hours_in


def _generate_row_conditioned(
    *,
    model: object,
    rng: np.random.Generator,
    static_batch: np.ndarray,
    n_trajectories: int,
    n_target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standard conditional loop used by HS3F / ForestFlow."""
    target_steps: list[np.ndarray] = []
    cond_steps: list[np.ndarray] = []
    patient_ids: list[np.ndarray] = []
    timesteps: list[np.ndarray] = []

    lagged = np.full((n_trajectories, n_target), -1.0, dtype=np.float32)
    for hour in range(SEQUENCE_TIMESTEPS):
        X_cond = np.hstack([static_batch, lagged]).astype(np.float32, copy=False)
        X_synth = model.sample(
            n_trajectories,
            X_condition=X_cond,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        target_steps.append(X_synth.astype(np.float32, copy=False))
        cond_steps.append(X_cond)
        patient_ids.append(np.arange(n_trajectories, dtype=np.int64))
        timesteps.append(np.full(n_trajectories, hour, dtype=np.int64))
        lagged = X_synth.astype(np.float32, copy=False)

    X_target_full = np.vstack(target_steps)
    X_cond_full = np.vstack(cond_steps)
    subject_id = np.concatenate(patient_ids)
    hours_in = np.concatenate(timesteps)
    print("    Generation complete!")
    return X_target_full, X_cond_full, subject_id, hours_in


def create_flat_dataframe(
    X_target: np.ndarray,
    X_condition: np.ndarray,
    preprocessor: TabularPreprocessor,
    target_indices: list[int],
    condition_indices: list[int],
    subject_id: np.ndarray | None = None,
    hours_in: np.ndarray | None = None,
) -> pd.DataFrame:
    """Inverse-transform target/condition arrays into a flat patient-row DataFrame."""
    if X_target.shape[1] != len(target_indices):
        raise ValueError(
            f"X_target width mismatch: got {X_target.shape[1]}, "
            f"expected {len(target_indices)}"
        )
    if X_condition.shape[1] != len(condition_indices):
        raise ValueError(
            f"X_condition width mismatch: got {X_condition.shape[1]}, "
            f"expected {len(condition_indices)}"
        )

    X_full = np.zeros((X_target.shape[0], preprocessor.n_features), dtype=np.float32)
    X_full[:, target_indices] = X_target
    X_full[:, condition_indices] = X_condition
    df = preprocessor.inverse_transform(X_full)

    for col in BINARY_OUTPUT_COLUMNS:
        if col in df.columns:
            df[col] = np.clip(np.round(df[col]), 0, 1).astype(int)

    if subject_id is not None:
        if len(subject_id) != len(df):
            raise ValueError("subject_id length must match synthetic row count")
        df.insert(0, "subject_id", subject_id.astype(int))
    if hours_in is not None:
        if len(hours_in) != len(df):
            raise ValueError("hours_in length must match synthetic row count")
        insert_idx = 1 if "subject_id" in df.columns else 0
        df.insert(insert_idx, "hours_in", hours_in.astype(int))

    return df


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns before quality/privacy comparison."""
    return df.drop(
        columns=[c for c in ID_COLUMNS_TO_DROP if c in df.columns],
        errors="ignore",
    )


__all__ = [
    "BINARY_OUTPUT_COLUMNS",
    "ID_COLUMNS_TO_DROP",
    "generate_synthetic_data",
    "create_flat_dataframe",
    "drop_id_columns",
]
