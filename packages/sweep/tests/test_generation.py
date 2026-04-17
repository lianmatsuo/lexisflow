"""Tests for autoregressive trajectory sampling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from packages.sweep.config import SEQUENCE_TIMESTEPS
from packages.sweep.generation import (
    BINARY_OUTPUT_COLUMNS,
    ID_COLUMNS_TO_DROP,
    drop_id_columns,
    generate_synthetic_data,
)


class _DummyConditionalModel:
    """Minimal conditional-sample stub that emits deterministic outputs.

    Mirrors the subset of the HS3F/ForestFlow API the sweep generator uses:
    ``d_target_``, ``uses_row_conditioning_``, and ``sample(n, X_condition, random_state)``.
    """

    uses_row_conditioning_ = True

    def __init__(self, d_target: int):
        self.d_target_ = d_target

    def sample(self, n: int, X_condition: np.ndarray, random_state: int):
        del random_state
        return np.arange(n * self.d_target_, dtype=np.float32).reshape(
            n, self.d_target_
        )


class _DummyUnconditionalModel:
    uses_row_conditioning_ = False

    def __init__(self, d_target: int):
        self.d_target_ = d_target

    def sample(self, n: int, random_state: int):
        del random_state
        return np.full((n, self.d_target_), 0.5, dtype=np.float32)


def test_generate_synthetic_row_conditioned_shapes():
    d_target = 3
    n_static = 2
    n_condition = n_static + d_target  # model conditions on static + lagged
    model = _DummyConditionalModel(d_target=d_target)
    X_cond_sample = np.zeros((8, n_condition), dtype=np.float32)

    # Request ~2 trajectories worth of rows
    X_target, X_cond, sid, hours = generate_synthetic_data(
        model, X_cond_sample, n_samples=SEQUENCE_TIMESTEPS * 2, trajectory_seed=7
    )

    assert X_target.shape == (SEQUENCE_TIMESTEPS * 2, d_target)
    assert X_cond.shape == (SEQUENCE_TIMESTEPS * 2, n_condition)
    assert sid.shape == (SEQUENCE_TIMESTEPS * 2,)
    assert hours.shape == (SEQUENCE_TIMESTEPS * 2,)
    # Hours cycle 0..SEQUENCE_TIMESTEPS-1 across trajectories
    assert set(np.unique(hours).tolist()) == set(range(SEQUENCE_TIMESTEPS))
    # Subject ids are contiguous integers for the two synthetic patients
    assert set(np.unique(sid).tolist()) == {0, 1}


def test_generate_synthetic_unconditional_baseline():
    d_target = 2
    n_condition = d_target  # no static features
    model = _DummyUnconditionalModel(d_target=d_target)
    X_cond_sample = np.zeros((4, n_condition), dtype=np.float32)

    X_target, X_cond, sid, hours = generate_synthetic_data(
        model, X_cond_sample, n_samples=SEQUENCE_TIMESTEPS, trajectory_seed=3
    )

    assert X_target.shape == (SEQUENCE_TIMESTEPS, d_target)
    assert X_cond.shape == (SEQUENCE_TIMESTEPS, n_condition)
    # Single trajectory -> all subject ids are 0
    assert (sid == 0).all()
    # First hour's condition uses the -1 sentinel (no history available yet)
    assert (X_cond[0] == -1.0).all()


def test_drop_id_columns_removes_only_known_ids():
    df = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "hadm_id": [3, 4],
            "hours_in": [0, 1],
            "heart_rate": [80.0, 85.0],
        }
    )
    cleaned = drop_id_columns(df)
    for col in ID_COLUMNS_TO_DROP:
        assert col not in cleaned.columns
    assert "heart_rate" in cleaned.columns


def test_binary_output_columns_are_tuple_of_strings():
    # Prevents accidental demotion to ``list`` (mutable shared state).
    assert isinstance(BINARY_OUTPUT_COLUMNS, tuple)
    assert all(isinstance(c, str) for c in BINARY_OUTPUT_COLUMNS)
