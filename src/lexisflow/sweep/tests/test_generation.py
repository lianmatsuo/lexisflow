"""Tests for autoregressive trajectory sampling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lexisflow.sweep.config import SEQUENCE_TIMESTEPS
from lexisflow.data.transformers import TabularPreprocessor
from lexisflow.sweep.generation import (
    BINARY_OUTPUT_COLUMNS,
    ID_COLUMNS_TO_DROP,
    create_flat_dataframe,
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


def test_create_flat_dataframe_inverse_roundtrip():
    """Guards index placement into the full transformed vector before inverse."""
    df_fit = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0],
            "b": [10.0, 11.0, 12.0],
            "vent": [0, 1, 0],
        }
    )
    prep = TabularPreprocessor(
        numeric_cols=["a", "b"],
        binary_cols=["vent"],
        categorical_cols=[],
    )
    prep.fit(df_fit)
    X_src = prep.transform(df_fit)
    target_indices = [0, 1]
    condition_indices = [2]
    X_target = X_src[:, target_indices]
    X_cond = X_src[:, condition_indices]

    out = create_flat_dataframe(
        X_target,
        X_cond,
        prep,
        target_indices,
        condition_indices,
        subject_id=np.array([7, 8, 9], dtype=np.int64),
        hours_in=np.array([0, 1, 2], dtype=np.int64),
    )

    assert list(out.columns[:2]) == ["subject_id", "hours_in"]
    pd.testing.assert_frame_equal(
        out[["a", "b", "vent"]].reset_index(drop=True),
        df_fit.reset_index(drop=True),
        check_dtype=False,
    )
