"""Tests for TabularPreprocessor.transformed_columns / split_indices."""

from __future__ import annotations

import pandas as pd
import pytest

from packages.data import TabularPreprocessor


def _fit_small():
    pp = TabularPreprocessor(
        numeric_cols=["age", "hr"],
        binary_cols=["vent"],
        categorical_cols=["gender"],
    )
    df = pd.DataFrame(
        {
            "age": [50.0, 60.0, 70.0],
            "hr": [80.0, 90.0, 100.0],
            "vent": [0, 1, 0],
            "gender": ["M", "F", "M"],
        }
    )
    pp.fit(df)
    return pp


def test_transformed_columns_order():
    pp = _fit_small()
    assert pp.transformed_columns() == ["age", "hr", "vent", "gender"]


def test_split_indices_basic():
    pp = _fit_small()
    target_idx, cond_idx = pp.split_indices(["age", "vent"], ["hr", "gender"])
    assert target_idx == [0, 2]
    assert cond_idx == [1, 3]


def test_split_indices_missing_raises():
    pp = _fit_small()
    with pytest.raises(ValueError, match="Missing target cols"):
        pp.split_indices(["age", "nonexistent"], ["hr"])


def test_split_indices_overlap_raises():
    pp = _fit_small()
    with pytest.raises(ValueError, match="overlap"):
        pp.split_indices(["age", "hr"], ["hr", "vent"])


def test_split_indices_incomplete_coverage_raises():
    pp = _fit_small()
    with pytest.raises(ValueError, match="cover transformed feature space"):
        pp.split_indices(["age"], ["hr"])
