"""Unit tests for CTGANAdapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import synth_gen.models.ctgan_adapter as ctgan_adapter_module
from synth_gen.models.ctgan_adapter import CTGANAdapter


class _DummyCTGAN:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._fit_df: pd.DataFrame | None = None
        self.discrete_columns: list[str] = []

    def fit(self, df: pd.DataFrame, discrete_columns: list[str]):
        self._fit_df = df.copy()
        self.discrete_columns = list(discrete_columns)

    def sample(self, n_samples: int) -> pd.DataFrame:
        assert self._fit_df is not None
        out = pd.concat(
            [self._fit_df.iloc[[0]].copy() for _ in range(n_samples)],
            ignore_index=True,
        )
        if self.discrete_columns:
            out.loc[0, self.discrete_columns[0]] = "not_numeric"
        numeric_cols = [c for c in out.columns if c not in self.discrete_columns]
        if numeric_cols:
            out.loc[0, numeric_cols[0]] = np.nan
        return out


def test_fit_and_sample_roundtrip_with_discrete_features(monkeypatch):
    monkeypatch.setattr(ctgan_adapter_module, "CTGAN", _DummyCTGAN)

    X_target = np.array(
        [
            [0.1, 0],
            [0.2, 1],
            [0.3, 1],
            [0.4, 0],
        ],
        dtype=np.float32,
    )
    X_condition = np.array([[10.0], [11.0], [12.0], [13.0]], dtype=np.float32)

    adapter = CTGANAdapter(batch_size=12, pac=7, epochs=1, random_state=3)
    # Largest divisor of 12 not exceeding requested pac=7 is 6.
    assert adapter.pac == 6

    adapter.fit(X_target, X_condition=X_condition, feature_types=["q", "c", "q"])
    sampled = adapter.sample(n_samples=3, random_state=7)

    assert sampled.shape == (3, 2)
    assert np.isfinite(sampled).all()
    # Discrete target feature should be integer-coded on output.
    assert np.allclose(sampled[:, 1], np.round(sampled[:, 1]))


def test_sample_condition_warning_is_printed_once(monkeypatch, capsys):
    monkeypatch.setattr(ctgan_adapter_module, "CTGAN", _DummyCTGAN)

    X_target = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32)
    adapter = CTGANAdapter(batch_size=4, pac=2, epochs=1)
    adapter.fit(X_target)

    adapter.sample(n_samples=2, X_condition=np.zeros((2, 1), dtype=np.float32))
    first = capsys.readouterr().out
    assert "ignores per-row X_condition" in first

    adapter.sample(n_samples=2, X_condition=np.zeros((2, 1), dtype=np.float32))
    second = capsys.readouterr().out
    assert second == ""


def test_validation_errors(monkeypatch):
    monkeypatch.setattr(ctgan_adapter_module, "CTGAN", _DummyCTGAN)
    adapter = CTGANAdapter(batch_size=4, pac=2)

    with pytest.raises(ValueError, match="X must be a 2D array"):
        adapter.fit(np.array([1.0, 2.0], dtype=np.float32))

    X_target = np.array([[0.0], [1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="same number of rows"):
        adapter.fit(X_target, X_condition=np.array([[0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="does not match expected"):
        adapter.fit(X_target, feature_types=["q", "q"])


def test_sample_requires_fit():
    adapter = CTGANAdapter(batch_size=4, pac=2)
    with pytest.raises(RuntimeError, match="not fitted"):
        adapter.sample(n_samples=2)


def test_missing_ctgan_dependency_raises_clear_error(monkeypatch):
    monkeypatch.setattr(ctgan_adapter_module, "CTGAN", None)
    adapter = CTGANAdapter(batch_size=4, pac=2)
    with pytest.raises(ImportError, match="ctgan is required"):
        adapter.fit(np.array([[0.0], [1.0]], dtype=np.float32))
