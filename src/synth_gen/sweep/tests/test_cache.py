"""Tests for the transformed-data cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from synth_gen.sweep.cache import (
    build_cache_signature,
    load_transformed_cache,
    save_transformed_cache,
)


def _write_dummy_csv(path: Path, n_rows: int = 8, n_cols: int = 4) -> list[str]:
    cols = [f"c{i}" for i in range(n_cols)]
    df = pd.DataFrame(np.arange(n_rows * n_cols).reshape(n_rows, n_cols), columns=cols)
    df.to_csv(path, index=False)
    return cols


def test_cache_roundtrip(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    cols = _write_dummy_csv(csv_path)

    X_target = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    X_cond = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    real_quality = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    sig = build_cache_signature(
        csv_path, cols, n_target=2, target_indices=[0, 1], condition_indices=[2, 3]
    )
    cache_dir = tmp_path / "cache"

    assert load_transformed_cache(cache_dir, sig) is None

    save_transformed_cache(cache_dir, sig, X_target, X_cond, real_quality, n_rows=2)
    cached = load_transformed_cache(cache_dir, sig)
    assert cached is not None
    loaded_target, loaded_cond, loaded_quality, n_rows = cached
    assert n_rows == 2
    np.testing.assert_allclose(np.asarray(loaded_target), X_target)
    np.testing.assert_allclose(np.asarray(loaded_cond), X_cond)
    pd.testing.assert_frame_equal(loaded_quality, real_quality)


def test_cache_invalidates_on_signature_change(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    cols = _write_dummy_csv(csv_path)

    sig = build_cache_signature(
        csv_path, cols, n_target=2, target_indices=[0, 1], condition_indices=[2, 3]
    )
    cache_dir = tmp_path / "cache"
    save_transformed_cache(
        cache_dir,
        sig,
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        pd.DataFrame({"a": [0]}),
        n_rows=1,
    )

    # Different target split -> incompatible signature -> cache miss
    sig_different = build_cache_signature(
        csv_path, cols, n_target=1, target_indices=[0], condition_indices=[1, 2, 3]
    )
    assert load_transformed_cache(cache_dir, sig_different) is None
