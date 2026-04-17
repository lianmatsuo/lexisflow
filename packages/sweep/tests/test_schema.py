"""Tests for the declarative sweep-result schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from packages.sweep.schema import (
    SWEEP_RESULT_COLUMNS,
    SEED_STAT_METRIC_MAP,
    append_result,
    build_error_row,
    build_result_row,
    ensure_results_schema,
    load_completed_runs,
)


CONTROL_COLS = [
    "nt",
    "n_noise",
    "hour0_train_time_sec",
    "autoregressive_train_time_sec",
    "total_train_time_sec",
]


def test_schema_has_expected_columns():
    assert SWEEP_RESULT_COLUMNS[: len(CONTROL_COLS)] == CONTROL_COLS
    for col in ("degenerate_flag", "timestamp", "error"):
        assert col in SWEEP_RESULT_COLUMNS
    # Each uncertainty metric emits exactly _std and _ci95 pairs
    for _src, column in SEED_STAT_METRIC_MAP:
        assert f"{column}_std" in SWEEP_RESULT_COLUMNS
        assert f"{column}_ci95" in SWEEP_RESULT_COLUMNS


def test_build_result_row_rounds_and_fills_missing():
    metrics = {
        "mortality_synth_roc_auc": 0.7234567,
        "los_synth_macro_f1": 0.5,
        "avg_ks_stat": 0.123456,
        "range_violation_pct": 1.2345,
        "degenerate_flag": 0,
    }
    uncertainty = {
        "synth_roc_auc_std": 0.03123456,
        "synth_roc_auc_ci95": 0.04123456,
    }
    row = build_result_row(
        nt=3,
        n_noise=5,
        hour0_train_time=10.123,
        autoregressive_train_time=20.456,
        metrics=metrics,
        uncertainty=uncertainty,
        trajectory_seed_count=3,
    )
    assert row["nt"] == 3
    assert row["n_noise"] == 5
    assert row["total_train_time_sec"] == 30.6
    assert row["synth_roc_auc"] == 0.7235
    # range_violation_pct rounds to 2 decimals
    assert row["range_violation_pct"] == 1.23
    # Missing metrics become NaN
    assert np.isnan(row["real_accuracy"])
    # Uncertainty rounds to 6 decimals
    assert row["synth_roc_auc_std"] == 0.031235
    # Degenerate flag and metadata populated
    assert row["degenerate_flag"] == 0
    assert isinstance(row["timestamp"], str)


def test_build_error_row_all_nan_except_meta():
    row = build_error_row(1, 2, RuntimeError("boom"), trajectory_seed_count=3)
    assert row["nt"] == 1
    assert row["n_noise"] == 2
    assert row["trajectory_seed_count"] == 3
    assert row["error"] == "boom"
    # Every metric column should be NaN
    for col in SWEEP_RESULT_COLUMNS:
        if col in {"nt", "n_noise", "trajectory_seed_count", "timestamp", "error"}:
            continue
        val = row[col]
        assert isinstance(val, float) and np.isnan(val), f"expected NaN for {col}"


def test_append_result_and_load_completed(tmp_path: Path):
    results_path = tmp_path / "sweep_results.csv"
    row1 = build_result_row(1, 1, 1.0, 2.0, {"mortality_synth_roc_auc": 0.6}, {}, 1)
    row2 = build_error_row(1, 2, RuntimeError("x"), trajectory_seed_count=1)
    append_result(results_path, row1)
    append_result(results_path, row2)

    df = pd.read_csv(results_path)
    assert list(df.columns) == SWEEP_RESULT_COLUMNS
    assert len(df) == 2
    completed = load_completed_runs(results_path)
    assert completed == {(1, 1), (1, 2)}


def test_ensure_results_schema_migrates_extra_and_missing(tmp_path: Path):
    results_path = tmp_path / "sweep_results.csv"
    # Write a legacy CSV with a subset of columns plus an extra
    legacy = pd.DataFrame(
        {
            "nt": [1, 3],
            "n_noise": [5, 7],
            "synth_roc_auc": [0.6, 0.7],
            "legacy_extra": ["a", "b"],
        }
    )
    legacy.to_csv(results_path, index=False)

    ensure_results_schema(results_path)
    df = pd.read_csv(results_path)

    assert list(df.columns) == SWEEP_RESULT_COLUMNS
    assert len(df) == 2
    assert "legacy_extra" not in df.columns
    # Preserved values survived the reordering
    assert df["synth_roc_auc"].tolist() == [0.6, 0.7]
    # Missing columns populated with NaN
    assert df["avg_ks_stat"].isna().all()
