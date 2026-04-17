"""Tests for seed-level metric aggregation."""

from __future__ import annotations

import numpy as np

from packages.sweep.metrics import (
    average_sweep_metrics,
    compute_seed_uncertainty,
    metric_stats,
)


def test_metric_stats_empty():
    mean, std, ci = metric_stats([], "x")
    assert np.isnan(mean) and np.isnan(std) and np.isnan(ci)


def test_metric_stats_single_value():
    mean, std, ci = metric_stats([{"x": 0.5}], "x")
    assert mean == 0.5
    assert std == 0.0
    assert np.isnan(ci)


def test_metric_stats_multiple_values():
    dicts = [{"x": 0.1}, {"x": 0.3}, {"x": 0.5}]
    mean, std, ci = metric_stats(dicts, "x")
    assert abs(mean - 0.3) < 1e-12
    # sample std of [0.1, 0.3, 0.5]
    assert std == 0.2
    assert ci > 0


def test_metric_stats_drops_nonfinite():
    dicts = [{"x": 0.1}, {"x": float("nan")}, {"x": float("inf")}, {"x": 0.5}]
    mean, _std, _ci = metric_stats(dicts, "x")
    assert mean == 0.3


def test_average_sweep_metrics_recomputes_degenerate_flag():
    dicts = [
        {"mortality_synth_roc_auc": 0.5, "los_synth_macro_f1": 0.1, "other": 1.0},
        {"mortality_synth_roc_auc": 0.5, "los_synth_macro_f1": 0.1, "other": 3.0},
    ]
    out = average_sweep_metrics(dicts)
    assert out["mortality_synth_roc_auc"] == 0.5
    assert out["other"] == 2.0
    # Both averages hit the degenerate threshold
    assert out["degenerate_flag"] == 1.0

    dicts_strong = [{"mortality_synth_roc_auc": 0.9, "los_synth_macro_f1": 0.6}]
    out_strong = average_sweep_metrics(dicts_strong)
    assert out_strong["degenerate_flag"] == 0.0


def test_compute_seed_uncertainty_keys_match_schema():
    # Just make sure it produces the expected key pattern.
    dicts = [
        {"mortality_synth_roc_auc": 0.7, "avg_ks_stat": 0.1},
        {"mortality_synth_roc_auc": 0.72, "avg_ks_stat": 0.12},
        {"mortality_synth_roc_auc": 0.68, "avg_ks_stat": 0.14},
    ]
    out = compute_seed_uncertainty(dicts)
    assert out["trajectory_seed_count"] == 3.0
    assert "synth_roc_auc_std" in out
    assert "synth_roc_auc_ci95" in out
    assert "avg_ks_stat_std" in out
    assert "avg_ks_stat_ci95" in out
