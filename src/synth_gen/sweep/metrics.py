"""Seed-level metric aggregation for sweep cells.

The sweep runs TSTR/quality/privacy evaluation once per trajectory-sampling
seed (see :data:`TSTR_TRAJECTORY_SAMPLING_SEEDS`) and records a single row
per (nt, n_noise) cell containing:
  - the mean of each metric across seeds
  - per-metric ``_std`` and ``_ci95`` uncertainty columns (for metrics flagged
    as ``uncertainty=True`` in the schema)
  - a ``degenerate_flag`` recomputed from the averaged metrics
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .schema import SEED_STAT_METRIC_MAP


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def metric_stats(metric_dicts: list[dict], key: str) -> tuple[float, float, float]:
    """Compute (mean, std, CI95) over seed-level metric dicts for ``key``.

    Non-finite values are dropped. Returns NaNs for empty input, a zero-std
    result with NaN CI95 for a single value (no variance estimate), and a
    proper t-based CI95 otherwise.
    """
    vals = np.asarray(
        [_safe_float(m.get(key, np.nan)) for m in metric_dicts], dtype=float
    )
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(np.mean(vals))
    if len(vals) == 1:
        return mean, 0.0, float("nan")

    std = float(np.std(vals, ddof=1))
    stderr = std / np.sqrt(len(vals))
    t_crit = float(stats.t.ppf(0.975, df=len(vals) - 1))
    ci95 = float(t_crit * stderr) if np.isfinite(t_crit) else float("nan")
    return mean, std, ci95


def average_sweep_metrics(metric_dicts: list[dict]) -> dict[str, float]:
    """Mean numeric metrics across repeated TSTR evaluations.

    ``degenerate_flag`` is recomputed from the averaged mortality/LOS scores
    rather than averaged directly.
    """
    if not metric_dicts:
        return {}
    skip_keys = {"degenerate_flag"}
    all_keys: set[str] = set()
    for d in metric_dicts:
        all_keys.update(d.keys())
    out: dict[str, float] = {}
    for key in all_keys:
        if key in skip_keys:
            continue
        vals: list[float] = []
        for d in metric_dicts:
            if key not in d:
                vals.append(float("nan"))
                continue
            vals.append(_safe_float(d[key]))
        out[key] = float(np.nanmean(np.asarray(vals, dtype=np.float64)))

    mortality_auc = out.get("mortality_synth_roc_auc", float("nan"))
    los_f1 = out.get("los_synth_macro_f1", float("nan"))
    is_degenerate = (
        np.isfinite(mortality_auc)
        and np.isfinite(los_f1)
        and mortality_auc <= 0.55
        and los_f1 <= 0.15
    )
    out["degenerate_flag"] = 1.0 if is_degenerate else 0.0
    return out


def compute_seed_uncertainty(seed_metrics: list[dict]) -> dict[str, float]:
    """Produce the ``*_std`` / ``*_ci95`` values consumed by build_result_row."""
    out: dict[str, float] = {"trajectory_seed_count": float(len(seed_metrics))}
    for metric_key, output_key in SEED_STAT_METRIC_MAP:
        _, std, ci95 = metric_stats(seed_metrics, metric_key)
        out[f"{output_key}_std"] = std
        out[f"{output_key}_ci95"] = ci95
    return out


__all__ = [
    "metric_stats",
    "average_sweep_metrics",
    "compute_seed_uncertainty",
]
