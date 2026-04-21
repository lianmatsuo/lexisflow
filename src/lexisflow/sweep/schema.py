"""Declarative schema for sweep result rows.

One ``SweepField`` entry per CSV column captures:
  - source key in the raw metrics dict returned by evaluate_cell
  - rounding precision
  - whether an uncertainty ``_std`` / ``_ci95`` pair is emitted

This collapses ~800 lines of hand-written ``round(...)`` boilerplate plus the
parallel column-order list in the old ``run_sweep.py`` into a single source of
truth. Adding a new metric only requires adding a field declaration here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SweepField:
    """One column in the sweep results CSV."""

    column: str
    source: str | None  # key in the raw metrics dict; None for control columns
    precision: int | None = 4  # round to this many decimals; None disables rounding
    uncertainty: bool = False  # emit ``{column}_std`` and ``{column}_ci95`` columns
    integer: bool = False  # cast to int


# Reserved non-metric columns written per row
_CONTROL_COLUMNS: tuple[str, ...] = (
    "nt",
    "n_noise",
    "hour0_train_time_sec",
    "autoregressive_train_time_sec",
    "total_train_time_sec",
)

# Metric fields. Order here defines order in the CSV (after the control cols).
# ``source`` is the key the evaluator emits; ``column`` is the CSV header.
_METRIC_FIELDS: tuple[SweepField, ...] = (
    # Mortality (backwards-compatible names)
    SweepField("synth_accuracy", "mortality_synth_accuracy", 4, uncertainty=True),
    SweepField(
        "synth_balanced_accuracy",
        "mortality_synth_balanced_accuracy",
        4,
        uncertainty=True,
    ),
    SweepField("synth_f1", "mortality_synth_f1", 4, uncertainty=True),
    SweepField("synth_roc_auc", "mortality_synth_roc_auc", 4, uncertainty=True),
    SweepField("real_accuracy", "mortality_real_accuracy", 4),
    SweepField("real_balanced_accuracy", "mortality_real_balanced_accuracy", 4),
    SweepField("real_f1", "mortality_real_f1", 4),
    SweepField("real_roc_auc", "mortality_real_roc_auc", 4),
    # Length-of-stay (LOS)
    SweepField("los_synth_accuracy", "los_synth_accuracy", 4, uncertainty=True),
    SweepField("los_synth_macro_f1", "los_synth_macro_f1", 4, uncertainty=True),
    SweepField("los_synth_roc_auc", "los_synth_roc_auc", 4),
    SweepField("los_real_accuracy", "los_real_accuracy", 4),
    SweepField("los_real_macro_f1", "los_real_macro_f1", 4),
    SweepField("los_real_roc_auc", "los_real_roc_auc", 4),
    # Quality
    SweepField("avg_ks_stat", "avg_ks_stat", 4, uncertainty=True),
    SweepField("corr_frobenius", "corr_frobenius", 4, uncertainty=True),
    SweepField("range_violation_pct", "range_violation_pct", 2, uncertainty=True),
    # Trajectory
    SweepField("autocorr_distance", "autocorr_distance", 4, uncertainty=True),
    SweepField("stay_length_ks", "stay_length_ks", 4, uncertainty=True),
    SweepField("transition_mse_ratio", "transition_mse_ratio", 4, uncertainty=True),
    SweepField("temporal_corr_drift", "temporal_corr_drift", 4, uncertainty=True),
    # Privacy
    SweepField("dcr_median", "dcr_median", 6, uncertainty=True),
    SweepField("dcr_p05", "dcr_p05", 6),
    SweepField("dcr_exact_match_rate", "dcr_exact_match_rate", 6),
    SweepField(
        "dcr_baseline_protection", "dcr_baseline_protection", 6, uncertainty=True
    ),
    SweepField("dcr_overfitting_protection", "dcr_overfitting_protection", 6),
    SweepField("dcr_closer_to_training_pct", "dcr_closer_to_training_pct", 6),
    SweepField("mia_roc_auc", "mia_roc_auc", 6, uncertainty=True),
    SweepField("mia_average_precision", "mia_average_precision", 6),
    SweepField("mia_attacker_advantage", "mia_attacker_advantage", 6),
)

# Trailing meta columns
_TRAILING_COLUMNS: tuple[str, ...] = (
    "trajectory_seed_count",
    "degenerate_flag",
    "timestamp",
    "error",
)


def _uncertainty_columns() -> list[str]:
    cols: list[str] = []
    for field in _METRIC_FIELDS:
        if field.uncertainty:
            cols.append(f"{field.column}_std")
            cols.append(f"{field.column}_ci95")
    return cols


def _build_result_columns() -> list[str]:
    cols = list(_CONTROL_COLUMNS)
    cols.extend(f.column for f in _METRIC_FIELDS)
    # Uncertainty columns come after the bare metrics to preserve the historical layout
    # where ``trajectory_seed_count`` is followed by ``*_std`` / ``*_ci95`` blocks.
    seed_count_idx = None
    cols.append("trajectory_seed_count")
    seed_count_idx = len(cols)
    cols.extend(_uncertainty_columns())
    cols.extend(c for c in ("degenerate_flag", "timestamp", "error") if c not in cols)
    # Keep seed_count_idx referenced for potential callers introspecting layout
    _ = seed_count_idx
    return cols


SWEEP_RESULT_COLUMNS: list[str] = _build_result_columns()

# Mapping used by uncertainty aggregation to align raw metric keys with
# the column stem used in the CSV.
SEED_STAT_METRIC_MAP: list[tuple[str, str]] = [
    (field.source, field.column)
    for field in _METRIC_FIELDS
    if field.uncertainty and field.source
]


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _round_if_finite(value: object, digits: int | None) -> float:
    out = _safe_float(value)
    if not np.isfinite(out):
        return float("nan")
    if digits is None:
        return out
    return round(out, digits)


def build_result_row(
    nt: int,
    n_noise: int,
    hour0_train_time: float,
    autoregressive_train_time: float,
    metrics: dict,
    uncertainty: dict,
    trajectory_seed_count: int,
) -> dict:
    """Build a complete CSV row from raw evaluator outputs.

    All missing metrics render as NaN in the correct column; callers do not
    need to remember the schema.
    """
    row: dict[str, object] = {
        "nt": nt,
        "n_noise": n_noise,
        "hour0_train_time_sec": round(float(hour0_train_time), 1),
        "autoregressive_train_time_sec": round(float(autoregressive_train_time), 1),
        "total_train_time_sec": round(
            float(hour0_train_time + autoregressive_train_time), 1
        ),
    }

    for field in _METRIC_FIELDS:
        row[field.column] = _round_if_finite(
            metrics.get(field.source, float("nan")), field.precision
        )

    row["trajectory_seed_count"] = int(trajectory_seed_count)
    for field in _METRIC_FIELDS:
        if not field.uncertainty:
            continue
        row[f"{field.column}_std"] = _round_if_finite(
            uncertainty.get(f"{field.column}_std"), 6
        )
        row[f"{field.column}_ci95"] = _round_if_finite(
            uncertainty.get(f"{field.column}_ci95"), 6
        )

    row["degenerate_flag"] = int(metrics.get("degenerate_flag", 0))
    row["timestamp"] = datetime.now().isoformat()
    row["error"] = row.get("error", "")
    return row


def build_error_row(
    nt: int,
    n_noise: int,
    error: Exception,
    trajectory_seed_count: int,
) -> dict:
    """Result row for a failed sweep cell (all metrics NaN, error message set)."""
    row: dict[str, object] = {c: float("nan") for c in SWEEP_RESULT_COLUMNS}
    row["nt"] = nt
    row["n_noise"] = n_noise
    row["trajectory_seed_count"] = int(trajectory_seed_count)
    row["timestamp"] = datetime.now().isoformat()
    row["error"] = str(error)
    return row


def load_completed_runs(results_path: Path) -> set[tuple[int, int]]:
    """Load already-completed ``(nt, n_noise)`` pairs for resume support."""
    completed: set[tuple[int, int]] = set()
    if not results_path.exists():
        return completed
    with open(results_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            error_field = (r.get("error") or "").strip()
            if error_field:
                continue
            try:
                completed.add((int(r["nt"]), int(r["n_noise"])))
            except (KeyError, ValueError, TypeError):
                continue
    return completed


def append_result(results_path: Path, result: dict) -> None:
    """Append ``result`` to the CSV, creating it with a header if needed."""
    file_exists = results_path.exists()
    row = {k: result.get(k, float("nan")) for k in SWEEP_RESULT_COLUMNS}
    if pd.isna(row.get("timestamp")):
        row["timestamp"] = datetime.now().isoformat()
    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def ensure_results_schema(results_path: Path) -> None:
    """Migrate an existing CSV to the canonical column layout if needed."""
    if not results_path.exists():
        return
    try:
        df = pd.read_csv(results_path)
    except pd.errors.EmptyDataError:
        return
    except Exception as e:
        print(f"Warning: could not validate existing results schema: {e}")
        return

    changed = False
    for col in SWEEP_RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
            changed = True

    extra_cols = [c for c in df.columns if c not in SWEEP_RESULT_COLUMNS]
    if extra_cols:
        df = df.drop(columns=extra_cols)
        changed = True

    if list(df.columns) != SWEEP_RESULT_COLUMNS:
        df = df[SWEEP_RESULT_COLUMNS]
        changed = True

    if changed:
        df.to_csv(results_path, index=False)
        print(f"Updated existing results schema at {results_path}")


__all__ = [
    "SweepField",
    "SWEEP_RESULT_COLUMNS",
    "SEED_STAT_METRIC_MAP",
    "build_result_row",
    "build_error_row",
    "load_completed_runs",
    "append_result",
    "ensure_results_schema",
]
