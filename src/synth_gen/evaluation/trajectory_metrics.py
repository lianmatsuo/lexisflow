"""Trajectory-level quality metrics for synthetic time-series data.

Metrics compare temporal structure between real and synthetic patient
trajectories, complementing per-row quality metrics (KS, correlation,
clinical ranges).

All public functions expect DataFrames with ``subject_id`` and
``hours_in`` columns for trajectory grouping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# Core vital signs present in the autoregressive CSV (verified column names).
TRAJECTORY_VITALS: list[str] = [
    "Heart_Rate_mean",
    "Systolic_blood_pressure_mean",
    "Diastolic_blood_pressure_mean",
    "Mean_blood_pressure_mean",
    "Respiratory_rate_mean",
    "Oxygen_saturation_mean",
    "Temperature_mean",
]

# Clinically meaningful feature pairs for within-trajectory correlation.
CLINICAL_PAIRS: list[tuple[str, str]] = [
    ("Heart_Rate_mean", "Mean_blood_pressure_mean"),
    ("Systolic_blood_pressure_mean", "Diastolic_blood_pressure_mean"),
    ("Oxygen_saturation_mean", "Respiratory_rate_mean"),
]

_MIN_TRAJECTORY_LEN = 4  # Minimum rows for meaningful temporal statistics.
# Below this std (after finite mask), Pearson / lag-1 correlation is undefined;
# skip np.corrcoef to avoid divide-by-zero RuntimeWarnings.
_MIN_STD_FOR_CORR = 1e-12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pearson_corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r for paired samples; NaN if inputs are constant or too short."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2 or x.shape != y.shape:
        return np.nan
    if np.std(x) < _MIN_STD_FOR_CORR or np.std(y) < _MIN_STD_FOR_CORR:
        return np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else np.nan


def _valid_vitals(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    candidates: list[str] | None = None,
) -> list[str]:
    """Return the subset of *candidates* present in both DataFrames."""
    if candidates is None:
        candidates = TRAJECTORY_VITALS
    return [c for c in candidates if c in real_df.columns and c in synth_df.columns]


def _per_trajectory_lag1_autocorr(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, list[float]]:
    """Compute lag-1 autocorrelation per trajectory per feature.

    Returns a dict mapping feature name → list of per-trajectory
    autocorrelation values (one float per qualifying trajectory).
    """
    result: dict[str, list[float]] = {f: [] for f in feature_cols}
    for _, grp in df.groupby("subject_id"):
        grp = grp.sort_values("hours_in")
        if len(grp) < _MIN_TRAJECTORY_LEN:
            continue
        for f in feature_cols:
            vals = grp[f].dropna().values
            if len(vals) < _MIN_TRAJECTORY_LEN:
                continue
            acf1 = _pearson_corrcoef_safe(vals[:-1], vals[1:])
            if np.isfinite(acf1):
                result[f].append(acf1)
    return result


def _within_trajectory_corr(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], list[float]]:
    """Compute per-trajectory Pearson correlation for each feature pair."""
    result: dict[tuple[str, str], list[float]] = {p: [] for p in pairs}
    for _, grp in df.groupby("subject_id"):
        grp = grp.sort_values("hours_in")
        if len(grp) < _MIN_TRAJECTORY_LEN:
            continue
        for pair in pairs:
            f1, f2 = pair
            v1 = grp[f1].values.astype(float)
            v2 = grp[f2].values.astype(float)
            mask = np.isfinite(v1) & np.isfinite(v2)
            if mask.sum() < _MIN_TRAJECTORY_LEN:
                continue
            r = _pearson_corrcoef_safe(v1[mask], v2[mask])
            if np.isfinite(r):
                result[pair].append(r)
    return result


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


def compute_autocorrelation_distance(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> float:
    """Mean absolute difference in median lag-1 autocorrelation per vital sign.

    Compares whether synthetic trajectories preserve the same hour-to-hour
    temporal dependency as real ones.  A value of 0 means perfect match;
    values above ~0.3 indicate substantial temporal structure loss.

    Args:
        real_df: Real data with ``subject_id``, ``hours_in``, and vital columns.
        synth_df: Synthetic data with the same schema.

    Returns:
        Scalar distance in [0, 2], or ``NaN`` if no features qualify.
    """
    cols = _valid_vitals(real_df, synth_df)
    if not cols:
        return np.nan

    real_acf = _per_trajectory_lag1_autocorr(real_df, cols)
    synth_acf = _per_trajectory_lag1_autocorr(synth_df, cols)

    diffs: list[float] = []
    for f in cols:
        if real_acf[f] and synth_acf[f]:
            diffs.append(
                abs(float(np.median(real_acf[f])) - float(np.median(synth_acf[f])))
            )
    return float(np.mean(diffs)) if diffs else np.nan


def compute_stay_length_ks(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> float:
    """Two-sample KS statistic on trajectory lengths (rows per subject_id).

    Measures whether the distribution of ICU stay lengths in synthetic data
    matches the real data.  Because the current generator produces
    fixed-length 48-hour trajectories while real stays vary, this metric
    quantifies a known structural limitation.

    Args:
        real_df: Real data with ``subject_id`` column.
        synth_df: Synthetic data with ``subject_id`` column.

    Returns:
        KS statistic in [0, 1], or ``NaN`` if insufficient data.
    """
    if "subject_id" not in real_df.columns or "subject_id" not in synth_df.columns:
        return np.nan

    real_lens = real_df.groupby("subject_id").size().values
    synth_lens = synth_df.groupby("subject_id").size().values

    if len(real_lens) < 2 or len(synth_lens) < 2:
        return np.nan

    ks_stat, _ = stats.ks_2samp(real_lens, synth_lens)
    return float(ks_stat)


def compute_transition_smoothness(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> float:
    """Ratio of mean hour-to-hour absolute change (synthetic / real).

    A ratio of 1.0 means transitions are equally smooth.  Values > 1
    indicate the synthetic data has larger jumps (rougher trajectories);
    values < 1 indicate over-smoothed trajectories.

    Args:
        real_df: Real data with grouping and vital columns.
        synth_df: Synthetic data with the same schema.

    Returns:
        Mean ratio across vital signs, or ``NaN`` if no features qualify.
    """
    cols = _valid_vitals(real_df, synth_df)
    if not cols:
        return np.nan

    ratios: list[float] = []
    for f in cols:
        real_deltas: list[float] = []
        for _, grp in real_df.groupby("subject_id"):
            grp = grp.sort_values("hours_in")
            vals = grp[f].dropna().values
            if len(vals) >= 2:
                real_deltas.extend(np.abs(np.diff(vals)).tolist())

        synth_deltas: list[float] = []
        for _, grp in synth_df.groupby("subject_id"):
            grp = grp.sort_values("hours_in")
            vals = grp[f].dropna().values
            if len(vals) >= 2:
                synth_deltas.extend(np.abs(np.diff(vals)).tolist())

        if real_deltas and synth_deltas:
            real_mean = float(np.mean(real_deltas))
            if real_mean > 1e-9:
                ratios.append(float(np.mean(synth_deltas)) / real_mean)

    return float(np.mean(ratios)) if ratios else np.nan


def compute_temporal_corr_drift(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> float:
    """Mean absolute difference in median within-trajectory cross-feature correlation.

    Checks whether clinically meaningful feature relationships (e.g.,
    heart-rate vs. MAP) are preserved *within* individual trajectories,
    not just in aggregate.

    Args:
        real_df: Real data with grouping and vital columns.
        synth_df: Synthetic data with the same schema.

    Returns:
        Scalar distance in [0, 2], or ``NaN`` if no pairs qualify.
    """
    valid_pairs = [
        (a, b)
        for a, b in CLINICAL_PAIRS
        if a in real_df.columns
        and b in real_df.columns
        and a in synth_df.columns
        and b in synth_df.columns
    ]
    if not valid_pairs:
        return np.nan

    real_corrs = _within_trajectory_corr(real_df, valid_pairs)
    synth_corrs = _within_trajectory_corr(synth_df, valid_pairs)

    diffs: list[float] = []
    for pair in valid_pairs:
        if real_corrs[pair] and synth_corrs[pair]:
            diffs.append(
                abs(
                    float(np.median(real_corrs[pair]))
                    - float(np.median(synth_corrs[pair]))
                )
            )
    return float(np.mean(diffs)) if diffs else np.nan


# ---------------------------------------------------------------------------
# Aggregate wrapper
# ---------------------------------------------------------------------------


def compute_trajectory_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute all trajectory-level quality metrics.

    Args:
        real_df: Real data with ``subject_id``, ``hours_in``, and feature columns.
        synth_df: Synthetic data with the same schema.

    Returns:
        Dictionary with keys ``autocorr_distance``, ``stay_length_ks``,
        ``transition_mse_ratio``, ``temporal_corr_drift``.
    """
    return {
        "autocorr_distance": compute_autocorrelation_distance(real_df, synth_df),
        "stay_length_ks": compute_stay_length_ks(real_df, synth_df),
        "transition_mse_ratio": compute_transition_smoothness(real_df, synth_df),
        "temporal_corr_drift": compute_temporal_corr_drift(real_df, synth_df),
    }
