"""Quality metrics for evaluating synthetic data.

Metrics:
- Kolmogorov-Smirnov tests for marginal distributions
- Correlation matrix preservation (Frobenius norm)
- Clinical range violations

Column-name scheme matches the flattened MIMIC-Extract columns produced by
``lexisflow.data.feature_utils.flatten_column_names`` (e.g.
``Heart_Rate_mean``, ``Systolic_blood_pressure_mean``).
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy import stats


RangeSpec = tuple[tuple[str, ...], tuple[float, float]]

# Clinically plausible ranges keyed by a semantic feature name and a preferred
# alias order. MIMIC flattened names stay first so existing behavior is
# preserved; Challenge 2012 aliases provide a fallback.
CLINICAL_RANGE_SPECS: dict[str, RangeSpec] = {
    "heart_rate": (("Heart_Rate_mean", "HR"), (0, 300)),
    "systolic_bp": (("Systolic_blood_pressure_mean", "SysABP", "NISysABP"), (50, 250)),
    "diastolic_bp": (
        ("Diastolic_blood_pressure_mean", "DiasABP", "NIDiasABP"),
        (20, 180),
    ),
    "mean_bp": (("Mean_blood_pressure_mean", "MAP", "NIMAP"), (30, 200)),
    "resp_rate": (("Respiratory_rate_mean", "RespRate"), (0, 80)),
    "temperature": (("Temperature_mean", "Temp"), (25, 45)),
    "oxygen_sat": (("Oxygen_saturation_mean", "SaO2"), (0, 100)),
    "glucose": (("Glucose_mean", "Glucose"), (0, 1500)),
    "bicarbonate": (("Bicarbonate_mean", "HCO3"), (0, 60)),
    "creatinine": (("Creatinine_mean", "Creatinine"), (0, 25)),
    "chloride": (("Chloride_mean", "Chloride"), (50, 150)),
    "hematocrit": (("Hematocrit_mean", "HCT"), (10, 70)),
    "hemoglobin": (("Hemoglobin_mean", "Hgb"), (3, 25)),
    "platelets": (("Platelets_mean", "Platelets"), (0, 2000)),
    "potassium": (("Potassium_mean", "K"), (1.5, 10)),
    "sodium": (("Sodium_mean", "Na"), (100, 180)),
    "blood_urea_nitrogen": (("Blood_urea_nitrogen_mean", "BUN"), (0, 200)),
    "white_blood_cell_count": (("White_blood_cell_count_mean", "WBC"), (0, 100)),
}

# Backwards-compatible export used by tests/callers that inspect this constant.
CLINICAL_RANGES: dict[str, tuple[float, float]] = {
    aliases[0]: bounds for aliases, bounds in CLINICAL_RANGE_SPECS.values()
}


def _resolve_first_present_column(
    columns: pd.Index, aliases: tuple[str, ...]
) -> str | None:
    """Return the first alias present in ``columns``."""
    for col in aliases:
        if col in columns:
            return col
    return None


def _valid_common_numeric_columns(
    real_df: pd.DataFrame, synth_df: pd.DataFrame
) -> list[str]:
    """Return common numeric columns with ≥2 finite values and non-zero variance in both frames."""
    real_numeric = real_df.select_dtypes(include=[np.number]).columns
    synth_numeric = synth_df.select_dtypes(include=[np.number]).columns
    common_cols = sorted(set(real_numeric) & set(synth_numeric))

    valid: list[str] = []
    for col in common_cols:
        real_vals = real_df[col].to_numpy(dtype=float)
        synth_vals = synth_df[col].to_numpy(dtype=float)
        real_finite = real_vals[np.isfinite(real_vals)]
        synth_finite = synth_vals[np.isfinite(synth_vals)]
        if len(real_finite) < 2 or len(synth_finite) < 2:
            continue
        # Correlation is undefined for zero-variance columns
        if np.nanstd(real_finite) == 0 or np.nanstd(synth_finite) == 0:
            continue
        valid.append(col)
    return valid


def compute_ks_statistics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> tuple[list[float], list[str]]:
    """Compute KS statistics for all valid common numeric columns.

    Returns:
        Tuple of (ks_statistics, column_names).
    """
    cols = _valid_common_numeric_columns(real_df, synth_df)
    ks_stats: list[float] = []
    ks_cols: list[str] = []
    for col in cols:
        real_vals = real_df[col].dropna().to_numpy()
        synth_vals = synth_df[col].dropna().to_numpy()
        if len(real_vals) and len(synth_vals):
            stat, _ = stats.ks_2samp(real_vals, synth_vals)
            ks_stats.append(float(stat))
            ks_cols.append(col)
    return ks_stats, ks_cols


def compute_correlation_frobenius(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> float:
    """Frobenius norm of ``synth_corr - real_corr`` over shared valid columns."""
    cols = _valid_common_numeric_columns(real_df, synth_df)
    if len(cols) < 2:
        return float("nan")
    try:
        real_corr = real_df[cols].corr()
        synth_corr = synth_df[cols].corr()

        if np.isnan(real_corr.values).any() or np.isnan(synth_corr.values).any():
            valid_mask = ~(
                np.isnan(real_corr.values).any(axis=0)
                | np.isnan(synth_corr.values).any(axis=0)
            )
            valid_cols = real_corr.columns[valid_mask]
            if len(valid_cols) < 2:
                return float("nan")
            real_corr = real_corr.loc[valid_cols, valid_cols]
            synth_corr = synth_corr.loc[valid_cols, valid_cols]

        diff = synth_corr.values - real_corr.values
        return float(np.linalg.norm(diff, ord="fro"))
    except Exception:
        return float("nan")


def compute_clinical_range_violations(
    synth_df: pd.DataFrame,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Percentage of synthetic values outside clinically plausible ranges.

    Returns:
        Percentage in [0, 100], or NaN when no ranged columns are present.
    """
    total_violations = 0
    total_values = 0
    features_checked = 0

    if ranges is not None:
        range_specs: list[RangeSpec] = [
            ((col,), bounds) for col, bounds in ranges.items()
        ]
    else:
        range_specs = list(CLINICAL_RANGE_SPECS.values())

    for aliases, (vmin, vmax) in range_specs:
        col = _resolve_first_present_column(synth_df.columns, aliases)
        if col is None:
            continue
        features_checked += 1
        vals = synth_df[col].dropna()
        if not len(vals):
            continue
        total_violations += int(((vals < vmin) | (vals > vmax)).sum())
        total_values += len(vals)

    if features_checked == 0 or total_values == 0:
        return float("nan")
    return 100.0 * total_violations / total_values


def compute_quality_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
) -> dict:
    """Compute all quality metrics.

    Returns a dict with keys ``avg_ks_stat``, ``corr_frobenius``, ``range_violation_pct``.
    """
    ks_stats, _ = compute_ks_statistics(real_df, synth_df)
    avg_ks = float(np.mean(ks_stats)) if ks_stats else float("nan")
    return {
        "avg_ks_stat": avg_ks,
        "corr_frobenius": compute_correlation_frobenius(real_df, synth_df),
        "range_violation_pct": compute_clinical_range_violations(synth_df),
    }


__all__ = [
    "CLINICAL_RANGES",
    "compute_ks_statistics",
    "compute_correlation_frobenius",
    "compute_clinical_range_violations",
    "compute_quality_metrics",
]
