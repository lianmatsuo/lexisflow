"""Evaluation tools for synthetic data quality assessment.

This package provides:
- Quality metrics (KS statistics, correlation, clinical ranges)
- Trajectory-level metrics (autocorrelation, stay length, transitions)
- TSTR (Train on Synthetic, Test on Real) multi-task framework
- Privacy metrics (DCR + membership inference)
"""

from .quality_metrics import (
    compute_quality_metrics,
    compute_ks_statistics,
    compute_correlation_frobenius,
    compute_clinical_range_violations,
    CLINICAL_RANGES,
)
from .tstr_framework import (
    TSTRTask,
    MortalityTask,
    VasopressorTask,
    LOSTask,
    evaluate_tstr_multi_task,
)
from .trajectory_metrics import (
    compute_trajectory_metrics,
    compute_autocorrelation_distance,
    compute_stay_length_ks,
    compute_transition_smoothness,
    compute_temporal_corr_drift,
)

__all__ = [
    # Quality metrics
    "compute_quality_metrics",
    "compute_ks_statistics",
    "compute_correlation_frobenius",
    "compute_clinical_range_violations",
    "CLINICAL_RANGES",
    # Privacy metrics (lazy-loaded below)
    "compute_distance_to_closest_record",
    "compute_dcr_baseline_protection",
    "compute_dcr_overfitting_protection",
    "compute_domias_like_membership_inference",
    "compute_privacy_metrics",
    # TSTR framework
    "TSTRTask",
    "MortalityTask",
    "VasopressorTask",
    "LOSTask",
    "evaluate_tstr_multi_task",
    # Trajectory metrics
    "compute_trajectory_metrics",
    "compute_autocorrelation_distance",
    "compute_stay_length_ks",
    "compute_transition_smoothness",
    "compute_temporal_corr_drift",
]

_PRIVACY_EXPORTS = {
    "compute_distance_to_closest_record",
    "compute_dcr_baseline_protection",
    "compute_dcr_overfitting_protection",
    "compute_domias_like_membership_inference",
    "compute_privacy_metrics",
}


def __getattr__(name: str):
    """Lazy-load privacy metrics to avoid eager module side-effects."""
    if name in _PRIVACY_EXPORTS:
        from . import privacy_metrics as _privacy_metrics

        return getattr(_privacy_metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
