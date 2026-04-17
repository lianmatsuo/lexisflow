"""Unit tests for trajectory-level quality metrics."""

import warnings

import numpy as np
import pandas as pd

from packages.evaluation.trajectory_metrics import (
    compute_autocorrelation_distance,
    compute_stay_length_ks,
    compute_transition_smoothness,
    compute_temporal_corr_drift,
    compute_trajectory_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_smooth_trajectories(
    n_patients: int = 20,
    hours: int = 48,
    seed: int = 42,
) -> pd.DataFrame:
    """Create trajectories with smooth, correlated vital signs."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        hr_base = rng.uniform(60, 100)
        map_base = rng.uniform(70, 110)
        for h in range(hours):
            noise = rng.normal(0, 1)
            hr = hr_base + noise
            # MAP correlated with HR
            map_val = map_base + 0.5 * noise + rng.normal(0, 0.5)
            rows.append(
                {
                    "subject_id": pid,
                    "hours_in": h,
                    "Heart_Rate_mean": hr,
                    "Mean_blood_pressure_mean": map_val,
                    "Systolic_blood_pressure_mean": map_val + 30 + rng.normal(0, 2),
                    "Diastolic_blood_pressure_mean": map_val - 20 + rng.normal(0, 2),
                    "Respiratory_rate_mean": 16 + rng.normal(0, 1),
                    "Oxygen_saturation_mean": 97 + rng.normal(0, 0.5),
                    "Temperature_mean": 37.0 + rng.normal(0, 0.1),
                }
            )
            hr_base = hr  # Smooth: next step depends on current
            map_base = map_val
    return pd.DataFrame(rows)


def _make_random_trajectories(
    n_patients: int = 20,
    hours: int = 48,
    seed: int = 99,
) -> pd.DataFrame:
    """Create trajectories with IID random values (no temporal structure)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        for h in range(hours):
            rows.append(
                {
                    "subject_id": pid,
                    "hours_in": h,
                    "Heart_Rate_mean": rng.uniform(40, 150),
                    "Mean_blood_pressure_mean": rng.uniform(50, 150),
                    "Systolic_blood_pressure_mean": rng.uniform(70, 200),
                    "Diastolic_blood_pressure_mean": rng.uniform(40, 120),
                    "Respiratory_rate_mean": rng.uniform(8, 40),
                    "Oxygen_saturation_mean": rng.uniform(85, 100),
                    "Temperature_mean": rng.uniform(35, 40),
                }
            )
    return pd.DataFrame(rows)


def _make_variable_length_trajectories(
    n_patients: int = 30,
    seed: int = 77,
) -> pd.DataFrame:
    """Create trajectories with variable lengths (5-48 hours)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_patients):
        length = rng.integers(5, 49)
        hr_base = rng.uniform(60, 100)
        for h in range(length):
            hr_base += rng.normal(0, 1)
            rows.append(
                {
                    "subject_id": pid,
                    "hours_in": h,
                    "Heart_Rate_mean": hr_base,
                    "Mean_blood_pressure_mean": 80 + rng.normal(0, 5),
                    "Systolic_blood_pressure_mean": 120 + rng.normal(0, 10),
                    "Diastolic_blood_pressure_mean": 70 + rng.normal(0, 5),
                    "Respiratory_rate_mean": 16 + rng.normal(0, 2),
                    "Oxygen_saturation_mean": 97 + rng.normal(0, 1),
                    "Temperature_mean": 37.0 + rng.normal(0, 0.2),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Autocorrelation Distance
# ---------------------------------------------------------------------------


class TestAutocorrelationDistance:
    """Test lag-1 autocorrelation distance metric."""

    def test_identical_data_near_zero(self):
        """Identical real and synth data should produce near-zero distance."""
        df = _make_smooth_trajectories(seed=42)
        dist = compute_autocorrelation_distance(df, df.copy())
        assert np.isfinite(dist)
        assert dist < 0.01

    def test_random_vs_smooth_larger(self):
        """Random trajectories vs smooth ones should have larger distance."""
        smooth = _make_smooth_trajectories(seed=42)
        random_df = _make_random_trajectories(seed=99)
        dist = compute_autocorrelation_distance(smooth, random_df)
        assert np.isfinite(dist)
        assert dist > 0.1

    def test_missing_columns_returns_nan(self):
        """If no vital columns are shared, return NaN."""
        df1 = pd.DataFrame({"subject_id": [0, 0], "hours_in": [0, 1], "foo": [1, 2]})
        df2 = pd.DataFrame({"subject_id": [0, 0], "hours_in": [0, 1], "bar": [1, 2]})
        assert np.isnan(compute_autocorrelation_distance(df1, df2))

    def test_short_trajectories_skipped(self):
        """Trajectories with < 4 rows should be skipped gracefully."""
        df = pd.DataFrame(
            {
                "subject_id": [0, 0, 0],
                "hours_in": [0, 1, 2],
                "Heart_Rate_mean": [80, 82, 81],
            }
        )
        # Only 3 rows — below _MIN_TRAJECTORY_LEN, should return NaN
        result = compute_autocorrelation_distance(df, df.copy())
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# Stay-Length KS
# ---------------------------------------------------------------------------


class TestStayLengthKS:
    """Test stay-length distribution KS metric."""

    def test_identical_lengths_near_zero(self):
        """Same-length trajectories should produce KS ≈ 0."""
        df = _make_smooth_trajectories(n_patients=20, hours=48)
        ks = compute_stay_length_ks(df, df.copy())
        assert np.isfinite(ks)
        assert ks < 0.01

    def test_variable_vs_fixed_length_high(self):
        """Variable-length real vs fixed-length synth should produce high KS."""
        real = _make_variable_length_trajectories(n_patients=30)
        synth = _make_smooth_trajectories(n_patients=30, hours=48)
        ks = compute_stay_length_ks(real, synth)
        assert np.isfinite(ks)
        assert ks > 0.3

    def test_missing_subject_id_returns_nan(self):
        """Missing subject_id column should return NaN."""
        df = pd.DataFrame({"hours_in": [0, 1], "val": [1, 2]})
        assert np.isnan(compute_stay_length_ks(df, df.copy()))

    def test_single_patient_returns_nan(self):
        """Only one patient in either dataset should return NaN."""
        df1 = pd.DataFrame({"subject_id": [0, 0], "hours_in": [0, 1]})
        df2 = pd.DataFrame({"subject_id": [0, 0, 1, 1], "hours_in": [0, 1, 0, 1]})
        assert np.isnan(compute_stay_length_ks(df1, df2))


# ---------------------------------------------------------------------------
# Transition Smoothness
# ---------------------------------------------------------------------------


class TestTransitionSmoothness:
    """Test hour-to-hour transition smoothness ratio."""

    def test_identical_data_ratio_one(self):
        """Identical data should produce ratio ≈ 1.0."""
        df = _make_smooth_trajectories(seed=42)
        ratio = compute_transition_smoothness(df, df.copy())
        assert np.isfinite(ratio)
        assert abs(ratio - 1.0) < 0.01

    def test_rougher_synth_ratio_above_one(self):
        """Random (rougher) synth vs smooth real should produce ratio > 1."""
        smooth = _make_smooth_trajectories(seed=42)
        random_df = _make_random_trajectories(seed=99)
        ratio = compute_transition_smoothness(smooth, random_df)
        assert np.isfinite(ratio)
        assert ratio > 1.5

    def test_no_shared_columns_returns_nan(self):
        """No shared vital columns should return NaN."""
        df1 = pd.DataFrame({"subject_id": [0, 0], "hours_in": [0, 1], "foo": [1, 2]})
        df2 = pd.DataFrame({"subject_id": [0, 0], "hours_in": [0, 1], "bar": [1, 2]})
        assert np.isnan(compute_transition_smoothness(df1, df2))


# ---------------------------------------------------------------------------
# Temporal Cross-Feature Correlation Drift
# ---------------------------------------------------------------------------


class TestTemporalCorrDrift:
    """Test within-trajectory cross-feature correlation drift."""

    def test_identical_data_near_zero(self):
        """Identical data should produce drift ≈ 0."""
        df = _make_smooth_trajectories(seed=42)
        drift = compute_temporal_corr_drift(df, df.copy())
        assert np.isfinite(drift)
        assert drift < 0.01

    def test_uncorrelated_synth_larger_drift(self):
        """Random (uncorrelated) synth should show larger drift."""
        smooth = _make_smooth_trajectories(seed=42)
        random_df = _make_random_trajectories(seed=99)
        drift = compute_temporal_corr_drift(smooth, random_df)
        assert np.isfinite(drift)
        assert drift > 0.1

    def test_no_valid_pairs_returns_nan(self):
        """If no clinical pairs have both columns present, return NaN."""
        df = pd.DataFrame(
            {
                "subject_id": [0, 0, 0, 0, 0],
                "hours_in": [0, 1, 2, 3, 4],
                "foo": [1, 2, 3, 4, 5],
            }
        )
        assert np.isnan(compute_temporal_corr_drift(df, df.copy()))


def test_no_runtime_warning_when_vitals_constant_within_trajectory() -> None:
    """Constant vitals give undefined corrcoef; helpers must not emit numpy warnings."""
    rows = []
    for pid in range(3):
        for h in range(8):
            rows.append(
                {
                    "subject_id": pid,
                    "hours_in": h,
                    "Heart_Rate_mean": 72.0,
                    "Mean_blood_pressure_mean": 80.0,
                    "Systolic_blood_pressure_mean": 120.0,
                    "Diastolic_blood_pressure_mean": 60.0,
                    "Respiratory_rate_mean": 16.0,
                    "Oxygen_saturation_mean": 98.0,
                    "Temperature_mean": 37.0,
                }
            )
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        compute_trajectory_metrics(df, df.copy())


# ---------------------------------------------------------------------------
# Aggregate wrapper
# ---------------------------------------------------------------------------


class TestComputeTrajectoryMetrics:
    """Test the aggregate compute_trajectory_metrics wrapper."""

    def test_returns_all_keys(self):
        """Wrapper should return all 4 metric keys."""
        df = _make_smooth_trajectories(n_patients=10, hours=10)
        result = compute_trajectory_metrics(df, df.copy())
        expected_keys = {
            "autocorr_distance",
            "stay_length_ks",
            "transition_mse_ratio",
            "temporal_corr_drift",
        }
        assert set(result.keys()) == expected_keys

    def test_identical_data_values(self):
        """All metrics should be near-zero or near-1.0 for identical data."""
        df = _make_smooth_trajectories(n_patients=15, hours=20)
        result = compute_trajectory_metrics(df, df.copy())
        assert result["autocorr_distance"] < 0.01
        assert result["stay_length_ks"] < 0.01
        assert abs(result["transition_mse_ratio"] - 1.0) < 0.01
        assert result["temporal_corr_drift"] < 0.01
