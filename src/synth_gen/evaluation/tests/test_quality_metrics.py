"""Unit tests for quality metrics."""

import numpy as np
import pandas as pd
import pytest

from synth_gen.evaluation.quality_metrics import (
    compute_ks_statistics,
    compute_correlation_frobenius,
    compute_clinical_range_violations,
    compute_quality_metrics,
)


class TestKSStatistics:
    """Test Kolmogorov-Smirnov statistics computation."""

    def test_identical_distributions(self):
        """Test KS statistic for identical distributions."""
        df = pd.DataFrame(
            {
                "age": [25, 35, 45, 55, 65],
                "weight": [60, 70, 80, 90, 100],
            }
        )

        ks_stats, ks_cols = compute_ks_statistics(df, df)

        # Identical distributions should have KS ≈ 0
        assert len(ks_stats) == 2
        assert all(ks < 0.01 for ks in ks_stats)

    def test_different_distributions(self):
        """Test KS statistic for different distributions."""
        real_df = pd.DataFrame(
            {
                "age": np.random.normal(50, 10, 100),
            }
        )

        synth_df = pd.DataFrame(
            {
                "age": np.random.normal(30, 5, 100),  # Different mean/std
            }
        )

        ks_stats, ks_cols = compute_ks_statistics(real_df, synth_df)

        # Different distributions should have higher KS
        assert len(ks_stats) == 1
        assert ks_stats[0] > 0.1  # Should be noticeably different

    def test_mixed_types_only_numeric(self):
        """Test that only numeric columns are compared."""
        real_df = pd.DataFrame(
            {
                "age": [25, 35, 45],
                "gender": ["M", "F", "M"],  # Should be ignored
            }
        )

        synth_df = pd.DataFrame(
            {
                "age": [30, 40, 50],
                "gender": ["F", "M", "F"],
            }
        )

        ks_stats, ks_cols = compute_ks_statistics(real_df, synth_df)

        # Only age should be compared
        assert len(ks_stats) == 1
        assert ks_cols[0] == "age"

    def test_missing_columns_handled(self):
        """Test that missing columns are skipped."""
        real_df = pd.DataFrame(
            {
                "age": [25, 35, 45],
                "weight": [60, 70, 80],
            }
        )

        synth_df = pd.DataFrame(
            {
                "age": [30, 40, 50],
                "height": [170, 180, 190],  # Different column
            }
        )

        ks_stats, ks_cols = compute_ks_statistics(real_df, synth_df)

        # Only 'age' is common
        assert len(ks_stats) == 1
        assert ks_cols[0] == "age"


class TestCorrelationFrobenius:
    """Test correlation preservation metric."""

    def test_identical_correlations(self):
        """Test Frobenius norm for identical correlations."""
        df = pd.DataFrame(
            {
                "a": np.arange(100),
                "b": np.arange(100) * 2,  # Perfect correlation
                "c": np.arange(100) * -1,  # Perfect negative correlation
            }
        )

        frob = compute_correlation_frobenius(df, df)

        # Identical correlations → Frobenius ≈ 0
        assert frob < 0.01

    def test_different_correlations(self):
        """Test Frobenius norm for different correlations."""
        real_df = pd.DataFrame(
            {
                "a": np.arange(100),
                "b": np.arange(100) * 2,  # Correlated
            }
        )

        synth_df = pd.DataFrame(
            {
                "a": np.random.randn(100),
                "b": np.random.randn(100),  # Uncorrelated
            }
        )

        frob = compute_correlation_frobenius(real_df, synth_df)

        # Different correlations → higher Frobenius
        assert frob > 0.5

    def test_single_column_returns_nan(self):
        """Test that single column returns NaN (can't compute correlation)."""
        real_df = pd.DataFrame({"age": [25, 35, 45]})
        synth_df = pd.DataFrame({"age": [30, 40, 50]})

        frob = compute_correlation_frobenius(real_df, synth_df)

        assert np.isnan(frob)

    def test_no_common_columns_returns_nan(self):
        """Test that no common columns returns NaN."""
        real_df = pd.DataFrame({"age": [25, 35, 45]})
        synth_df = pd.DataFrame({"weight": [60, 70, 80]})

        frob = compute_correlation_frobenius(real_df, synth_df)

        assert np.isnan(frob)


class TestClinicalRangeViolations:
    """Test clinical range violation detection."""

    def test_no_violations(self):
        """Test data with no violations."""
        synth_df = pd.DataFrame(
            {
                "Heart_Rate_mean": [70, 80, 90, 100, 110],
                "Temperature_mean": [36.5, 37.0, 37.5, 38.0, 38.5],
            }
        )

        pct = compute_clinical_range_violations(synth_df)

        assert pct == 0.0

    def test_with_violations(self):
        """Test data with violations using canonical CLINICAL_RANGES."""
        # Canonical ranges: Heart_Rate_mean=(0, 300), Temperature_mean=(25, 45)
        synth_df = pd.DataFrame(
            {
                "Heart_Rate_mean": [10, 80, 200, 400, 110],  # 400 violates -> 1/5
                "Temperature_mean": [36.5, 37.0, 50.0, 38.0, 20.0],  # 50, 20 -> 2/5
            }
        )

        pct = compute_clinical_range_violations(synth_df)

        # 3 violations out of 10 values = 30%
        assert pct == 30.0

    def test_custom_ranges(self):
        """Test with custom range dictionary."""
        synth_df = pd.DataFrame(
            {
                "custom_feature": [5, 10, 15, 20, 25],
            }
        )

        custom_ranges = {
            "custom_feature": (10, 20),  # Valid range [10, 20]
        }

        pct = compute_clinical_range_violations(synth_df, ranges=custom_ranges)

        # 5 and 25 are out of range (2/5 = 40%)
        assert pct == 40.0

    def test_no_matching_features_returns_nan(self):
        """Test that no matching features returns NaN."""
        synth_df = pd.DataFrame(
            {
                "unrelated_feature": [1, 2, 3, 4, 5],
            }
        )

        pct = compute_clinical_range_violations(synth_df)

        assert np.isnan(pct)


class TestQualityMetricsIntegration:
    """Test integrated quality metrics computation."""

    def test_compute_all_metrics(self):
        """Test that compute_quality_metrics returns all metrics."""
        real_df = pd.DataFrame(
            {
                "age": np.random.normal(50, 15, 100),
                "weight": np.random.normal(75, 20, 100),
                "Heart_Rate_mean": np.random.normal(80, 15, 100),
            }
        )

        synth_df = pd.DataFrame(
            {
                "age": np.random.normal(52, 14, 100),
                "weight": np.random.normal(76, 19, 100),
                "Heart_Rate_mean": np.random.normal(82, 14, 100),
            }
        )

        metrics = compute_quality_metrics(real_df, synth_df)

        # Check all keys present
        assert "avg_ks_stat" in metrics
        assert "corr_frobenius" in metrics
        assert "range_violation_pct" in metrics

        # Check reasonable values
        assert 0 <= metrics["avg_ks_stat"] <= 1
        assert metrics["corr_frobenius"] >= 0
        assert 0 <= metrics["range_violation_pct"] <= 100

    def test_with_missing_values(self):
        """Test metrics with missing values in data."""
        real_df = pd.DataFrame(
            {
                "age": [25, np.nan, 45, 55, np.nan],
                "weight": [60, 70, np.nan, 90, 100],
            }
        )

        synth_df = pd.DataFrame(
            {
                "age": [30, 40, np.nan, 50, 60],
                "weight": [np.nan, 75, 85, 95, 105],
            }
        )

        metrics = compute_quality_metrics(real_df, synth_df)

        # Should not crash with NaN
        assert "avg_ks_stat" in metrics
        assert not np.isnan(metrics["avg_ks_stat"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
