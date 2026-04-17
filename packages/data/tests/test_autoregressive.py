"""Unit tests for autoregressive data preparation."""

import numpy as np
import pandas as pd
import pytest

from packages.data.autoregressive import (
    prepare_autoregressive_data,
    split_static_dynamic,
)


class TestPrepareAutogressiveData:
    """Test prepare_autoregressive_data function."""

    def test_basic_lag_creation(self):
        """Test basic lag feature creation."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 1, 2, 2, 2],
                "hours_in": [0, 1, 2, 0, 1, 2],
                "hr": [80, 82, 79, 85, 87, 84],
                "bp": [120, 118, 115, 125, 123, 120],
                "age": [65, 65, 65, 45, 45, 45],
            }
        )

        df_ar, target_cols, condition_cols = prepare_autoregressive_data(
            df,
            id_col="subject_id",
            time_col="hours_in",
            static_cols=["age"],
            lag=1,
        )

        # Check columns created
        assert "hr_lag1" in df_ar.columns
        assert "bp_lag1" in df_ar.columns
        assert "age" in df_ar.columns

        # Check target_cols
        assert set(target_cols) == {"hr", "bp"}

        # Check condition_cols
        assert "age" in condition_cols
        assert "hr_lag1" in condition_cols
        assert "bp_lag1" in condition_cols

        # Check lag values
        # Patient 1, hour 1: lag should be hour 0 values
        patient1_hour1 = df_ar[(df_ar["subject_id"] == 1) & (df_ar["hours_in"] == 1)]
        assert patient1_hour1["hr_lag1"].values[0] == 80
        assert patient1_hour1["bp_lag1"].values[0] == 120

    def test_first_timestep_fill(self):
        """Test that first timestep is filled correctly."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "hours_in": [0, 1, 2],
                "hr": [80, 82, 79],
                "age": [65, 65, 65],
            }
        )

        # Test special fill
        df_ar, _, _ = prepare_autoregressive_data(
            df,
            "subject_id",
            "hours_in",
            ["age"],
            lag=1,
            fill_strategy="special",
            fill_value=-1.0,
        )

        # Hour 0 should have lag = -1
        hour0 = df_ar[df_ar["hours_in"] == 0]
        assert hour0["hr_lag1"].values[0] == -1.0

        # Test zero fill
        df_ar_zero, _, _ = prepare_autoregressive_data(
            df, "subject_id", "hours_in", ["age"], lag=1, fill_strategy="zero"
        )

        hour0_zero = df_ar_zero[df_ar_zero["hours_in"] == 0]
        assert hour0_zero["hr_lag1"].values[0] == 0.0

    def test_multiple_patients(self):
        """Test lag features are created within patient groups."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 2, 3, 3],
                "hours_in": [0, 1, 0, 1, 0, 1],
                "hr": [80, 82, 90, 92, 70, 72],
                "age": [65, 65, 55, 55, 75, 75],
            }
        )

        df_ar, _, _ = prepare_autoregressive_data(
            df, "subject_id", "hours_in", ["age"], lag=1
        )

        # Patient 2, hour 1: lag should be patient 2 hour 0, NOT patient 1 hour 1
        p2_h1 = df_ar[(df_ar["subject_id"] == 2) & (df_ar["hours_in"] == 1)]
        assert p2_h1["hr_lag1"].values[0] == 90  # Patient 2's hour 0

        # Patient 3, hour 1: lag should be patient 3 hour 0
        p3_h1 = df_ar[(df_ar["subject_id"] == 3) & (df_ar["hours_in"] == 1)]
        assert p3_h1["hr_lag1"].values[0] == 70  # Patient 3's hour 0

    def test_static_columns_unchanged(self):
        """Test that static columns don't get lag versions."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 1],
                "hours_in": [0, 1, 2],
                "hr": [80, 82, 79],
                "age": [65, 65, 65],
                "gender": ["M", "M", "M"],
            }
        )

        df_ar, target_cols, condition_cols = prepare_autoregressive_data(
            df, "subject_id", "hours_in", ["age", "gender"], lag=1
        )

        # Static columns should NOT have lag versions
        assert "age_lag1" not in df_ar.columns
        assert "gender_lag1" not in df_ar.columns

        # Static columns should be in condition, not target
        assert "age" not in target_cols
        assert "gender" not in target_cols
        assert "age" in condition_cols
        assert "gender" in condition_cols

        # Dynamic should have lag
        assert "hr" in target_cols
        assert "hr_lag1" in condition_cols

    def test_id_columns_not_lagged(self):
        """Test that ID columns (hadm_id, icustay_id) are never lagged or used as features."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 1, 2, 2, 2],
                "hadm_id": [100, 100, 100, 200, 200, 200],
                "icustay_id": [1000, 1000, 1000, 2000, 2000, 2000],
                "hours_in": [0, 1, 2, 0, 1, 2],
                "hr": [80, 82, 84, 90, 92, 94],
                "bp": [120, 118, 122, 130, 128, 132],
                "age": [65, 65, 65, 70, 70, 70],
            }
        )

        df_ar, target_cols, condition_cols = prepare_autoregressive_data(
            df, "subject_id", "hours_in", ["age"], lag=1
        )

        # ID columns should NOT have lagged versions created
        assert "hadm_id_lag1" not in df_ar.columns, "hadm_id should not be lagged"
        assert "icustay_id_lag1" not in df_ar.columns, "icustay_id should not be lagged"
        assert "subject_id_lag1" not in df_ar.columns, "subject_id should not be lagged"

        # ID columns should NOT appear in target or condition
        assert "hadm_id" not in target_cols
        assert "icustay_id" not in target_cols
        assert "hadm_id" not in condition_cols
        assert "icustay_id" not in condition_cols

        # Only dynamic features (hr, bp) should be in targets
        assert set(target_cols) == {"hr", "bp"}

        # Condition should have static + lagged dynamic
        assert set(condition_cols) == {"age", "hr_lag1", "bp_lag1"}

    def test_mean_only_drops_std_and_count_when_mean_exists(self):
        """Hourly lab/vital aggregates: keep *_mean, drop matching *_std and *_count."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hours_in": [0, 1],
                "Heart_Rate_mean": [80.0, 82.0],
                "Heart_Rate_std": [1.0, 2.0],
                "Heart_Rate_count": [3.0, 4.0],
                "vent": [0, 1],
                "age": [65, 65],
            }
        )

        df_ar, target_cols, _ = prepare_autoregressive_data(
            df,
            "subject_id",
            "hours_in",
            static_cols=["age"],
            lag=1,
            keep_biophysical_mean_only=True,
        )

        assert "Heart_Rate_mean" in target_cols
        assert "Heart_Rate_std" not in df_ar.columns
        assert "Heart_Rate_count" not in df_ar.columns
        assert "Heart_Rate_mean_lag1" in df_ar.columns
        assert "vent" in target_cols

    def test_mean_only_disabled_keeps_std_count(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hours_in": [0, 1],
                "Heart_Rate_mean": [80.0, 82.0],
                "Heart_Rate_std": [1.0, 2.0],
                "age": [65, 65],
            }
        )

        df_ar, target_cols, _ = prepare_autoregressive_data(
            df,
            "subject_id",
            "hours_in",
            static_cols=["age"],
            lag=1,
            keep_biophysical_mean_only=False,
        )

        assert "Heart_Rate_std" in target_cols
        assert "Heart_Rate_std" in df_ar.columns

    def test_orphan_std_not_dropped_without_mean(self):
        """_std without sibling _mean is not treated as biophysical aggregate."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hours_in": [0, 1],
                "rolling_std": [0.1, 0.2],
                "age": [65, 65],
            }
        )

        df_ar, target_cols, _ = prepare_autoregressive_data(
            df,
            "subject_id",
            "hours_in",
            static_cols=["age"],
            lag=1,
            keep_biophysical_mean_only=True,
        )

        assert "rolling_std" in target_cols
        assert "rolling_std" in df_ar.columns

    def test_default_pruning_drops_sparse_remove_feature(self):
        df = pd.DataFrame(
            {
                "subject_id": [1, 1],
                "hours_in": [0, 1],
                "Creatinine_pleural_mean": [0.5, 0.6],
                "Heart_Rate_mean": [80.0, 82.0],
                "age": [65, 65],
            }
        )

        df_ar, target_cols, condition_cols = prepare_autoregressive_data(
            df,
            "subject_id",
            "hours_in",
            static_cols=["age"],
            lag=1,
            keep_biophysical_mean_only=True,
        )

        assert "Creatinine_pleural_mean" not in target_cols
        assert "Creatinine_pleural_mean" not in df_ar.columns
        assert "Creatinine_pleural_mean_lag1" not in condition_cols
        assert "Heart_Rate_mean" in target_cols


class TestSplitStaticDynamic:
    """Test split_static_dynamic function."""

    def test_split_without_reference_file(self):
        """Test split when static_data.csv doesn't exist."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 2, 3],
                "hours_in": [0, 0, 0],
                "hr": [80, 85, 90],
                "age": [65, 55, 75],
            }
        )

        # Should fall back to heuristic
        static_cols, dynamic_cols = split_static_dynamic(
            df, static_data_path="nonexistent_path.csv"
        )

        # Without reference, should identify all non-ID/time as dynamic
        assert "subject_id" not in static_cols
        assert "subject_id" not in dynamic_cols
        assert "hours_in" not in static_cols
        assert "hours_in" not in dynamic_cols

    def test_correct_identification(self):
        """Test that ID and time columns are excluded."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 2, 2],
                "hadm_id": [10, 10, 20, 20],
                "icustay_id": [100, 100, 200, 200],
                "hours_in": [0, 1, 0, 1],
                "hr": [80, 82, 85, 87],
                "age": [65, 65, 55, 55],
            }
        )

        static_cols, dynamic_cols = split_static_dynamic(
            df, static_data_path="nonexistent.csv"
        )

        # ID/time columns should be excluded
        all_cols = static_cols + dynamic_cols
        assert "subject_id" not in all_cols
        assert "hadm_id" not in all_cols
        assert "icustay_id" not in all_cols
        assert "hours_in" not in all_cols

        # Age and hr should be present
        assert "age" in all_cols or len(static_cols) == 0
        assert "hr" in all_cols or len(dynamic_cols) == 0


class TestEdgeCases:
    """Test edge cases."""

    def test_unsorted_data(self):
        """Test that data is sorted before creating lags."""
        # Deliberately unsorted data
        df = pd.DataFrame(
            {
                "subject_id": [1, 2, 1, 2, 1, 2],
                "hours_in": [2, 1, 0, 0, 1, 2],
                "hr": [79, 87, 80, 85, 82, 84],
                "age": [65, 55, 65, 55, 65, 55],
            }
        )

        df_ar, _, _ = prepare_autoregressive_data(
            df, "subject_id", "hours_in", ["age"], lag=1
        )

        # Should be sorted by subject_id and hours_in
        assert df_ar["subject_id"].is_monotonic_increasing or (
            df_ar.groupby("subject_id")["hours_in"].is_monotonic_increasing.all()
        )

        # Check lags are correct after sorting
        p1_h1 = df_ar[(df_ar["subject_id"] == 1) & (df_ar["hours_in"] == 1)]
        assert p1_h1["hr_lag1"].values[0] == 80  # Patient 1's hour 0

    def test_missing_values_in_lag_columns(self):
        """Test that missing values propagate through lag creation."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 1, 1],
                "hours_in": [0, 1, 2, 3],
                "hr": [80, np.nan, 79, 85],
                "age": [65, 65, 65, 65],
            }
        )

        df_ar, _, _ = prepare_autoregressive_data(
            df, "subject_id", "hours_in", ["age"], lag=1
        )

        # Hour 2 should have lag = NaN (from hour 1 which was NaN)
        hour2 = df_ar[df_ar["hours_in"] == 2]
        # After prepare_autoregressive_data, NaN in original propagates to lag
        # The function uses fillna(-1.0) by default, so check for that
        assert hour2["hr_lag1"].values[0] == -1.0 or pd.isna(hour2["hr_lag1"].values[0])

    def test_lag_greater_than_1(self):
        """Test lag > 1 creates correct features."""
        df = pd.DataFrame(
            {
                "subject_id": [1, 1, 1, 1, 1],
                "hours_in": [0, 1, 2, 3, 4],
                "hr": [80, 82, 84, 86, 88],
                "age": [65, 65, 65, 65, 65],
            }
        )

        df_ar, _, _ = prepare_autoregressive_data(
            df, "subject_id", "hours_in", ["age"], lag=2
        )

        # Check lag2 column created
        assert "hr_lag2" in df_ar.columns

        # Hour 2 should have lag2 from hour 0
        hour2 = df_ar[df_ar["hours_in"] == 2]
        assert hour2["hr_lag2"].values[0] == 80

        # Hour 1 should have fill value (no hour -1)
        hour1 = df_ar[df_ar["hours_in"] == 1]
        assert hour1["hr_lag2"].values[0] == -1.0  # Default fill


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
