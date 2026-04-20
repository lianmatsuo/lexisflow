"""Tests for feature detection utilities."""

import pandas as pd

from synth_gen.data.feature_utils import (
    columns_to_drop_mean_only_biophysical,
    columns_to_drop_default_feature_pruning,
    normalize_column_name,
    normalize_column_name_list,
    flatten_column_names,
    is_lagged,
    get_base_feature_name,
    is_binary_feature,
    identify_feature_types,
    get_known_binary_features_with_lag,
    KNOWN_BINARY_FEATURES,
    SPARSE_REMOVE_FEATURES,
    SPARSE_REMOVE_FEATURES_IF_PANEL,
)


class TestFlattenColumnNames:
    """Test column name flattening."""

    def test_tuple_columns(self):
        """Test flattening tuple columns."""
        df = pd.DataFrame(
            {
                ("Heart Rate", "mean"): [80, 90],
                ("Heart Rate", "std"): [5, 10],
                "age": [65, 70],
            }
        )

        df_flat = flatten_column_names(df)

        assert "Heart_Rate_mean" in df_flat.columns
        assert "Heart_Rate_std" in df_flat.columns
        assert "age" in df_flat.columns
        assert len(df_flat.columns) == 3

    def test_already_flat(self):
        """Test that flat columns remain unchanged."""
        df = pd.DataFrame(
            {
                "heart_rate": [80, 90],
                "temperature": [98.6, 99.1],
            }
        )

        df_flat = flatten_column_names(df)

        assert list(df_flat.columns) == ["heart_rate", "temperature"]
        pd.testing.assert_frame_equal(df, df_flat)

    def test_spaces_in_tuples(self):
        """Test that spaces in tuple elements are replaced with underscores."""
        df = pd.DataFrame(
            {
                ("Systolic BP", "max"): [120, 130],
            }
        )

        df_flat = flatten_column_names(df)

        assert "Systolic_BP_max" in df_flat.columns

    def test_tuple_like_string_headers_from_csv(self):
        """flat_table.csv uses string headers like \"('Albumin', 'mean')\" not tuples."""
        df = pd.DataFrame(
            {
                "('Albumin', 'mean')": [3.5, 3.6],
                "('Albumin', 'std')": [0.1, 0.2],
                "vent": [0, 1],
            }
        )

        df_flat = flatten_column_names(df)

        assert "Albumin_mean" in df_flat.columns
        assert "Albumin_std" in df_flat.columns
        assert "vent" in df_flat.columns
        drop = columns_to_drop_mean_only_biophysical(df_flat.columns)
        assert "Albumin_std" in drop
        assert "Albumin_mean" not in drop


class TestColumnNameNormalization:
    def test_normalize_tuple_like_string(self):
        assert (
            normalize_column_name("('Albumin ascites', 'std')") == "Albumin_ascites_std"
        )

    def test_normalize_list_deduplicates_preserving_order(self):
        names = [
            "('Albumin ascites', 'std')",
            "Albumin_ascites_std",
            "age",
            "age",
        ]
        assert normalize_column_name_list(names) == ["Albumin_ascites_std", "age"]


class TestColumnsToDropMeanOnlyBiophysical:
    """Test MIMIC hourly aggregate column pruning (keep *_mean only)."""

    def test_drops_std_and_count_when_mean_present(self):
        cols = ["Heart_Rate_mean", "Heart_Rate_std", "Heart_Rate_count", "age"]
        drop = columns_to_drop_mean_only_biophysical(cols)
        assert set(drop) == {"Heart_Rate_std", "Heart_Rate_count"}

    def test_keeps_orphan_std(self):
        cols = ["rolling_std", "age_mean"]
        assert columns_to_drop_mean_only_biophysical(cols) == []

    def test_white_blood_cell_count_naming(self):
        cols = [
            "White_blood_cell_count_mean",
            "White_blood_cell_count_std",
            "White_blood_cell_count_count",
        ]
        drop = columns_to_drop_mean_only_biophysical(cols)
        assert set(drop) == {
            "White_blood_cell_count_std",
            "White_blood_cell_count_count",
        }


class TestColumnsToDropDefaultFeaturePruning:
    """Test the shared default pruning policy."""

    def test_combines_std_count_and_sparse_remove_targets(self):
        cols = [
            "Heart_Rate_mean",
            "Heart_Rate_std",
            "Creatinine_pleural_mean",
            "age",
        ]
        drop = columns_to_drop_default_feature_pruning(cols)
        assert set(drop) == {"Heart_Rate_std", "Creatinine_pleural_mean"}

    def test_drops_sparse_paired_lag_when_present(self):
        cols = [
            "Creatinine_pleural_mean",
            "Creatinine_pleural_mean_lag1",
            "age",
        ]
        drop = columns_to_drop_default_feature_pruning(cols)
        assert set(drop) == {"Creatinine_pleural_mean", "Creatinine_pleural_mean_lag1"}

    def test_sparse_remove_list_size(self):
        assert len(SPARSE_REMOVE_FEATURES) == 37

    def test_sparse_remove_if_panel_list_size(self):
        assert len(SPARSE_REMOVE_FEATURES_IF_PANEL) == 27

    def test_drops_remove_if_panel_target_and_lag(self):
        cols = [
            "Lymphocytes_mean",
            "Lymphocytes_mean_lag1",
            "age",
        ]
        drop = columns_to_drop_default_feature_pruning(cols)
        assert set(drop) == {"Lymphocytes_mean", "Lymphocytes_mean_lag1"}


class TestIsLagged:
    """Test lagged column detection."""

    def test_lagged_column(self):
        """Test detection of lagged columns."""
        assert is_lagged("heart_rate_lag1")
        assert is_lagged("temperature_lag1")
        assert is_lagged("Heart_Rate_mean_lag1")

    def test_non_lagged_column(self):
        """Test that non-lagged columns return False."""
        assert not is_lagged("heart_rate")
        assert not is_lagged("age")
        assert not is_lagged("Heart_Rate_mean")

    def test_partial_match(self):
        """Test that partial matches don't trigger false positives."""
        assert not is_lagged("heart_rate_lag10")  # Different lag
        assert not is_lagged("lag1_heart_rate")  # Prefix not suffix


class TestGetBaseFeatureName:
    """Test base feature name extraction."""

    def test_lagged_feature(self):
        """Test removing _lag1 suffix."""
        assert get_base_feature_name("heart_rate_lag1") == "heart_rate"
        assert get_base_feature_name("vent_lag1") == "vent"

    def test_non_lagged_feature(self):
        """Test that non-lagged features remain unchanged."""
        assert get_base_feature_name("age") == "age"
        assert get_base_feature_name("heart_rate") == "heart_rate"


class TestIsBinaryFeature:
    """Test binary feature detection."""

    def test_known_binary_feature(self):
        """Test detection of known binary features."""
        assert is_binary_feature("vent", df=None, strict=False)
        assert is_binary_feature("vaso", df=None, strict=False)
        assert is_binary_feature("mort_icu", df=None, strict=False)

    def test_known_binary_lagged(self):
        """Test detection of lagged known binary features."""
        assert is_binary_feature("vent_lag1", df=None, strict=False)
        assert is_binary_feature("vaso_lag1", df=None, strict=False)

    def test_auto_detect_binary(self):
        """Test auto-detection of binary features from data."""
        df = pd.DataFrame(
            {
                "flag": [0, 1, 0, 1],
                "score": [10, 20, 30, 40],
            }
        )

        assert is_binary_feature("flag", df, strict=True)
        assert not is_binary_feature("score", df, strict=True)

    def test_binary_with_nulls(self):
        """Test binary detection with null values.

        Note: With strict=True, columns with nulls may fail integer check
        since pandas converts int columns to float when nulls are present.
        """
        df = pd.DataFrame(
            {
                "flag": [0, 1, None, 1],
            }
        )

        # With nulls, pandas makes it float, so strict=True won't detect it
        # But known binary features should still be detected by name
        assert not is_binary_feature("flag", df, strict=True)

        # But 'vent' (known binary) would be detected even with nulls
        df2 = pd.DataFrame(
            {
                "vent": [0, 1, None, 1],
            }
        )
        assert is_binary_feature("vent", df2, strict=False)

    def test_not_binary_with_other_values(self):
        """Test that columns with values other than 0/1 aren't binary."""
        df = pd.DataFrame(
            {
                "multi": [0, 1, 2, 3],  # Has values beyond {0, 1}
            }
        )

        assert not is_binary_feature("multi", df, strict=True)


class TestIdentifyFeatureTypes:
    """Test feature type identification."""

    def test_basic_types(self):
        """Test identification of basic feature types."""
        df = pd.DataFrame(
            {
                "age": [25, 30, 35],
                "vent": [0, 1, 0],
                "gender": ["M", "F", "M"],
                "heart_rate": [80.5, 90.2, 85.7],
            }
        )

        types = identify_feature_types(df)

        assert "age" in types["numeric"]
        assert "heart_rate" in types["numeric"]
        assert "vent" in types["binary"]
        assert "gender" in types["categorical"]

    def test_integer_tracking(self):
        """Test that integer columns are tracked separately."""
        df = pd.DataFrame(
            {
                "age": [25, 30, 35],  # Integer
                "heart_rate": [80.5, 90.2, 85.7],  # Float
            }
        )

        types = identify_feature_types(df)

        assert "age" in types["int"]
        assert "heart_rate" not in types["int"]

    def test_column_subset(self):
        """Test identification on column subset."""
        df = pd.DataFrame(
            {
                "age": [25, 30, 35],
                "vent": [0, 1, 0],
                "ignore_me": ["x", "y", "z"],
            }
        )

        types = identify_feature_types(df, columns=["age", "vent"])

        assert "age" in types["numeric"]
        assert "vent" in types["binary"]
        assert "ignore_me" not in types["categorical"]

    def test_empty_categories(self):
        """Test that empty categories are returned."""
        df = pd.DataFrame(
            {
                "age": [25, 30, 35],
            }
        )

        types = identify_feature_types(df)

        assert len(types["binary"]) == 0
        assert len(types["categorical"]) == 0


class TestGetKnownBinaryFeaturesWithLag:
    """Test known binary features helper."""

    def test_includes_base_and_lagged(self):
        """Test that both base and lagged versions are included."""
        binary_feats = get_known_binary_features_with_lag()

        assert "vent" in binary_feats
        assert "vent_lag1" in binary_feats
        assert "vaso" in binary_feats
        assert "vaso_lag1" in binary_feats

    def test_size(self):
        """Test that result contains twice the base features."""
        binary_feats = get_known_binary_features_with_lag()

        # Should have base + lagged for each known binary feature
        assert len(binary_feats) == 2 * len(KNOWN_BINARY_FEATURES)


class TestKnownBinaryFeatures:
    """Test the KNOWN_BINARY_FEATURES constant."""

    def test_common_features_present(self):
        """Test that common clinical binary features are included."""
        assert "vent" in KNOWN_BINARY_FEATURES
        assert "vaso" in KNOWN_BINARY_FEATURES
        assert "mort_icu" in KNOWN_BINARY_FEATURES
        assert "mort_hosp" in KNOWN_BINARY_FEATURES
        assert "gender" in KNOWN_BINARY_FEATURES
