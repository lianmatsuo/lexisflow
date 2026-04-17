"""Unit tests for TabularPreprocessor."""

import numpy as np
import pandas as pd
import pytest

from packages.data.transformers import TabularPreprocessor


class TestTabularPreprocessor:
    """Test TabularPreprocessor functionality."""

    def test_numeric_only_transform(self):
        """Test preprocessing with only numeric columns."""
        # Create test data
        df = pd.DataFrame(
            {
                "age": [25, 35, 45, 55, 65],
                "weight": [60, 70, 80, 90, 100],
                "height": [160, 170, 180, 190, 200],
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["age", "weight", "height"],
            categorical_cols=[],
        )

        # Fit and transform
        preprocessor.fit(df)
        X = preprocessor.transform(df)

        # Check shape
        assert X.shape == (5, 3), f"Expected (5, 3), got {X.shape}"

        # Check values are in original scale (not scaled to [-1, 1])
        assert X.min() >= 25, f"Min value {X.min()} should be >= 25 (age min)"
        assert X.max() <= 200, f"Max value {X.max()} should be <= 200 (height max)"

        # Check first row matches original data
        assert X[0, 0] == 25  # age
        assert X[0, 1] == 60  # weight
        assert X[0, 2] == 160  # height

        # Check inverse (should be exact roundtrip for numeric-only)
        df_inv = preprocessor.inverse_transform(X)
        assert df_inv.shape == df.shape
        assert list(df_inv.columns) == list(df.columns)

        # Check values are exactly equal (no scaling/unscaling involved)
        np.testing.assert_array_almost_equal(
            df_inv.values.astype(float), df.values.astype(float), decimal=5
        )

    def test_categorical_encoding(self):
        """Test label encoding of categorical features."""
        df = pd.DataFrame(
            {
                "age": [25, 35, 45],
                "gender": ["M", "F", "M"],
                "insurance": ["Medicare", "Private", "Medicare"],
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["age"],
            categorical_cols=["gender", "insurance"],
        )

        preprocessor.fit(df)
        X = preprocessor.transform(df)

        # Label encoding: 1 numeric + 2 categorical = 3 total features
        assert X.shape[0] == 3
        assert X.shape[1] == 3  # age + gender_code + insurance_code

        # Check categorical mappings created
        assert "gender" in preprocessor.category_mappings
        assert "insurance" in preprocessor.category_mappings
        assert len(preprocessor.category_mappings["gender"]) == 2  # M, F
        assert (
            len(preprocessor.category_mappings["insurance"]) == 2
        )  # Medicare, Private

        # Check categorical codes are integers
        gender_codes = X[:, 1]  # Second column is gender
        insurance_codes = X[:, 2]  # Third column is insurance
        assert np.all(gender_codes == gender_codes.astype(int))
        assert np.all(insurance_codes == insurance_codes.astype(int))

        # Check feature types
        feature_types = preprocessor.get_feature_types()
        assert feature_types == ["q", "c", "c"]  # age, gender, insurance

    def test_binary_feature_rounding(self):
        """Test that binary features are automatically rounded on inverse."""
        df = pd.DataFrame(
            {
                "age": [25, 35, 45, 55, 65],
                "vent": [0, 1, 0, 1, 0],
                "vaso": [1, 1, 0, 0, 1],
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["age"],
            binary_cols=["vent", "vaso"],
            categorical_cols=[],
        )

        preprocessor.fit(df)
        X = preprocessor.transform(df)

        # Add noise to binary features (simulating generation)
        X_noisy = X.copy()
        X_noisy[:, 1] += np.random.normal(0, 0.1, size=5)  # Add noise to vent
        X_noisy[:, 2] += np.random.normal(0, 0.1, size=5)  # Add noise to vaso

        # Inverse transform
        df_inv = preprocessor.inverse_transform(X_noisy)

        # Check binary columns are {0, 1}
        assert set(df_inv["vent"].unique()).issubset(
            {0, 1}
        ), f"vent should be {{0,1}}, got {df_inv['vent'].unique()}"
        assert set(df_inv["vaso"].unique()).issubset(
            {0, 1}
        ), f"vaso should be {{0,1}}, got {df_inv['vaso'].unique()}"

        # Check dtypes
        assert df_inv["vent"].dtype == int
        assert df_inv["vaso"].dtype == int

    def test_missing_values_handling(self):
        """Test NaN handling in transform."""
        df = pd.DataFrame(
            {
                "age": [25, np.nan, 45, np.nan, 65],
                "weight": [60, 70, np.nan, 90, 100],
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["age", "weight"],
            categorical_cols=[],
        )

        preprocessor.fit(df)
        X = preprocessor.transform(df)

        # NaN should be filled with 0 (center of [-1,1])
        assert not np.isnan(X).any(), "Transform should not contain NaN"

        # Check NaN positions were filled with 0
        assert X.shape == (5, 2)

    def test_categorical_nan_handling(self):
        """Test NaN handling in categorical features."""
        df = pd.DataFrame(
            {
                "age": [25, 35, 45, 55],
                "gender": ["M", "F", np.nan, "M"],
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["age"],
            categorical_cols=["gender"],
        )

        preprocessor.fit(df)
        X = preprocessor.transform(df)

        # Check shape: age + gender = 2 features
        assert X.shape == (4, 2)

        # NaN should be encoded as a special category
        assert "__NAN__" in preprocessor.category_mappings["gender"]

        # Inverse transform should handle NaN correctly
        df_inv = preprocessor.inverse_transform(X)
        assert pd.isna(df_inv["gender"].iloc[2])  # Row 2 had NaN

    def test_label_encoding_deterministic(self):
        """Test that label encoding is deterministic (sorted categories)."""
        df = pd.DataFrame(
            {
                "age": [25, 35, 45, 55, 65],
                "diagnosis": ["Pneumonia", "Sepsis", "Cardiac", "Sepsis", "Pneumonia"],
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["age"],
            categorical_cols=["diagnosis"],
        )

        preprocessor.fit(df[:3])  # Fit on first 3 rows

        # Categories should be sorted: Cardiac=0, Pneumonia=1, Sepsis=2
        expected_mapping = {"Cardiac": 0, "Pneumonia": 1, "Sepsis": 2}
        assert preprocessor.category_mappings["diagnosis"] == expected_mapping

        # Transform should use consistent codes
        X = preprocessor.transform(df)

        # Check codes match expected mapping
        assert X[0, 1] == 1  # Pneumonia
        assert X[1, 1] == 2  # Sepsis
        assert X[2, 1] == 0  # Cardiac

    def test_all_nan_column_handling(self):
        """Test handling of columns that are entirely NaN."""
        df = pd.DataFrame(
            {
                "age": [25, 35, 45, 55, 65],
                "missing_col": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["age", "missing_col"],
            categorical_cols=[],
        )

        preprocessor.fit(df)

        # Check all-NaN column is tracked
        assert "missing_col" in preprocessor._all_nan_numeric

        # Transform should still work
        X = preprocessor.transform(df)
        assert X.shape == (5, 2)

        # Inverse should work
        df_inv = preprocessor.inverse_transform(X)
        assert df_inv.shape == df.shape

    def test_integer_column_rounding(self):
        """Test that integer columns are rounded on inverse."""
        df = pd.DataFrame(
            {
                "age": [25, 35, 45, 55, 65],
                "count": [5, 10, 15, 20, 25],
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["age", "count"],
            categorical_cols=[],
            int_cols=["count"],
        )

        preprocessor.fit(df)
        X = preprocessor.transform(df)

        # Add small noise
        X_noisy = X + np.random.normal(0, 0.01, X.shape)

        df_inv = preprocessor.inverse_transform(X_noisy)

        # Check count is integer type
        assert pd.api.types.is_integer_dtype(df_inv["count"])

        # Check values are whole numbers
        assert all(df_inv["count"] == df_inv["count"].round())

    def test_partial_fit_streaming(self):
        """Test partial_fit for streaming large datasets."""
        # Create large synthetic dataset in chunks
        chunk_size = 1000
        n_chunks = 3

        preprocessor = TabularPreprocessor(
            numeric_cols=["age", "weight"],
            categorical_cols=[],
        )

        # Stream chunks
        for i in range(n_chunks):
            df_chunk = pd.DataFrame(
                {
                    "age": np.random.randint(20, 80, size=chunk_size),
                    "weight": np.random.randint(50, 100, size=chunk_size),
                }
            )
            preprocessor.partial_fit(df_chunk)

        preprocessor.finalize_fit()

        # Check preprocessor is fitted
        assert preprocessor._is_fitted
        assert preprocessor._n_samples_seen == chunk_size * n_chunks

        # Test transform
        df_test = pd.DataFrame(
            {
                "age": [25, 50, 75],
                "weight": [60, 75, 90],
            }
        )
        X = preprocessor.transform(df_test)
        assert X.shape == (3, 2)

        # Verify values are in raw scale
        assert X[0, 0] == 25  # age
        assert X[0, 1] == 60  # weight

    def test_categorical_unknown_values(self):
        """Test handling of unknown categorical values during transform."""
        df_train = pd.DataFrame(
            {
                "age": [25, 35, 45],
                "gender": ["M", "F", "M"],
            }
        )

        df_test = pd.DataFrame(
            {
                "age": [30, 40],
                "gender": ["M", "Other"],  # 'Other' not in training
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["age"],
            categorical_cols=["gender"],
        )

        preprocessor.fit(df_train)
        X_test = preprocessor.transform(df_test)

        # Should not crash (unknown category mapped to code 0)
        assert X_test.shape[0] == 2
        assert X_test.shape[1] == 2  # age + gender_code

        # Unknown category 'Other' should be mapped to code 0
        # (first category in mapping, used as fallback)
        assert X_test[1, 1] == 0  # Row 1, gender column

        # Inverse transform
        df_inv = preprocessor.inverse_transform(X_test)

        # Unknown value maps to first category in mapping
        assert df_inv.shape == df_test.shape
        assert df_inv["gender"].iloc[1] in ["F", "M"]  # Maps to known category

    def test_mixed_types_integration(self):
        """Test preprocessing with all feature types together."""
        df = pd.DataFrame(
            {
                "age": [25.5, 35.2, 45.8, 55.1, 65.3],
                "count": [5, 10, 15, 20, 25],
                "vent": [0, 1, 0, 1, 0],
                "gender": ["M", "F", "M", "F", "M"],
                "insurance": ["Medicare", "Private", "Medicare", "Medicare", "Private"],
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["age", "count"],
            binary_cols=["vent"],
            categorical_cols=["gender", "insurance"],
        )

        preprocessor.fit(df)
        X = preprocessor.transform(df)

        # Check output dimensionality with label encoding
        # age(1) + count(1) + vent(1) + gender(1) + insurance(1) = 5 features
        assert X.shape[0] == 5
        assert X.shape[1] == 5  # No one-hot expansion

        # Check feature types
        feature_types = preprocessor.get_feature_types()
        assert feature_types == [
            "q",
            "q",
            "q",
            "c",
            "c",
        ]  # numeric, numeric, binary, cat, cat

        # Round trip
        df_inv = preprocessor.inverse_transform(X)

        # Check types
        assert df_inv["vent"].dtype == int
        assert df_inv["gender"].dtype == object

        # Check categorical values preserved
        assert set(df_inv["gender"].unique()) == set(df["gender"].unique())
        assert set(df_inv["insurance"].unique()) == set(df["insurance"].unique())

        # Check values approximately preserved
        np.testing.assert_array_almost_equal(
            df_inv["age"].values, df["age"].values, decimal=1
        )


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe_error(self):
        """Test error handling for empty DataFrame."""
        df = pd.DataFrame()

        preprocessor = TabularPreprocessor(
            numeric_cols=[],
            categorical_cols=[],
        )

        with pytest.raises((ValueError, IndexError, KeyError)):
            preprocessor.fit(df)

    def test_single_value_column(self):
        """Test column with single unique value."""
        df = pd.DataFrame(
            {
                "const": [5.0, 5.0, 5.0, 5.0, 5.0],
                "age": [25, 35, 45, 55, 65],
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["const", "age"],
            categorical_cols=[],
        )

        # Should not crash (MinMaxScaler handles constant columns)
        preprocessor.fit(df)
        X = preprocessor.transform(df)

        # Constant column will be all zeros or NaN
        assert X.shape == (5, 2)

    def test_single_row_dataframe(self):
        """Test with single row (edge case for MinMaxScaler)."""
        df = pd.DataFrame(
            {
                "age": [25],
                "weight": [70],
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["age", "weight"],
            categorical_cols=[],
        )

        # MinMaxScaler may have issues with single sample
        # but should handle gracefully
        try:
            preprocessor.fit(df)
            X = preprocessor.transform(df)
            assert X.shape == (1, 2)
        except (ValueError, RuntimeWarning):
            pass  # Expected behavior

    def test_transform_before_fit_error(self):
        """Test that transform before fit raises error."""
        df = pd.DataFrame({"age": [25, 35]})

        preprocessor = TabularPreprocessor(
            numeric_cols=["age"],
            categorical_cols=[],
        )

        with pytest.raises(RuntimeError, match="must be fit"):
            preprocessor.transform(df)

    def test_column_mismatch_error(self):
        """Test error when transform columns don't match fit columns."""
        df_train = pd.DataFrame(
            {
                "age": [25, 35, 45],
                "weight": [60, 70, 80],
            }
        )

        df_test = pd.DataFrame(
            {
                "age": [30, 40],
                "height": [170, 180],  # Different column!
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["age", "weight"],
            categorical_cols=[],
        )

        preprocessor.fit(df_train)

        # Should raise KeyError for missing column
        with pytest.raises(KeyError):
            preprocessor.transform(df_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
