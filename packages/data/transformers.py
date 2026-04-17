"""Tabular preprocessing for Forest-Flow."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TabularPreprocessor:
    """Preprocess mixed-type DataFrames for Forest-Flow.

    Minimal schema-based preprocessing optimized for XGBoost:
    - Categorical features: Label encoding (necessary for flow matching arithmetic)
    - Numeric features: Pass through raw values (XGBoost is scale-invariant)
    - Binary features: Enforce 0/1 dtype
    - NaN handling: Consistent fillna(0) strategy

    Supports streaming via partial_fit() to handle datasets larger than RAM.

    IMPORTANT: For production use, fit the preprocessor on the FULL dataset
    before training on subsamples. This ensures all categorical values are
    encoded (including rare ones).

    Use scripts/01b_fit_preprocessor.py to fit on full data once, then reuse
    the saved preprocessor for all subsequent training runs.

    Design Philosophy:
    - Schema enforcement, not feature engineering
    - Minimal transformations (XGBoost handles raw values well)
    - Label encoding for categoricals (can't interpolate strings)
    - No scaling for numerics (trees are scale-invariant)
    """

    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        int_cols: list[str] | None = None,
        binary_cols: list[str] | None = None,
    ):
        """Initialize preprocessor.

        Args:
            numeric_cols: List of numeric column names (continuous values).
            categorical_cols: List of categorical column names (multi-class strings).
            int_cols: Optional list of numeric columns to cast to int on inverse.
            binary_cols: Optional list of binary column names (0/1 values).
                        These are treated as numeric during preprocessing but automatically
                        rounded to {0, 1} during inverse transformation.
                        Provides explicit handling for binary flags (interventions, medications).

        Note on binary features:
            Binary features CAN be included in numeric_cols (current approach), but
            specifying them in binary_cols provides:
            - Explicit semantic meaning (clearer code)
            - Automatic rounding on inverse transform
            - Better documentation of data types
            - More efficient than categorical (50% less memory)
        """
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.int_cols = list(int_cols) if int_cols else []
        self.binary_cols = list(binary_cols) if binary_cols else []

        self.category_mappings: dict[str, dict] | None = None
        self.category_inverse_mappings: dict[str, dict] | None = None
        self._is_fitted = False
        self._all_nan_numeric: list[str] = []
        self._all_nan_categorical: list[str] = []
        # Track which columns have seen at least one non-NaN value (for streaming)
        self._has_non_nan: dict[str, bool] = {}
        # Track data statistics for validation (not used for scaling)
        self._n_samples_seen: int = 0

    def partial_fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Incrementally fit the preprocessor on a chunk of data.

        Call this repeatedly on chunks, then call finalize_fit() when done.
        This allows fitting on datasets larger than RAM.

        Args:
            df: Chunk of training DataFrame.

        Returns:
            self
        """
        # Initialize on first call
        if self._n_samples_seen == 0:
            # Initialize tracking for all-NaN detection
            all_numeric = self.numeric_cols + self.binary_cols
            for col in all_numeric:
                self._has_non_nan[col] = False
            for col in self.categorical_cols:
                self._has_non_nan[col] = False
            # Initialize categorical value collection (for label encoding)
            self._categorical_values = {col: set() for col in self.categorical_cols}

        # Track which columns have non-NaN values
        all_numeric = self.numeric_cols + self.binary_cols
        for col in all_numeric:
            if not self._has_non_nan.get(col, False):
                if col in df.columns:
                    has_data = bool(df[col].notna().any())
                    if has_data:
                        self._has_non_nan[col] = True
        for col in self.categorical_cols:
            if not self._has_non_nan.get(col, False):
                if col in df.columns:
                    has_data = bool(df[col].notna().any())
                    if has_data:
                        self._has_non_nan[col] = True

        # Collect categorical values from this chunk
        for col in self.categorical_cols:
            if col in df.columns:
                # Add all unique values (including handling NaN)
                values = df[col].dropna().unique()
                self._categorical_values[col].update(values)
                # Track if this column has NaN
                if df[col].isna().any():
                    self._categorical_values[col].add("__NAN__")  # Special token

        # Track number of samples seen (for statistics)
        self._n_samples_seen += len(df)

        return self

    def finalize_fit(self) -> "TabularPreprocessor":
        """Finalize fitting after all partial_fit() calls.

        Must be called after streaming through all data with partial_fit().
        Builds label encoding mappings for categorical features.
        """
        if self._n_samples_seen == 0:
            raise RuntimeError("Must call partial_fit() before finalize_fit()")

        # Identify columns that were all-NaN across entire dataset
        all_numeric = self.numeric_cols + self.binary_cols
        self._all_nan_numeric = [
            col for col in all_numeric if not self._has_non_nan.get(col, False)
        ]
        self._all_nan_categorical = [
            col
            for col in self.categorical_cols
            if not self._has_non_nan.get(col, False)
        ]

        if self._all_nan_numeric:
            print(
                f"  Found {len(self._all_nan_numeric)} all-NaN numeric columns (will fill with 0)"
            )
        if self._all_nan_categorical:
            print(
                f"  Found {len(self._all_nan_categorical)} all-NaN categorical columns (will fill with 0)"
            )

        # Build label encoding mappings from collected categorical values
        self.category_mappings = {}
        self.category_inverse_mappings = {}

        for col in self.categorical_cols:
            if col in self._categorical_values:
                # Sort categories for consistent encoding
                categories = sorted(self._categorical_values[col])
                # Map category → integer code
                self.category_mappings[col] = {
                    cat: idx for idx, cat in enumerate(categories)
                }
                # Map integer code → category
                self.category_inverse_mappings[col] = {
                    idx: cat for idx, cat in enumerate(categories)
                }
            else:
                self.category_mappings[col] = {}
                self.category_inverse_mappings[col] = {}

        self._is_fitted = True
        return self

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fit the preprocessor on training data (loads all into memory).

        For large datasets, use partial_fit() + finalize_fit() instead.

        Args:
            df: Training DataFrame with numeric, binary, and categorical columns.

        Returns:
            self
        """
        if len(df) == 0:
            raise ValueError("Cannot fit on empty DataFrame")

        # Detect all-NaN columns (include binary in numeric check)
        all_numeric = self.numeric_cols + self.binary_cols
        self._all_nan_numeric = [col for col in all_numeric if df[col].isna().all()]
        self._all_nan_categorical = [
            col for col in self.categorical_cols if df[col].isna().all()
        ]

        if self._all_nan_numeric or self._all_nan_categorical:
            print(
                f"  Found {len(self._all_nan_numeric)} all-NaN numeric columns (will fill with 0)"
            )
            print(
                f"  Found {len(self._all_nan_categorical)} all-NaN categorical columns (will fill with 0)"
            )

        # Build label encoding mappings for categorical features
        self.category_mappings = {}
        self.category_inverse_mappings = {}

        for col in self.categorical_cols:
            if col in df.columns:
                # Get all unique values (including NaN handling)
                values = df[col].dropna().unique()
                categories = sorted(values)

                # Add special token for NaN
                if df[col].isna().any():
                    categories = ["__NAN__"] + list(categories)

                # Build bidirectional mappings
                self.category_mappings[col] = {
                    cat: idx for idx, cat in enumerate(categories)
                }
                self.category_inverse_mappings[col] = {
                    idx: cat for idx, cat in enumerate(categories)
                }
            else:
                self.category_mappings[col] = {}
                self.category_inverse_mappings[col] = {}

        # Track samples seen (for compatibility)
        self._n_samples_seen = len(df)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame to numpy array.

        Uses label encoding for categorical features (not one-hot encoding).
        Categorical features are encoded as integers for XGBoost's native
        categorical support.

        Numeric features are passed through in their original scale (no scaling).
        XGBoost decision trees are scale-invariant and work optimally with raw values.

        Args:
            df: DataFrame with same columns as training data.

        Returns:
            Array of shape (n_samples, d) where:
            - Numeric/binary features are in original units (raw values)
            - Categorical features are integer codes (0, 1, 2, ...)
            - NaN values are filled with 0
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fit before transform")

        # Convert numeric and binary columns to float
        all_numeric_cols = self.numeric_cols + self.binary_cols
        df_numeric = df[all_numeric_cols].astype(float)

        # Fill all-NaN columns with 0
        for col in self._all_nan_numeric:
            df_numeric[col] = 0.0

        # Pass through raw values (no scaling)
        numeric_values = df_numeric.values
        # NaN → 0 (consistent missing value strategy)
        numeric_values = np.nan_to_num(numeric_values, nan=0.0)

        # Label encode categorical columns
        if self.categorical_cols:
            categorical_encoded = np.zeros(
                (len(df), len(self.categorical_cols)), dtype=np.int32
            )

            for idx, col in enumerate(self.categorical_cols):
                if col in df.columns:
                    # Map values to integer codes
                    mapping = self.category_mappings[col]

                    # Handle each value
                    encoded_values = []
                    for val in df[col]:
                        if pd.isna(val):
                            # Encode NaN as special token
                            code = mapping.get("__NAN__", 0)
                        else:
                            # Encode known value, or 0 for unknown (rare in test set)
                            code = mapping.get(val, 0)
                        encoded_values.append(code)

                    categorical_encoded[:, idx] = encoded_values
                else:
                    # Column missing - fill with 0
                    categorical_encoded[:, idx] = 0

            # Concatenate: [numeric_values, categorical_codes]
            result = np.concatenate([numeric_values, categorical_encoded], axis=1)
        else:
            result = numeric_values

        return result

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        """Inverse transform array back to DataFrame.

        Decodes label-encoded categorical features back to original strings.
        Numeric features are passed through (no unscaling needed).

        Args:
            X: Array of shape (n_samples, d) where:
               - First (n_numeric + n_binary) columns are raw numeric values
               - Last n_categorical columns are integer codes

        Returns:
            DataFrame with original column structure.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fit before inverse_transform")

        # Split into numeric/binary and categorical parts
        n_numeric = len(self.numeric_cols)
        n_binary = len(self.binary_cols)
        n_categorical = len(self.categorical_cols)
        n_total_numeric = n_numeric + n_binary

        # Separate numeric/binary from categorical codes
        X_numeric_binary = X[:, :n_total_numeric]
        X_categorical_codes = (
            X[:, n_total_numeric:] if n_categorical > 0 else np.empty((len(X), 0))
        )

        # Split numeric and binary (no inverse scaling needed)
        X_numeric = X_numeric_binary[:, :n_numeric]
        X_binary = (
            X_numeric_binary[:, n_numeric:] if n_binary > 0 else np.empty((len(X), 0))
        )

        # Reconstruct numeric columns
        numeric_df = pd.DataFrame(X_numeric, columns=self.numeric_cols)

        # Round and cast integer columns
        for col in self.int_cols:
            if col in numeric_df.columns:
                numeric_df[col] = numeric_df[col].round().astype("Int64")

        # Reconstruct binary columns with automatic rounding
        if n_binary > 0:
            binary_df = pd.DataFrame(X_binary, columns=self.binary_cols)
            # Automatically round binary columns to {0, 1}
            for col in self.binary_cols:
                binary_df[col] = np.clip(np.round(binary_df[col]), 0, 1).astype(int)
        else:
            binary_df = pd.DataFrame()

        # Decode categorical columns from integer codes
        categorical_dfs = {}
        if n_categorical > 0:
            for idx, col in enumerate(self.categorical_cols):
                inverse_mapping = self.category_inverse_mappings[col]

                # Decode integer codes to categories
                codes = X_categorical_codes[:, idx].astype(int)
                decoded_values = []

                for code in codes:
                    # Get category from code, default to first category if unknown
                    category = inverse_mapping.get(code, inverse_mapping.get(0, np.nan))
                    # Replace __NAN__ token with actual NaN
                    if category == "__NAN__":
                        category = np.nan
                    decoded_values.append(category)

                categorical_dfs[col] = pd.Series(decoded_values, name=col)

        # Concatenate all columns at once to avoid fragmentation
        dfs_to_concat = [numeric_df]
        if n_binary > 0:
            dfs_to_concat.append(binary_df)
        if categorical_dfs:
            dfs_to_concat.extend(list(categorical_dfs.values()))

        if len(dfs_to_concat) > 1:
            result = pd.concat(dfs_to_concat, axis=1)
        else:
            result = numeric_df

        return result

    @property
    def n_features(self) -> int:
        """Return total number of features after transformation.

        With label encoding, each categorical column becomes a single integer column.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fit first")
        return (
            len(self.numeric_cols) + len(self.binary_cols) + len(self.categorical_cols)
        )

    def get_feature_types(self) -> list[str]:
        """Get feature type list for XGBoost DMatrix.

        Returns list where each element is 'q' (quantitative) or 'c' (categorical).
        Order matches the transformed feature array:
        [numeric..., binary..., categorical...]

        Returns:
            List of 'q' or 'c' for each feature.

        Example:
            >>> preprocessor = TabularPreprocessor(
            ...     numeric_cols=['age', 'hr'],
            ...     binary_cols=['vent'],
            ...     categorical_cols=['gender', 'insurance']
            ... )
            >>> preprocessor.fit(df)
            >>> preprocessor.get_feature_types()
            ['q', 'q', 'q', 'c', 'c']  # age, hr, vent, gender, insurance
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fit first")

        feature_types = []
        feature_types.extend(["q"] * len(self.numeric_cols))
        feature_types.extend(["q"] * len(self.binary_cols))
        feature_types.extend(["c"] * len(self.categorical_cols))

        return feature_types

    def transformed_columns(self) -> list[str]:
        """Return the transformed-array column order: [numeric, binary, categorical]."""
        return (
            list(self.numeric_cols)
            + list(self.binary_cols)
            + list(self.categorical_cols)
        )

    def split_indices(
        self,
        target_cols: list[str],
        condition_cols: list[str],
    ) -> tuple[list[int], list[int]]:
        """Map semantic target/condition column names to transformed-array indices.

        Used by autoregressive training/sampling code that needs to slice the
        transformed feature array into ``X_target`` and ``X_condition`` columns.

        Raises:
            ValueError: On missing columns, overlap, or incomplete coverage.
        """
        transformed = self.transformed_columns()
        index_by_col = {col: idx for idx, col in enumerate(transformed)}

        missing_target = [c for c in target_cols if c not in index_by_col]
        missing_condition = [c for c in condition_cols if c not in index_by_col]
        if missing_target or missing_condition:
            raise ValueError(
                "Preprocessor/output column mismatch. "
                f"Missing target cols: {missing_target[:5]}, "
                f"missing condition cols: {missing_condition[:5]}"
            )

        target_indices = [index_by_col[c] for c in target_cols]
        condition_indices = [index_by_col[c] for c in condition_cols]

        overlap = set(target_indices).intersection(condition_indices)
        if overlap:
            raise ValueError(
                f"Target/condition split overlap at indices: {sorted(overlap)[:10]}"
            )

        if len(target_indices) + len(condition_indices) != len(transformed):
            raise ValueError(
                "Target + condition index counts do not cover transformed feature space. "
                f"target={len(target_indices)}, condition={len(condition_indices)}, "
                f"total={len(transformed)}"
            )

        return target_indices, condition_indices
