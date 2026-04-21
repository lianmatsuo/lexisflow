"""Data loading, preprocessing, and transformation for synthetic data generation.

This package handles:
- Loading raw CSV/Parquet data
- Preprocessing and feature transformation (TabularPreprocessor)
- Autoregressive data preparation for time series
- Feature detection and column utilities
"""

from .loaders import (
    load_csv,
    load_parquet,
    load_mimic_flat_table,
)
from .transformers import TabularPreprocessor
from .autoregressive import (
    prepare_autoregressive_data,
    split_static_dynamic,
)
from .feature_utils import (
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
    DEFAULT_DATETIME_COLUMNS,
)

__all__ = [
    # Loaders
    "load_csv",
    "load_parquet",
    "load_mimic_flat_table",
    # Transformers
    "TabularPreprocessor",
    # Autoregressive
    "prepare_autoregressive_data",
    "split_static_dynamic",
    # Feature utilities
    "columns_to_drop_mean_only_biophysical",
    "columns_to_drop_default_feature_pruning",
    "normalize_column_name",
    "normalize_column_name_list",
    "flatten_column_names",
    "is_lagged",
    "get_base_feature_name",
    "is_binary_feature",
    "identify_feature_types",
    "get_known_binary_features_with_lag",
    "KNOWN_BINARY_FEATURES",
    "SPARSE_REMOVE_FEATURES",
    "SPARSE_REMOVE_FEATURES_IF_PANEL",
    "DEFAULT_DATETIME_COLUMNS",
]
