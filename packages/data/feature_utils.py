"""Centralized feature detection and column utilities."""

from __future__ import annotations

import ast
from collections.abc import Iterable

import pandas as pd


def _flatten_one_column_name(col) -> str:
    """Map a tuple or tuple-like string header to ``Name_stat`` (e.g. ``Heart_Rate_mean``)."""
    if isinstance(col, tuple):
        return "_".join(str(x).replace(" ", "_") for x in col)
    if isinstance(col, str):
        s = col.strip()
        if len(s) >= 2 and s.startswith("(") and s.endswith(")"):
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError, TypeError):
                return col
            if isinstance(parsed, tuple) and len(parsed) == 2:
                return "_".join(str(x).replace(" ", "_") for x in parsed)
    return col


DEFAULT_DATETIME_COLUMNS = (
    "admittime",
    "dischtime",
    "intime",
    "outtime",
    "deathtime",
    "edregtime",
    "edouttime",
    "dnr_first_charttime",
    "timecmo_chart",
)


def normalize_column_name(col: object) -> str:
    """Normalize one column name into flattened project style.

    Handles tuple-style names and tuple-like strings from CSV headers such as
    ``"('Albumin ascites', 'std')"`` -> ``"Albumin_ascites_std"``.
    """
    out = _flatten_one_column_name(col)
    return str(out)


def normalize_column_name_list(columns: Iterable[object]) -> list[str]:
    """Normalize and de-duplicate a list of column names while preserving order."""
    normalized = [normalize_column_name(col) for col in columns]
    return list(dict.fromkeys(normalized))


SPARSE_REMOVE_FEATURES = frozenset(
    {
        "Creatinine_body_fluid_mean",
        "Lymphocytes_atypical_CSL_mean",
        "Creatinine_pleural_mean",
        "Creatinine_ascites_mean",
        "Albumin_ascites_mean",
        "Albumin_urine_mean",
        "Lymphocytes_percent_mean",
        "Albumin_pleural_mean",
        "Calcium_urine_mean",
        "Red_blood_cell_count_pleural_mean",
        "Lymphocytes_pleural_mean",
        "Lactate_dehydrogenase_pleural_mean",
        "Eosinophils_mean",
        "Lymphocytes_ascites_mean",
        "Red_blood_cell_count_ascites_mean",
        "Total_Protein_mean",
        "Lymphocytes_body_fluid_mean",
        "Monocytes_CSL_mean",
        "Red_blood_cell_count_CSF_mean",
        "Total_Protein_Urine_mean",
        "Venous_PvO2_mean",
        "Lymphocytes_atypical_mean",
        "Troponin-I_mean",
        "Chloride_urine_mean",
        "Cholesterol_LDL_mean",
        "Post_Void_Residual_mean",
        "Cholesterol_HDL_mean",
        "Cholesterol_mean",
        "White_blood_cell_count_urine_mean",
        "Red_blood_cell_count_urine_mean",
        "Pulmonary_Capillary_Wedge_Pressure_mean",
        "Creatinine_urine_mean",
        "Height_mean",
        "Cardiac_Output_fick_mean",
        "Basophils_mean",
        "Lactate_dehydrogenase_mean",
        "Fibrinogen_mean",
    }
)


def columns_to_drop_mean_only_biophysical(columns: Iterable[str]) -> list[str]:
    """Names of ``*_std`` / ``*_count`` columns to drop when a sibling ``*_mean`` exists.

    After :func:`flatten_column_names`, MIMIC-Extract vitals/labs appear as
    ``{Variable}_mean``, ``{Variable}_std``, ``{Variable}_count``. Dropping the
    list returned here leaves one trajectory per marker (e.g. keep
    ``White_blood_cell_count_mean``, drop ``White_blood_cell_count_std``).

    A column ending in ``_std`` or ``_count`` is listed only if ``{base}_mean`` is
    present, so unrelated names like ``rolling_std`` without ``rolling_mean`` are
    not dropped.
    """
    names = list(columns)
    col_set = set(names)
    mean_suffix = "_mean"
    to_drop: list[str] = []
    for col in names:
        if col.endswith(mean_suffix):
            continue
        if col.endswith("_std"):
            base = col[: -len("_std")]
            if f"{base}{mean_suffix}" in col_set:
                to_drop.append(col)
        elif col.endswith("_count"):
            base = col[: -len("_count")]
            if f"{base}{mean_suffix}" in col_set:
                to_drop.append(col)
    return to_drop


def columns_to_drop_default_feature_pruning(columns: Iterable[str]) -> list[str]:
    """Unified feature-pruning policy used by hour-0 and autoregressive prep.

    Includes:
    1) ``*_std`` / ``*_count`` when sibling ``*_mean`` exists.
    2) A fixed sparse-feature removal list (the 37 ``REMOVE`` columns from
       ``results/column_removal_recommendations.csv``), plus matching ``_lag1``
       columns if they are already present.
    """
    names = list(columns)
    col_set = set(names)
    drop = set(columns_to_drop_mean_only_biophysical(names))

    for col in SPARSE_REMOVE_FEATURES:
        if col in col_set:
            drop.add(col)
        lag_col = f"{col}_lag1"
        if lag_col in col_set:
            drop.add(lag_col)

    # Preserve source ordering for deterministic logs/tests.
    return [col for col in names if col in drop]


# Known binary features from clinical domain knowledge
KNOWN_BINARY_FEATURES = {
    "vent",
    "vaso",
    "adenosine",
    "dobutamine",
    "dopamine",
    "epinephrine",
    "isuprel",
    "milrinone",
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
    "colloid_bolus",
    "crystalloid_bolus",
    "nivdurations",
    "mort_icu",
    "mort_hosp",
    "hospital_expire_flag",
    "readmission_30",
    "fullcode_first",
    "dnr_first",
    "fullcode",
    "dnr",
    "cmo_first",
    "cmo_last",
    "cmo",
    "hospstay_seq",
    "gender",
    "sedation",
}


def flatten_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert tuple columns (or CSV string headers like ``\"('Albumin', 'mean')\"``) to flat names.

    MIMIC-Extract ``flat_table.csv`` often stores multi-index columns as the string
    form ``('Variable name', 'mean')`` rather than actual Python tuples; those must
    be parsed so :func:`columns_to_drop_mean_only_biophysical` can match ``*_mean``.

    Example:
        ('Heart Rate', 'mean') → 'Heart_Rate_mean'
        \"('Temperature', 'std')\" → 'Temperature_std'

    Args:
        df: DataFrame with potentially tuple or tuple-string column names

    Returns:
        DataFrame with flattened string column names
    """
    new_cols = [normalize_column_name(col) for col in df.columns]
    df.columns = new_cols
    return df


def is_lagged(col: str) -> bool:
    """Check if column is a lag feature.

    Args:
        col: Column name (string only after flattening)

    Returns:
        True if column represents a lagged feature

    Example:
        >>> is_lagged('Heart_Rate_mean_lag1')
        True
        >>> is_lagged('Heart_Rate_mean')
        False
    """
    return isinstance(col, str) and col.endswith("_lag1")


def get_base_feature_name(col: str) -> str:
    """Get base feature name by removing _lag1 suffix.

    Args:
        col: Column name

    Returns:
        Base feature name without lag suffix

    Example:
        >>> get_base_feature_name('Heart_Rate_mean_lag1')
        'Heart_Rate_mean'
        >>> get_base_feature_name('age')
        'age'
    """
    if is_lagged(col):
        return col.replace("_lag1", "")
    return col


def is_binary_feature(col: str, df: pd.DataFrame = None, strict: bool = True) -> bool:
    """Identify binary features.

    Args:
        col: Column name (string only, no tuples)
        df: Optional DataFrame sample to check values
        strict: If True, require integer dtype AND only {0,1} values

    Returns:
        True if binary feature

    Example:
        >>> df = pd.DataFrame({'vent': [0, 1, 0, 1], 'hr': [80, 90, 85, 88]})
        >>> is_binary_feature('vent', df)
        True
        >>> is_binary_feature('hr', df)
        False
    """
    # Check known list
    base_name = get_base_feature_name(col)
    if base_name in KNOWN_BINARY_FEATURES:
        return True

    # Auto-detect if DataFrame provided
    if strict and df is not None and col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_integer_dtype(df[col]):
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                    return True

    return False


def identify_feature_types(
    df: pd.DataFrame,
    columns: list[str] = None,
) -> dict[str, list[str]]:
    """Identify numeric, binary, and categorical columns.

    Args:
        df: DataFrame to analyze
        columns: Optional list of columns to analyze (default: all columns)

    Returns:
        Dictionary with keys:
            - 'numeric': Continuous numeric features
            - 'binary': Binary 0/1 features
            - 'categorical': Categorical string features
            - 'int': Integer features (subset of numeric, for rounding)

    Example:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35],
        ...     'vent': [0, 1, 0],
        ...     'gender': ['M', 'F', 'M'],
        ...     'heart_rate': [80.5, 90.2, 85.7]
        ... })
        >>> types = identify_feature_types(df)
        >>> types['binary']
        ['vent']
        >>> types['numeric']
        ['age', 'heart_rate']
        >>> types['categorical']
        ['gender']
    """
    if columns is None:
        columns = df.columns.tolist()

    numeric_cols = []
    binary_cols = []
    categorical_cols = []
    int_cols = []

    for col in columns:
        if col not in df.columns:
            continue

        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            # Check if binary
            if is_binary_feature(col, df, strict=True):
                binary_cols.append(col)
            else:
                numeric_cols.append(col)
                # Track integer columns for rounding on inverse transform
                if pd.api.types.is_integer_dtype(dtype):
                    int_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "binary": binary_cols,
        "categorical": categorical_cols,
        "int": int_cols,
    }


def get_known_binary_features_with_lag() -> set[str]:
    """Get set of all known binary features including their lagged versions.

    Returns:
        Set of binary feature names (base + _lag1 versions)

    Example:
        >>> binary_feats = get_known_binary_features_with_lag()
        >>> 'vent' in binary_feats
        True
        >>> 'vent_lag1' in binary_feats
        True
    """
    binary_with_lag = set()
    for feat in KNOWN_BINARY_FEATURES:
        binary_with_lag.add(feat)
        binary_with_lag.add(f"{feat}_lag1")
    return binary_with_lag
