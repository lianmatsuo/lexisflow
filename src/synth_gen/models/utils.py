"""Utility functions for Forest-Flow with MIMIC-III data."""

from __future__ import annotations

import pandas as pd


def load_static_columns(
    static_data_path: str = "data/processed/static_data.csv",
) -> list[str]:
    """Load list of static columns from MIMIC-III static_data.csv.

    Static columns are patient characteristics that don't change over time:
    - Demographics: gender, ethnicity, age
    - Admission info: insurance, admission_type, first_careunit
    - Outcome flags: mort_icu, mort_hosp, hospital_expire_flag
    - Time info: admittime, dischtime, intime, outtime, los_icu
    - Code status: fullcode, dnr, cmo flags

    Args:
        static_data_path: Path to static_data.csv file.

    Returns:
        List of static column names (excluding ID columns).

    Example:
        >>> static_cols = load_static_columns()
        >>> print(static_cols[:5])
        ['gender', 'ethnicity', 'age', 'insurance', 'admittime']
    """
    df_static = pd.read_csv(static_data_path)

    # Exclude ID columns
    id_columns = ["subject_id", "hadm_id", "icustay_id"]
    static_cols = [col for col in df_static.columns if col not in id_columns]

    return static_cols


def get_dynamic_columns(
    flat_table_path: str = "data/processed/flat_table.csv",
    static_data_path: str = "data/processed/static_data.csv",
    nrows: int | None = 1,
) -> list[str]:
    """Get list of dynamic (time-varying) columns from flat table data.

    Dynamic columns are features that change over time:
    - Vitals: heart rate, blood pressure, temperature, etc.
    - Lab values: various blood tests, chemistry panels
    - Medications: vasopressors, sedatives, etc.
    - Interventions: vent, CRRT, etc.
    - Fluid balance: inputs, outputs

    Args:
        flat_table_path: Path to flat table CSV file (default: data/processed/flat_table.csv).
        static_data_path: Path to static_data.csv file.
        nrows: Number of rows to read (1 is enough to get column names).

    Returns:
        List of dynamic column names.

    Example:
        >>> dynamic_cols = get_dynamic_columns()
        >>> print(dynamic_cols[:5])
        ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine']
    """
    df = pd.read_csv(flat_table_path, nrows=nrows)
    static_cols = load_static_columns(static_data_path)

    # Exclude ID columns, time column, and static columns
    exclude = ["subject_id", "hadm_id", "icustay_id", "hours_in"] + static_cols
    dynamic_cols = [col for col in df.columns if col not in exclude]

    return dynamic_cols


def split_static_dynamic(
    df: pd.DataFrame, static_data_path: str = "data/processed/static_data.csv"
) -> tuple[list[str], list[str]]:
    """Split DataFrame columns into static and dynamic.

    Args:
        df: DataFrame with both static and dynamic columns.
        static_data_path: Path to static_data.csv file.

    Returns:
        Tuple of (static_cols, dynamic_cols) present in df.

    Example:
        >>> df = pd.read_csv("data/processed/flat_table.csv", nrows=100)
        >>> static_cols, dynamic_cols = split_static_dynamic(df)
        >>> print(f"Static: {len(static_cols)}, Dynamic: {len(dynamic_cols)}")
    """
    static_cols_all = load_static_columns(static_data_path)

    # Only keep static columns that exist in df
    static_cols = [col for col in static_cols_all if col in df.columns]

    # Dynamic columns are everything except IDs, time, and static
    exclude = ["subject_id", "hadm_id", "icustay_id", "hours_in"] + static_cols
    dynamic_cols = [col for col in df.columns if col not in exclude]

    return static_cols, dynamic_cols


def get_mimic_column_info(
    static_data_path: str = "data/processed/static_data.csv",
) -> dict:
    """Get detailed information about MIMIC-III column categories.

    Returns:
        Dictionary with column categories and descriptions.
    """
    static_cols = load_static_columns(static_data_path)

    # Categorize static columns
    demographics = [col for col in static_cols if col in ["gender", "ethnicity", "age"]]
    admission_info = [
        col
        for col in static_cols
        if col
        in ["insurance", "admission_type", "first_careunit", "diagnosis_at_admission"]
    ]
    outcomes = [
        col
        for col in static_cols
        if "mort" in col or "expire" in col or "readmission" in col
    ]
    code_status = [
        col for col in static_cols if "code" in col or "dnr" in col or "cmo" in col
    ]
    timing = [col for col in static_cols if "time" in col.lower() or col == "los_icu"]
    discharge = [col for col in static_cols if "disch" in col.lower()]
    other_static = [
        col
        for col in static_cols
        if col
        not in demographics
        + admission_info
        + outcomes
        + code_status
        + timing
        + discharge
    ]

    return {
        "static": {
            "demographics": demographics,
            "admission_info": admission_info,
            "outcomes": outcomes,
            "code_status": code_status,
            "timing": timing,
            "discharge": discharge,
            "other": other_static,
        },
        "dynamic_categories": {
            "note": "Dynamic columns vary by preprocessing. Common categories:",
            "categories": [
                "Ventilation (vent, nivdurations)",
                "Vasopressors (vaso, norepinephrine, dopamine, etc.)",
                "Lab values (lactate, creatinine, bilirubin, etc.)",
                "Vitals (heart rate, blood pressure, temperature, etc.)",
                "Fluid balance (crystalloid_bolus, colloid_bolus, urine output)",
                "Medications (sedatives, antibiotics, etc.)",
            ],
        },
    }


if __name__ == "__main__":
    # Demo usage
    print("MIMIC-III Column Information")
    print("=" * 60)

    try:
        static_cols = load_static_columns()
        print(f"\nStatic columns ({len(static_cols)}):")
        print(f"  {', '.join(static_cols[:5])}...")

        info = get_mimic_column_info()
        print("\nStatic column categories:")
        for category, cols in info["static"].items():
            if cols:
                print(f"  {category}: {len(cols)} columns")
                print(f"    {', '.join(cols[:3])}{'...' if len(cols) > 3 else ''}")

        print("\nDynamic column categories:")
        for cat in info["dynamic_categories"]["categories"]:
            print(f"  - {cat}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you're running from the project root directory.")
