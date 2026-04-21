#!/usr/bin/env python3
"""Prepare hour-0 training rows from PhysioNet Challenge 2012 set-A.

Secondary reproducibility benchmark alongside MIMIC-III.

Input (downloaded to data/challenge2012/raw/):
    set-a/*.txt       - 4,000 per-patient long-format files
                        (columns: Time, Parameter, Value)
    Outcomes-a.txt    - one row per patient with mortality + LOS

Output:
    data/challenge2012/processed/hour0_data.csv
        subject_id, hours_in=0, static demographics, hour-0 vital medians,
        hospital_expire_flag, los_icu

Column-naming matches MIMIC output so downstream MortalityTask / LOSTask
work unchanged.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


STATIC_PARAMS = ["Age", "Gender", "Height", "Weight", "ICUType"]
# Dropped per sparsity audit (<10% coverage across 4000 patients):
DROP_PARAMS = {"TroponinI", "Cholesterol"}
# Sentinel for missing static values in raw files
MISSING_SENTINEL = -1


def parse_patient_file(path: Path) -> tuple[int, dict, pd.DataFrame]:
    """Return (record_id, static_dict, hour0_timeseries_df)."""
    df = pd.read_csv(path)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    record_id_row = df[df["Parameter"] == "RecordID"]
    record_id = int(record_id_row["Value"].iloc[0])

    static = {}
    for p in STATIC_PARAMS:
        row = df[df["Parameter"] == p]
        if row.empty:
            static[p] = pd.NA
            continue
        val = row["Value"].iloc[0]
        if pd.notna(val) and val == MISSING_SENTINEL and p in {"Height", "Weight"}:
            val = pd.NA
        static[p] = val

    df = df[~df["Parameter"].isin(STATIC_PARAMS + ["RecordID"])]
    df = df[~df["Parameter"].isin(DROP_PARAMS)]

    if df.empty:
        return record_id, static, df.assign(hour=pd.Series(dtype=int))

    # Parse HH:MM → hour index
    hour_series = df["Time"].str.split(":", expand=True)[0].astype(int)
    df = df.assign(hour=hour_series)
    hour0_df = df[df["hour"] == 0]

    return record_id, static, hour0_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Challenge 2012 hour-0 training data."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/challenge2012/raw",
        help="Directory with set-a/ and Outcomes-a.txt (default: data/challenge2012/raw)",
    )
    parser.add_argument(
        "--output-path",
        default="data/challenge2012/processed/hour0_data.csv",
        help="Output CSV path (default: data/challenge2012/processed/hour0_data.csv)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    set_a_dir = raw_dir / "set-a"
    outcomes_path = raw_dir / "Outcomes-a.txt"

    if not set_a_dir.is_dir() or not outcomes_path.is_file():
        print(f"ERROR: expected {set_a_dir}/ and {outcomes_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("CHALLENGE 2012 — HOUR-0 PREPARATION")
    print("=" * 70)

    outcomes = pd.read_csv(outcomes_path)
    print(f"\nLoaded outcomes: {len(outcomes):,} patients")

    n_censored = int((outcomes["Length_of_stay"] == -1).sum())
    outcomes = outcomes[outcomes["Length_of_stay"] != -1].copy()
    print(f"  Dropped {n_censored} rows with LOS=-1 (censored)")
    print(f"  Remaining: {len(outcomes):,} patients")

    patient_files = sorted(glob.glob(str(set_a_dir / "*.txt")))
    print(f"\nParsing {len(patient_files):,} patient files...")

    rows = []
    for fp in tqdm(patient_files, unit="patient"):
        record_id, static, hour0_df = parse_patient_file(Path(fp))
        row = {"subject_id": record_id}
        row.update(static)
        # Aggregate hour-0 measurements by parameter → median (handles multiple
        # readings within the first hour, robust to outliers)
        if not hour0_df.empty:
            medians = hour0_df.groupby("Parameter")["Value"].median()
            row.update(medians.to_dict())
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n  Pivoted: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Merge outcomes (inner join drops censored / missing patients)
    df = df.merge(
        outcomes[["RecordID", "In-hospital_death", "Length_of_stay"]],
        left_on="subject_id",
        right_on="RecordID",
        how="inner",
    ).drop(columns=["RecordID"])
    df = df.rename(
        columns={
            "In-hospital_death": "hospital_expire_flag",
            "Length_of_stay": "los_icu",
        }
    )
    df["hours_in"] = 0
    print(f"  After outcome merge: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Missing-value handling: median for numerics, mode for Gender/ICUType
    print("\nImputing missing values...")
    feature_cols = [c for c in df.columns if c not in {"subject_id", "hours_in"}]
    missing_before = int(df[feature_cols].isna().sum().sum())
    for col in feature_cols:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    missing_after = int(df[feature_cols].isna().sum().sum())
    print(f"  Missing values: {missing_before:,} → {missing_after:,}")

    # Cast Gender / ICUType / MechVent to int for consistency
    for col in ["Gender", "ICUType", "MechVent"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    print("\nDataset summary:")
    print(f"  Patients: {len(df):,}")
    print(f"  Features (excl. IDs/labels): {len(feature_cols) - 2}")
    print(f"  Mortality rate: {df['hospital_expire_flag'].mean():.3%}")
    print(
        f"  LOS median/mean: {df['los_icu'].median():.1f} / {df['los_icu'].mean():.1f} days"
    )
    print("\nColumns:")
    for c in df.columns:
        print(f"    {c:<25} dtype={df[c].dtype}")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved hour-0 data to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
