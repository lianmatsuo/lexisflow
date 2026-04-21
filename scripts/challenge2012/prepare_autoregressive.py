#!/usr/bin/env python3
"""Prepare autoregressive training data from PhysioNet Challenge 2012 set-A.

Secondary reproducibility benchmark alongside MIMIC-III.

Builds hourly-binned trajectories (48 hours/patient), then calls the shared
``prepare_autoregressive_data`` helper to create lag-1 (current, previous)
pairs suitable for the Forest-Flow autoregressive model.

Output:
    data/challenge2012/processed/autoregressive_data.csv
        Columns: subject_id, hours_in, static (Age/Gender/Height/Weight/ICUType),
        dynamic features at current step, lag-1 conditions, plus
        hospital_expire_flag and los_icu labels (duplicated across rows).
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lexisflow.config import get_dataset_config
from lexisflow.data import prepare_autoregressive_data


STATIC_PARAMS = ["Age", "Gender", "Height", "Weight", "ICUType"]
DROP_PARAMS = {"TroponinI", "Cholesterol"}
MISSING_SENTINEL = -1
N_HOURS = 48
CFG = get_dataset_config("challenge2012")


def parse_patient_file(path: Path) -> tuple[int, dict, pd.DataFrame]:
    """Return (record_id, static_dict, hourly_wide_df with 48 rows)."""
    raw = pd.read_csv(path)
    raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce")

    record_id_row = raw[raw["Parameter"] == "RecordID"]
    record_id = int(record_id_row["Value"].iloc[0])

    static = {}
    for p in STATIC_PARAMS:
        row = raw[raw["Parameter"] == p]
        if row.empty:
            static[p] = pd.NA
            continue
        val = row["Value"].iloc[0]
        if pd.notna(val) and val == MISSING_SENTINEL and p in {"Height", "Weight"}:
            val = pd.NA
        static[p] = val

    ts = raw[~raw["Parameter"].isin(STATIC_PARAMS + ["RecordID"])]
    ts = ts[~ts["Parameter"].isin(DROP_PARAMS)]

    # Hour index from HH:MM
    if ts.empty:
        hourly = pd.DataFrame({"hours_in": range(N_HOURS)})
        return record_id, static, hourly

    hour_series = ts["Time"].str.split(":", expand=True)[0].astype(int)
    ts = ts.assign(hour=hour_series)
    ts = ts[ts["hour"] < N_HOURS]

    # Hour × Parameter median, pivot wide
    hourly = (
        ts.groupby(["hour", "Parameter"])["Value"]
        .median()
        .unstack("Parameter")
        .reindex(range(N_HOURS))
        .reset_index()
        .rename(columns={"hour": "hours_in"})
    )

    return record_id, static, hourly


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Challenge 2012 autoregressive training data."
    )
    parser.add_argument("--raw-dir", default="data/challenge2012/raw")
    parser.add_argument(
        "--output-path",
        default="data/challenge2012/processed/autoregressive_data.csv",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=-1.0,
        help="Sentinel value used by prepare_autoregressive_data for "
        "first-timestep lag columns (default: -1.0).",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    set_a_dir = raw_dir / "set-a"
    outcomes_path = raw_dir / "Outcomes-a.txt"

    if not set_a_dir.is_dir() or not outcomes_path.is_file():
        print(f"ERROR: expected {set_a_dir}/ and {outcomes_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("CHALLENGE 2012 — AUTOREGRESSIVE PREPARATION")
    print("=" * 70)

    outcomes = pd.read_csv(outcomes_path)
    outcomes = outcomes[outcomes["Length_of_stay"] != -1].copy()
    valid_ids = set(outcomes["RecordID"].tolist())
    print(f"\nLoaded {len(outcomes):,} patients with valid LOS")

    patient_files = sorted(glob.glob(str(set_a_dir / "*.txt")))
    print(f"\nParsing {len(patient_files):,} patient files into hourly grid...")

    per_patient_frames = []
    static_rows = []
    for fp in tqdm(patient_files, unit="patient"):
        record_id, static, hourly = parse_patient_file(Path(fp))
        if record_id not in valid_ids:
            continue
        hourly["subject_id"] = record_id
        per_patient_frames.append(hourly)
        static_rows.append({"subject_id": record_id, **static})

    long_df = pd.concat(per_patient_frames, ignore_index=True)
    static_df = pd.DataFrame(static_rows)
    print(
        f"\n  Hourly long-frame: {long_df.shape[0]:,} rows × {long_df.shape[1]} columns"
    )
    print(f"  Static frame: {static_df.shape[0]:,} rows × {static_df.shape[1]} columns")

    # Impute static: median for numerics, mode for Gender/ICUType
    for col in STATIC_PARAMS:
        if col not in static_df.columns:
            continue
        if static_df[col].isna().any():
            if col in {"Gender", "ICUType"}:
                mode = static_df[col].mode()
                static_df[col] = static_df[col].fillna(
                    mode.iloc[0] if not mode.empty else 0
                )
            else:
                static_df[col] = static_df[col].fillna(static_df[col].median())
    for col in ("Gender", "ICUType"):
        if col in static_df.columns:
            static_df[col] = static_df[col].astype(int)

    # Merge static onto long frame (duplicated across hours)
    long_df = long_df.merge(static_df, on="subject_id", how="left")

    # Impute dynamic columns: forward-fill within patient → back-fill → global median
    dynamic_cols = [
        c
        for c in long_df.columns
        if c not in {"subject_id", "hours_in", *STATIC_PARAMS}
    ]
    print(
        f"\n  Dynamic columns ({len(dynamic_cols)}): imputing "
        "ffill→bfill within patient, then global median..."
    )
    long_df = long_df.sort_values(["subject_id", "hours_in"]).reset_index(drop=True)
    long_df[dynamic_cols] = long_df.groupby("subject_id")[dynamic_cols].ffill().bfill()
    # Patients with no measurements at all for a column → fill with global median
    for col in dynamic_cols:
        if long_df[col].isna().any():
            long_df[col] = long_df[col].fillna(long_df[col].median())

    # Cast MechVent to int (binary flag)
    if "MechVent" in long_df.columns:
        long_df["MechVent"] = long_df["MechVent"].round().clip(0, 1).astype(int)

    missing = int(long_df.isna().sum().sum())
    print(f"  Remaining missing values: {missing}")

    # Build autoregressive pairs via shared helper
    print("\n  Building lag-1 autoregressive pairs...")
    ar_df, target_cols, condition_cols = prepare_autoregressive_data(
        df=long_df,
        id_col="subject_id",
        time_col="hours_in",
        static_cols=STATIC_PARAMS,
        lag=1,
        fill_strategy="special",
        fill_value=args.fill_value,
        keep_biophysical_mean_only=False,
    )

    # Attach labels (duplicated across timesteps — training filters as needed)
    label_map = outcomes.rename(
        columns={
            "RecordID": "subject_id",
            "In-hospital_death": "hospital_expire_flag",
            "Length_of_stay": "los_icu",
        }
    )[["subject_id", "hospital_expire_flag", "los_icu"]]
    ar_df = ar_df.merge(label_map, on="subject_id", how="left")

    print("\nDataset summary:")
    print(f"  Rows: {ar_df.shape[0]:,} (patients × ~48 hours)")
    print(f"  Columns total: {ar_df.shape[1]}")
    print(f"  Target columns ({len(target_cols)}): {target_cols[:5]}...")
    print(f"  Condition columns ({len(condition_cols)}): {condition_cols[:5]}...")
    print(f"  Patients: {ar_df['subject_id'].nunique():,}")

    # Patient-disjoint 80/10/10 train/test/holdout split (mirrors MIMIC
    # prepare_autoregressive.py). The primary output path receives train-only
    # rows so quality/privacy metrics computed against the test/holdout CSVs
    # cannot overlap training data.
    print("\nPatient-level 80/10/10 train/test/holdout split...")
    subjects = ar_df["subject_id"].unique()
    rng_split = np.random.default_rng(CFG.split.shuffle_seed)
    rng_split.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, int(CFG.split.test_fraction * n))
    n_holdout = max(1, int(CFG.split.holdout_fraction * n))
    test_subjects = set(subjects[:n_test].tolist())
    holdout_subjects = set(subjects[n_test : n_test + n_holdout].tolist())
    train_subjects = set(subjects[n_test + n_holdout :].tolist())

    df_train = ar_df[ar_df["subject_id"].isin(train_subjects)]
    df_test = ar_df[ar_df["subject_id"].isin(test_subjects)]
    df_holdout = ar_df[ar_df["subject_id"].isin(holdout_subjects)]

    print(
        f"  Patients: {len(train_subjects):,} train / "
        f"{len(test_subjects):,} test / {len(holdout_subjects):,} holdout"
    )
    print(
        f"  Rows:     {len(df_train):,} train / "
        f"{len(df_test):,} test / {len(df_holdout):,} holdout"
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    real_test_path = out_path.parent / "real_test.csv"
    real_holdout_path = out_path.parent / "real_holdout.csv"

    df_train.to_csv(out_path, index=False)
    df_test.to_csv(real_test_path, index=False)
    df_holdout.to_csv(real_holdout_path, index=False)
    print(f"\n✓ Saved train split to     {out_path}")
    print(f"✓ Saved test split to      {real_test_path}")
    print(f"✓ Saved holdout split to   {real_holdout_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
