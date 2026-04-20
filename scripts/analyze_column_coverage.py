#!/usr/bin/env python3
"""Full-table column coverage and simple distribution stats for a processed CSV.

Streams the file in chunks (same idea as ``fit_autoregressive_preprocessor.py``) so you can
run this on the entire autoregressive (or hour-0) table without loading it into
memory at once.

For each column we report how often values are present (non-missing), plus
lightweight numeric summaries. Optional alignment with ``preprocessor_full.pkl``
marks each column as target / condition / other.

Examples:

    uv run python scripts/analyze_column_coverage.py

    uv run python scripts/analyze_column_coverage.py \\
      --input data/processed/hour0_data.csv \\
      --output results/column_coverage_hour0.csv \\
      --no-preprocessor
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def _count_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _role_for_column(
    col: str,
    target_cols: set[str],
    condition_cols: set[str],
) -> str:
    if col in target_cols and col in condition_cols:
        return "target+condition"
    if col in target_cols:
        return "target"
    if col in condition_cols:
        return "condition"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/autoregressive_data.csv"),
        help="CSV path to scan.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/column_coverage_autoregressive.csv"),
        help="Where to write the per-column report CSV.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows per read_csv chunk.",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=Path("artifacts/preprocessor_full.pkl"),
        help="Pickle with target_cols / condition_cols / all_cols (0 = skip).",
    )
    parser.add_argument(
        "--no-preprocessor",
        action="store_true",
        help="Ignore preprocessor; analyze every column in the CSV.",
    )
    parser.add_argument(
        "--sentinel",
        type=float,
        default=-1.0,
        help="For numeric columns, also report %% of rows equal to this value.",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    target_cols: set[str] = set()
    condition_cols: set[str] = set()
    feature_cols: list[str] | None = None

    if args.no_preprocessor:
        print("Preprocessor ignored: scanning all columns present in the CSV.")
    else:
        pkl = args.preprocessor
        if pkl.exists():
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            target_cols = set(data.get("target_cols", []))
            condition_cols = set(data.get("condition_cols", []))
            feature_cols = list(data.get("all_cols", []))
            print(f"Loaded preprocessor metadata from {pkl}")
            print(f"  all_cols (model inputs): {len(feature_cols)}")
        else:
            print(
                f"Warning: preprocessor not found at {pkl}; scanning all CSV columns."
            )

    total_rows = _count_rows(input_path)
    print(f"Total data rows (excluding header): {total_rows:,}")

    chunk_iter = pd.read_csv(input_path, chunksize=args.chunksize, low_memory=False)

    first = True
    columns: list[str] = []
    non_null = np.array([], dtype=np.int64)
    numeric_mask = np.array([], dtype=bool)
    n_eq_sentinel = np.array([], dtype=np.int64)
    n_numeric_non_null = np.array([], dtype=np.int64)
    sum_x = np.array([], dtype=np.float64)
    sum_x2 = np.array([], dtype=np.float64)
    vmin = np.array([], dtype=np.float64)
    vmax = np.array([], dtype=np.float64)
    n_nonempty_str = np.array([], dtype=np.int64)

    processed = 0
    n_chunks = (total_rows + args.chunksize - 1) // args.chunksize if total_rows else 1

    for chunk in tqdm(chunk_iter, total=n_chunks, desc="Scanning chunks", unit="chunk"):
        if feature_cols is not None:
            missing = [c for c in feature_cols if c not in chunk.columns]
            if missing:
                print(
                    f"Error: preprocessor column missing from CSV (first few): {missing[:5]}",
                    file=sys.stderr,
                )
                sys.exit(1)
            chunk = chunk[feature_cols]

        if first:
            columns = list(chunk.columns)
            n = len(columns)
            non_null = np.zeros(n, dtype=np.int64)
            numeric_mask = np.array(
                [pd.api.types.is_numeric_dtype(chunk[c]) for c in columns],
                dtype=bool,
            )
            n_eq_sentinel = np.zeros(n, dtype=np.int64)
            n_numeric_non_null = np.zeros(n, dtype=np.int64)
            sum_x = np.zeros(n, dtype=np.float64)
            sum_x2 = np.zeros(n, dtype=np.float64)
            vmin = np.full(n, np.inf, dtype=np.float64)
            vmax = np.full(n, -np.inf, dtype=np.float64)
            n_nonempty_str = np.zeros(n, dtype=np.int64)
            first = False
        elif list(chunk.columns) != columns:
            print("Error: chunk column set changed between reads.", file=sys.stderr)
            sys.exit(1)

        for j, col in enumerate(columns):
            s = chunk[col]
            nn = int(s.notna().sum())
            non_null[j] += nn

            if numeric_mask[j]:
                sn = pd.to_numeric(s, errors="coerce")
                valid = sn.notna()
                cnt = int(valid.sum())
                if cnt:
                    n_numeric_non_null[j] += cnt
                    arr = sn[valid].to_numpy(dtype=np.float64, copy=False)
                    sum_x[j] += float(arr.sum())
                    sum_x2[j] += float(np.square(arr).sum())
                    cmin = float(arr.min())
                    cmax = float(arr.max())
                    if cmin < vmin[j]:
                        vmin[j] = cmin
                    if cmax > vmax[j]:
                        vmax[j] = cmax
                    n_eq_sentinel[j] += int(np.sum(arr == args.sentinel))
            else:
                str_nonempty = s.dropna().astype(str).str.strip()
                n_nonempty_str[j] += int((str_nonempty != "").sum())

        processed += len(chunk)

    if first:
        print("Error: no rows read (empty file?).", file=sys.stderr)
        sys.exit(1)

    rows_eff = processed if processed else 1
    out_rows: list[dict[str, object]] = []
    for j, col in enumerate(columns):
        pct_nn = 100.0 * non_null[j] / rows_eff
        role = _role_for_column(col, target_cols, condition_cols)
        row: dict[str, object] = {
            "column": col,
            "role_in_preprocessor": role,
            "dtype_numeric": bool(numeric_mask[j]),
            "n_rows_scanned": rows_eff,
            "n_non_null": int(non_null[j]),
            "pct_non_null": round(pct_nn, 4),
        }
        if numeric_mask[j]:
            nn_num = int(n_numeric_non_null[j])
            row["n_numeric_non_null"] = nn_num
            row["pct_numeric_non_null_of_rows"] = round(100.0 * nn_num / rows_eff, 4)
            row[f"pct_eq_sentinel_{args.sentinel:g}"] = round(
                100.0 * n_eq_sentinel[j] / rows_eff, 4
            )
            if nn_num > 0:
                mean = sum_x[j] / nn_num
                var = max(sum_x2[j] / nn_num - mean * mean, 0.0)
                row["mean_numeric_non_null"] = round(mean, 6)
                row["std_numeric_non_null"] = round(float(np.sqrt(var)), 6)
                row["min_numeric_non_null"] = round(float(vmin[j]), 6)
                row["max_numeric_non_null"] = round(float(vmax[j]), 6)
            else:
                row["mean_numeric_non_null"] = np.nan
                row["std_numeric_non_null"] = np.nan
                row["min_numeric_non_null"] = np.nan
                row["max_numeric_non_null"] = np.nan
        else:
            row["n_nonempty_string"] = int(n_nonempty_str[j])
            row["pct_nonempty_string_of_rows"] = round(
                100.0 * n_nonempty_str[j] / rows_eff, 4
            )

        out_rows.append(row)

    report = pd.DataFrame(out_rows)
    report = report.sort_values("pct_non_null", ascending=True).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(f"\nWrote {len(report)} rows to {args.output}")
    print("\nSparsest columns (lowest pct_non_null):")
    show = report.head(15)[["column", "role_in_preprocessor", "pct_non_null"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
