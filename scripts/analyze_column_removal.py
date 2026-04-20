#!/usr/bin/env python3
"""Build results/column_removal_recommendations.csv from column_coverage_*.csv.

Tiers use **current-timestep target** ``pct_non_null`` (not the lag column, which is
always dense because of ``-1`` fill). Each row lists the paired ``*_lag1`` column
to drop together when trimming the autoregressive schema.

Usage:
    uv run python scripts/analyze_column_removal.py
    uv run python scripts/analyze_column_removal.py \\
      --coverage-csv results/column_coverage_autoregressive.csv \\
      --output results/column_removal_recommendations.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--coverage-csv",
        type=Path,
        default=Path("results/column_coverage_autoregressive.csv"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/column_removal_recommendations.csv"),
    )
    args = p.parse_args()

    cov = pd.read_csv(args.coverage_csv)
    cov["column"] = cov["column"].astype(str)
    by_col = cov.set_index("column")

    rows_out: list[dict[str, object]] = []
    seen: set[str] = set()

    for _, r in cov.iterrows():
        col = str(r["column"])
        role = str(r["role_in_preprocessor"])
        pct = float(r["pct_non_null"])
        if role != "target" or col in seen:
            continue
        seen.add(col)

        lag = f"{col}_lag1"
        has_lag = lag in by_col.index
        lag_pct = float(by_col.loc[lag, "pct_non_null"]) if has_lag else float("nan")
        raw_sent = by_col.loc[lag, "pct_eq_sentinel_-1"] if has_lag else float("nan")
        lag_sent = float(raw_sent) if has_lag and pd.notna(raw_sent) else float("nan")

        if pct < 1.0:
            rec, rationale = (
                "REMOVE",
                "Target pct_non_null <1% on hourly rows — rare alternate specimen "
                "or niche lab; weak generative signal.",
            )
        elif pct < 5.0:
            rec, rationale = (
                "REMOVE_IF_PANEL",
                "Target pct_non_null 1–5%: good candidate to drop for a tighter "
                "ICU core panel unless you need this analyte.",
            )
        elif pct < 15.0:
            rec, rationale = (
                "REVIEW",
                "Target pct_non_null 5–15%: optional trim; keep if clinically "
                "central for your story.",
            )
        else:
            continue

        rows_out.append(
            {
                "target_column": col,
                "paired_lag_column": lag if has_lag else "",
                "role_target": role,
                "pct_non_null_target": round(pct, 4),
                "pct_non_null_lag": round(lag_pct, 4) if has_lag else "",
                "pct_lag_rows_sentinel_minus1": round(lag_sent, 2)
                if has_lag and lag_sent == lag_sent
                else "",
                "recommendation": rec,
                "rationale": rationale,
                "action_note": (
                    "Drop target and paired *_lag1 from flat table + re-run 01b "
                    "preprocessor so AR stays consistent."
                    if has_lag
                    else "Drop from flat table + re-run 01b; no matching *_lag1.",
                ),
            }
        )

    out = pd.DataFrame(rows_out)
    out = out.sort_values(
        ["recommendation", "pct_non_null_target"],
        ascending=[True, True],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out["recommendation"].value_counts().to_string())
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
