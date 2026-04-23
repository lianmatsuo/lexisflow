#!/usr/bin/env python3
"""Visualize hour-0 diagnostics CSV as heatmaps and line charts.

This script is a compact companion to ``scripts/common/analyze_sweep.py`` for
the hour-0-only diagnostics report produced by
``scripts/common/evaluate_hour0_models.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS = Path("results/hour0_diagnostics.csv")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize hour-0 diagnostics CSV as plots."
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Hour-0 diagnostics CSV (default: {DEFAULT_RESULTS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated plots (default: <results_stem>_plots).",
    )
    return parser


def load_results(path: Path) -> pd.DataFrame:
    """Load diagnostics CSV and coerce numeric fields."""
    df = pd.read_csv(path)
    for col in df.columns:
        if col in {"model_path", "model_type", "synth_seeds"}:
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except (TypeError, ValueError):
            # Keep non-numeric columns as-is.
            pass
    if "nt" in df.columns:
        df["nt"] = pd.to_numeric(df["nt"], errors="coerce")
    if "n_noise" in df.columns:
        df["n_noise"] = pd.to_numeric(df["n_noise"], errors="coerce")
    return df


def _plot_metric_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    specs: list[tuple[str, str]] = [
        ("avg_ks_stat", "Average KS (Lower = Better)"),
        ("corr_frobenius", "Correlation Frobenius Gap (Lower = Better)"),
        ("wasserstein_scaled_mean", "Scaled Wasserstein Mean (Lower = Better)"),
        ("range_violation_pct", "Range Violation % (Lower = Better)"),
        ("categorical_tv_mean", "Categorical TV Mean (Lower = Better)"),
    ]
    available = [(col, title) for col, title in specs if col in df.columns]
    if not available:
        print("No core metric columns found for heatmaps.")
        return

    n = len(available)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (col, title) in zip(axes, available):
        pivot = df.pivot(index="n_noise", columns="nt", values=col)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            ax=ax,
            cbar_kws={"label": col},
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("nt")
        ax.set_ylabel("n_noise")

    for ax in axes[len(available) :]:
        ax.axis("off")

    plt.tight_layout()
    out = output_dir / "hour0_core_metric_heatmaps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _plot_metric_line_charts(df: pd.DataFrame, output_dir: Path) -> None:
    specs: list[tuple[str, str]] = [
        ("avg_ks_stat", "Average KS"),
        ("corr_frobenius", "Corr Frobenius"),
        ("wasserstein_scaled_mean", "Scaled Wasserstein"),
    ]
    available = [(col, label) for col, label in specs if col in df.columns]
    if not available:
        print("No line-chart metric columns found.")
        return

    for col, label in available:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ci_col = f"{col}_ci95" if f"{col}_ci95" in df.columns else None

        for n_noise in sorted(df["n_noise"].dropna().unique()):
            sub = df[df["n_noise"] == n_noise].sort_values("nt")
            yerr = (
                pd.to_numeric(sub[ci_col], errors="coerce")
                if ci_col is not None
                else None
            )
            axes[0].errorbar(
                sub["nt"],
                sub[col],
                yerr=yerr,
                marker="o",
                capsize=3 if ci_col is not None else 0,
                label=f"n_noise={int(n_noise)}",
            )

        axes[0].set_title(f"{label} vs nt")
        axes[0].set_xlabel("nt")
        axes[0].set_ylabel(label)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        for nt in sorted(df["nt"].dropna().unique()):
            sub = df[df["nt"] == nt].sort_values("n_noise")
            yerr = (
                pd.to_numeric(sub[ci_col], errors="coerce")
                if ci_col is not None
                else None
            )
            axes[1].errorbar(
                sub["n_noise"],
                sub[col],
                yerr=yerr,
                marker="s",
                capsize=3 if ci_col is not None else 0,
                label=f"nt={int(nt)}",
            )

        axes[1].set_title(f"{label} vs n_noise")
        axes[1].set_xlabel("n_noise")
        axes[1].set_ylabel(label)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        plt.tight_layout()
        out = output_dir / f"hour0_{col}_line_charts.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def _plot_uncertainty_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    specs: list[tuple[str, str]] = [
        ("avg_ks_stat_ci95", "Average KS CI95"),
        ("corr_frobenius_ci95", "Corr Frobenius CI95"),
        ("wasserstein_scaled_mean_ci95", "Scaled Wasserstein CI95"),
        ("range_violation_pct_ci95", "Range Violation % CI95"),
        ("categorical_tv_mean_ci95", "Categorical TV Mean CI95"),
    ]
    available = [(col, label) for col, label in specs if col in df.columns]
    if not available:
        print("No CI95 columns found for uncertainty heatmaps.")
        return

    n = len(available)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (col, label) in zip(axes, available):
        pivot = df.pivot(index="n_noise", columns="nt", values=col)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": label},
        )
        ax.set_title(f"{label} (Lower = Tighter)", fontweight="bold")
        ax.set_xlabel("nt")
        ax.set_ylabel("n_noise")

    for ax in axes[len(available) :]:
        ax.axis("off")

    plt.tight_layout()
    out = output_dir / "hour0_uncertainty_heatmaps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("HOUR-0 DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Rows: {len(df)}")
    print(
        "Grid: "
        f"nt [{int(df['nt'].min())}, {int(df['nt'].max())}] | "
        f"n_noise [{int(df['n_noise'].min())}, {int(df['n_noise'].max())}]"
    )
    if "seed_count" in df.columns:
        print(f"Synthetic seed count per model: {int(df['seed_count'].iloc[0])}")

    rank_cols = [
        c for c in ["avg_ks_stat", "wasserstein_scaled_mean"] if c in df.columns
    ]
    if not rank_cols:
        print("No rank columns found.")
        return

    ranked = df.sort_values(rank_cols, ascending=True).reset_index(drop=True)
    best = ranked.iloc[0]
    worst = ranked.iloc[-1]

    print("\nBest configuration (lower is better):")
    print(
        f"  nt={int(best['nt'])}, n_noise={int(best['n_noise'])} | "
        f"KS={best.get('avg_ks_stat', np.nan):.4f} | "
        f"Wass(z)={best.get('wasserstein_scaled_mean', np.nan):.4f} | "
        f"CorrF={best.get('corr_frobenius', np.nan):.4f}"
    )

    print("\nWorst configuration:")
    print(
        f"  nt={int(worst['nt'])}, n_noise={int(worst['n_noise'])} | "
        f"KS={worst.get('avg_ks_stat', np.nan):.4f} | "
        f"Wass(z)={worst.get('wasserstein_scaled_mean', np.nan):.4f} | "
        f"CorrF={worst.get('corr_frobenius', np.nan):.4f}"
    )

    show_cols = [
        c
        for c in [
            "nt",
            "n_noise",
            "avg_ks_stat",
            "avg_ks_stat_ci95",
            "wasserstein_scaled_mean",
            "wasserstein_scaled_mean_ci95",
            "corr_frobenius",
            "range_violation_pct",
        ]
        if c in ranked.columns
    ]
    print("\nTop 10:")
    print(ranked[show_cols].head(10).to_string(index=False))
    print("=" * 70)


def main() -> None:
    args = _build_argparser().parse_args()
    results_path = args.results_path
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = results_path.parent / f"{results_path.stem}_plots"

    print("=" * 70)
    print("Hour-0 Diagnostics Visualization")
    print("=" * 70)
    print(f"Results CSV: {results_path}")

    if not results_path.exists():
        print(f"\nError: {results_path} not found.")
        print("Run evaluate_hour0_models.py first.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(results_path)
    print(f"Loaded rows: {len(df)}")
    if len(df) < 2:
        print("Not enough rows to plot.")
        sys.exit(1)

    print("\nGenerating plots...")
    _plot_metric_heatmaps(df, output_dir)
    _plot_metric_line_charts(df, output_dir)
    _plot_uncertainty_heatmaps(df, output_dir)
    print_summary(df)
    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
