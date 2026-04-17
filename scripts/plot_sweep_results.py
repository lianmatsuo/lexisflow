#!/usr/bin/env python3
"""Visualize hyperparameter sweep results.

Generates heatmaps and line plots to analyze the effect of nt and n_noise
on model performance (ROC-AUC, F1, accuracy from TSTR), plus training time,
quality metrics, uncertainty intervals, and optional extrapolation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit


def load_results(results_path: Path) -> pd.DataFrame:
    """Load sweep results from CSV."""
    df = pd.read_csv(results_path)
    # Drop rows with NaN in key metrics
    df = df.dropna(subset=["synth_roc_auc", "real_roc_auc"])
    # run_sweep writes total_train_time_sec; older CSVs used train_time_sec
    if "train_time_sec" not in df.columns and "total_train_time_sec" in df.columns:
        df = df.copy()
        df["train_time_sec"] = df["total_train_time_sec"]
    return df


def plot_quality_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot quality metrics (KS, correlation, range violations) as heatmaps."""
    # Check if quality metrics exist
    quality_cols = ["avg_ks_stat", "corr_frobenius", "range_violation_pct"]
    if not all(col in df.columns for col in quality_cols):
        print("Quality metrics not found in results, skipping quality plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Average KS statistic (lower is better)
    pivot_ks = df.pivot(index="n_noise", columns="nt", values="avg_ks_stat")
    sns.heatmap(
        pivot_ks,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",  # Reverse: red=high (bad), green=low (good)
        ax=axes[0],
        cbar_kws={"label": "Avg KS Statistic"},
    )
    axes[0].set_title(
        "Distribution Similarity (KS)\n(Lower = Better)", fontweight="bold"
    )
    axes[0].set_xlabel("nt (time levels)")
    axes[0].set_ylabel("n_noise (noise samples)")

    # 2. Correlation Frobenius norm (lower is better)
    pivot_corr = df.pivot(index="n_noise", columns="nt", values="corr_frobenius")
    sns.heatmap(
        pivot_corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        ax=axes[1],
        cbar_kws={"label": "Frobenius Norm"},
    )
    axes[1].set_title("Correlation Preservation\n(Lower = Better)", fontweight="bold")
    axes[1].set_xlabel("nt (time levels)")
    axes[1].set_ylabel("n_noise (noise samples)")

    # 3. Range violations percentage (lower is better)
    pivot_range = df.pivot(index="n_noise", columns="nt", values="range_violation_pct")
    sns.heatmap(
        pivot_range,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        ax=axes[2],
        cbar_kws={"label": "Violations (%)"},
    )
    axes[2].set_title("Clinical Range Violations\n(Lower = Better)", fontweight="bold")
    axes[2].set_xlabel("nt (time levels)")
    axes[2].set_ylabel("n_noise (noise samples)")

    plt.tight_layout()
    output_path = output_dir / "sweep_quality_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved quality metrics heatmap to {output_path}")
    plt.close()


def _plot_tstr_heatmap_pair(
    df: pd.DataFrame,
    output_dir: Path,
    synth_col: str,
    real_col: str,
    filename: str,
    cbar_label: str,
    left_title: str,
    vmin: float,
    vmax: float,
) -> None:
    """Heatmap of synthetic metric and (synth - real) gap."""
    pivot = df.pivot(index="n_noise", columns="nt", values=synth_col)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        ax=axes[0],
        cbar_kws={"label": cbar_label},
    )
    axes[0].set_title(left_title)
    axes[0].set_xlabel("nt (time levels)")
    axes[0].set_ylabel("n_noise (noise samples)")

    pivot_gap = df.pivot(index="n_noise", columns="nt", values=synth_col) - df.pivot(
        index="n_noise", columns="nt", values=real_col
    )

    sns.heatmap(
        pivot_gap,
        annot=True,
        fmt="+.3f",
        cmap="RdYlGn",
        center=0,
        ax=axes[1],
        cbar_kws={"label": "Gap (Synth - Real)"},
    )
    axes[1].set_title("Performance Gap vs Real Model")
    axes[1].set_xlabel("nt (time levels)")
    axes[1].set_ylabel("n_noise (noise samples)")

    plt.tight_layout()
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _plot_tstr_line_charts(
    df: pd.DataFrame,
    output_dir: Path,
    synth_col: str,
    real_col: str,
    filename: str,
    ylabel: str,
    ci_col: str | None = None,
) -> None:
    """Line plots vs nt and n_noise with mean real baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    real_baseline = df[real_col].mean()

    for n_noise in sorted(df["n_noise"].unique()):
        subset = df[df["n_noise"] == n_noise].sort_values("nt")
        if ci_col is not None and ci_col in subset.columns:
            yerr = pd.to_numeric(subset[ci_col], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(yerr)
            if finite.any():
                yerr = np.where(finite, yerr, 0.0)
                axes[0].errorbar(
                    subset["nt"],
                    subset[synth_col],
                    yerr=yerr,
                    marker="o",
                    capsize=3,
                    label=f"n_noise={n_noise}",
                )
            else:
                axes[0].plot(
                    subset["nt"],
                    subset[synth_col],
                    marker="o",
                    label=f"n_noise={n_noise}",
                )
        else:
            axes[0].plot(
                subset["nt"],
                subset[synth_col],
                marker="o",
                label=f"n_noise={n_noise}",
            )

    axes[0].axhline(y=real_baseline, color="black", linestyle="--", label="Real Model")
    axes[0].set_xlabel("nt (time levels)")
    axes[0].set_ylabel(ylabel)
    axes[0].set_title("Effect of Time Levels (nt)")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[0].grid(True, alpha=0.3)

    for nt in sorted(df["nt"].unique()):
        subset = df[df["nt"] == nt].sort_values("n_noise")
        if ci_col is not None and ci_col in subset.columns:
            yerr = pd.to_numeric(subset[ci_col], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(yerr)
            if finite.any():
                yerr = np.where(finite, yerr, 0.0)
                axes[1].errorbar(
                    subset["n_noise"],
                    subset[synth_col],
                    yerr=yerr,
                    marker="s",
                    capsize=3,
                    label=f"nt={nt}",
                )
            else:
                axes[1].plot(
                    subset["n_noise"],
                    subset[synth_col],
                    marker="s",
                    label=f"nt={nt}",
                )
        else:
            axes[1].plot(
                subset["n_noise"],
                subset[synth_col],
                marker="s",
                label=f"nt={nt}",
            )

    axes[1].axhline(y=real_baseline, color="black", linestyle="--", label="Real Model")
    axes[1].set_xlabel("n_noise (noise samples)")
    axes[1].set_ylabel(ylabel)
    axes[1].set_title("Effect of Noise Samples (n_noise)")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create heatmap of ROC-AUC vs (nt, n_noise)."""
    _plot_tstr_heatmap_pair(
        df,
        output_dir,
        "synth_roc_auc",
        "real_roc_auc",
        "sweep_heatmap.png",
        "ROC-AUC",
        "Synthetic Model ROC-AUC (TSTR)",
        vmin=0.5,
        vmax=1.0,
    )


def plot_line_charts(df: pd.DataFrame, output_dir: Path):
    """Create line plots showing effect of each parameter (ROC-AUC)."""
    _plot_tstr_line_charts(
        df,
        output_dir,
        "synth_roc_auc",
        "real_roc_auc",
        "sweep_line_charts.png",
        "ROC-AUC",
        ci_col="synth_roc_auc_ci95",
    )


def plot_f1_charts(df: pd.DataFrame, output_dir: Path):
    """F1 score heatmap and line charts (mortality TSTR)."""
    need = ("synth_f1", "real_f1")
    if not all(c in df.columns for c in need):
        print("  Skipping F1 plots (synth_f1 / real_f1 not in CSV)")
        return
    sub = df.dropna(subset=list(need))
    if len(sub) < 2:
        print("  Skipping F1 plots (insufficient non-null F1 rows)")
        return
    _plot_tstr_heatmap_pair(
        sub,
        output_dir,
        "synth_f1",
        "real_f1",
        "sweep_f1_heatmap.png",
        "F1",
        "Synthetic Model F1 (TSTR)",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_tstr_line_charts(
        sub,
        output_dir,
        "synth_f1",
        "real_f1",
        "sweep_f1_line_charts.png",
        "F1",
        ci_col="synth_f1_ci95",
    )


def plot_accuracy_charts(df: pd.DataFrame, output_dir: Path):
    """Accuracy heatmap and line charts (mortality TSTR)."""
    need = ("synth_accuracy", "real_accuracy")
    if not all(c in df.columns for c in need):
        print("  Skipping accuracy plots (synth_accuracy / real_accuracy not in CSV)")
        return
    sub = df.dropna(subset=list(need))
    if len(sub) < 2:
        print("  Skipping accuracy plots (insufficient non-null accuracy rows)")
        return
    _plot_tstr_heatmap_pair(
        sub,
        output_dir,
        "synth_accuracy",
        "real_accuracy",
        "sweep_accuracy_heatmap.png",
        "Accuracy",
        "Synthetic Model Accuracy (TSTR)",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_tstr_line_charts(
        sub,
        output_dir,
        "synth_accuracy",
        "real_accuracy",
        "sweep_accuracy_line_charts.png",
        "Accuracy",
        ci_col="synth_accuracy_ci95",
    )


def plot_balanced_accuracy_charts(df: pd.DataFrame, output_dir: Path):
    """Balanced-accuracy heatmap and line charts (mortality TSTR)."""
    need = ("synth_balanced_accuracy", "real_balanced_accuracy")
    if not all(c in df.columns for c in need):
        print(
            "  Skipping balanced-accuracy plots "
            "(synth_balanced_accuracy / real_balanced_accuracy not in CSV)"
        )
        return
    sub = df.dropna(subset=list(need))
    if len(sub) < 2:
        print("  Skipping balanced-accuracy plots (insufficient non-null rows)")
        return
    _plot_tstr_heatmap_pair(
        sub,
        output_dir,
        "synth_balanced_accuracy",
        "real_balanced_accuracy",
        "sweep_balanced_accuracy_heatmap.png",
        "Balanced Accuracy",
        "Synthetic Model Balanced Accuracy (TSTR)",
        vmin=0.0,
        vmax=1.0,
    )
    _plot_tstr_line_charts(
        sub,
        output_dir,
        "synth_balanced_accuracy",
        "real_balanced_accuracy",
        "sweep_balanced_accuracy_line_charts.png",
        "Balanced Accuracy",
        ci_col="synth_balanced_accuracy_ci95",
    )


def plot_uncertainty_heatmaps(df: pd.DataFrame, output_dir: Path):
    """Plot CI95 heatmaps for key utility/quality/privacy metrics when available."""
    metric_specs = [
        ("synth_roc_auc_ci95", "Mortality ROC-AUC CI95"),
        ("los_synth_macro_f1_ci95", "LOS Macro-F1 CI95"),
        ("avg_ks_stat_ci95", "Avg KS CI95"),
        ("range_violation_pct_ci95", "Range Violation % CI95"),
        ("dcr_median_ci95", "DCR Median CI95"),
        ("mia_roc_auc_ci95", "MIA ROC-AUC CI95"),
    ]
    available = [
        (col, label)
        for col, label in metric_specs
        if col in df.columns and df[col].notna().any()
    ]
    if not available:
        print("  Skipping uncertainty heatmaps (no CI95 columns found)")
        return

    n_cols = 3
    n_rows = int(np.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (col, label) in zip(axes, available):
        pivot = df.pivot(index="n_noise", columns="nt", values=col)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": label},
        )
        ax.set_title(f"{label}\n(Lower = tighter uncertainty)")
        ax.set_xlabel("nt (time levels)")
        ax.set_ylabel("n_noise (noise samples)")

    for ax in axes[len(available) :]:
        ax.axis("off")

    plt.tight_layout()
    output_path = output_dir / "sweep_uncertainty_heatmaps.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_training_time(df: pd.DataFrame, output_dir: Path):
    """Plot training time vs parameters."""
    if "train_time_sec" not in df.columns:
        print(
            "  Skipping training time plot (no train_time_sec / total_train_time_sec)"
        )
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Time vs nt
    for n_noise in sorted(df["n_noise"].unique()):
        subset = df[df["n_noise"] == n_noise].sort_values("nt")
        axes[0].plot(
            subset["nt"],
            subset["train_time_sec"] / 60,  # Convert to minutes
            marker="o",
            label=f"n_noise={n_noise}",
        )

    axes[0].set_xlabel("nt (time levels)")
    axes[0].set_ylabel("Training Time (minutes)")
    axes[0].set_title("Training Time vs nt")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Time vs n_noise
    for nt in sorted(df["nt"].unique()):
        subset = df[df["nt"] == nt].sort_values("n_noise")
        axes[1].plot(
            subset["n_noise"],
            subset["train_time_sec"] / 60,
            marker="s",
            label=f"nt={nt}",
        )

    axes[1].set_xlabel("n_noise (noise samples)")
    axes[1].set_ylabel("Training Time (minutes)")
    axes[1].set_title("Training Time vs n_noise")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "sweep_training_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'sweep_training_time.png'}")


def log_fit(x, a, b):
    """Logarithmic fit function: y = a * log(x) + b"""
    return a * np.log(x) + b


def plot_extrapolation(df: pd.DataFrame, output_dir: Path):
    """Fit curves and extrapolate to higher values."""
    extrap_nt_max = 30
    extrap_n_noise_max = 30

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Aggregate by averaging across the other parameter
    nt_agg = df.groupby("nt")["synth_roc_auc"].mean().reset_index()
    noise_agg = df.groupby("n_noise")["synth_roc_auc"].mean().reset_index()

    real_baseline = df["real_roc_auc"].mean()

    # Extrapolate nt
    x_nt = nt_agg["nt"].values
    y_nt = nt_agg["synth_roc_auc"].values

    try:
        popt_nt, _ = curve_fit(log_fit, x_nt, y_nt, maxfev=1000)
        x0_nt = max(float(np.min(x_nt)), 1.0)
        x_extrap_nt = np.linspace(x0_nt, extrap_nt_max, 80)
        y_extrap_nt = log_fit(x_extrap_nt, *popt_nt)

        axes[0].scatter(x_nt, y_nt, s=100, zorder=3, label="Measured")
        axes[0].plot(x_extrap_nt, y_extrap_nt, "--", color="red", label="Extrapolated")
        axes[0].axhline(
            y=real_baseline, color="green", linestyle=":", label="Real Model"
        )

        y_prod_nt = log_fit(extrap_nt_max, *popt_nt)
        axes[0].axvline(x=extrap_nt_max, color="orange", linestyle="--", alpha=0.5)
        axes[0].scatter(
            [extrap_nt_max],
            [y_prod_nt],
            s=150,
            marker="*",
            color="orange",
            zorder=4,
            label=f"nt={extrap_nt_max}: {y_prod_nt:.3f}",
        )

        axes[0].set_xlabel("nt (time levels)")
        axes[0].set_ylabel("ROC-AUC (averaged)")
        axes[0].set_title("Extrapolation: nt Effect")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, extrap_nt_max + 2)
    except Exception as e:
        axes[0].text(
            0.5, 0.5, f"Fit failed: {e}", transform=axes[0].transAxes, ha="center"
        )

    # Extrapolate n_noise
    x_noise = noise_agg["n_noise"].values
    y_noise = noise_agg["synth_roc_auc"].values

    try:
        popt_noise, _ = curve_fit(log_fit, x_noise, y_noise, maxfev=1000)
        x0_noise = max(float(np.min(x_noise)), 1.0)
        x_extrap_noise = np.linspace(x0_noise, extrap_n_noise_max, 80)
        y_extrap_noise = log_fit(x_extrap_noise, *popt_noise)

        axes[1].scatter(x_noise, y_noise, s=100, zorder=3, label="Measured")
        axes[1].plot(
            x_extrap_noise, y_extrap_noise, "--", color="red", label="Extrapolated"
        )
        axes[1].axhline(
            y=real_baseline, color="green", linestyle=":", label="Real Model"
        )

        y_prod_noise = log_fit(extrap_n_noise_max, *popt_noise)
        axes[1].axvline(x=extrap_n_noise_max, color="orange", linestyle="--", alpha=0.5)
        axes[1].scatter(
            [extrap_n_noise_max],
            [y_prod_noise],
            s=150,
            marker="*",
            color="orange",
            zorder=4,
            label=f"n_noise={extrap_n_noise_max}: {y_prod_noise:.3f}",
        )

        axes[1].set_xlabel("n_noise (noise samples)")
        axes[1].set_ylabel("ROC-AUC (averaged)")
        axes[1].set_title("Extrapolation: n_noise Effect")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, extrap_n_noise_max + 2)
    except Exception as e:
        axes[1].text(
            0.5, 0.5, f"Fit failed: {e}", transform=axes[1].transAxes, ha="center"
        )

    plt.tight_layout()
    plt.savefig(output_dir / "sweep_extrapolation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'sweep_extrapolation.png'}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)

    print(f"\nRuns completed: {len(df)}")
    print(f"nt range: {df['nt'].min()} - {df['nt'].max()}")
    print(f"n_noise range: {df['n_noise'].min()} - {df['n_noise'].max()}")

    print("\nBest configuration (by synth_roc_auc):")
    best = df.loc[df["synth_roc_auc"].idxmax()]
    print(f"  nt={int(best['nt'])}, n_noise={int(best['n_noise'])}")
    if "synth_roc_auc_ci95" in df.columns and pd.notna(best.get("synth_roc_auc_ci95")):
        print(
            f"  Synth ROC-AUC: {best['synth_roc_auc']:.4f} ± {best['synth_roc_auc_ci95']:.4f} (95% CI half-width)"
        )
    else:
        print(f"  Synth ROC-AUC: {best['synth_roc_auc']:.4f}")
    print(f"  Real ROC-AUC:  {best['real_roc_auc']:.4f}")
    print(f"  Gap: {best['synth_roc_auc'] - best['real_roc_auc']:+.4f}")
    if "synth_roc_auc_ci95" in df.columns:
        ci = pd.to_numeric(df["synth_roc_auc_ci95"], errors="coerce")
        mean = pd.to_numeric(df["synth_roc_auc"], errors="coerce")
        best_low = best["synth_roc_auc"] - best.get("synth_roc_auc_ci95", np.nan)
        best_high = best["synth_roc_auc"] + best.get("synth_roc_auc_ci95", np.nan)
        if np.isfinite(best_low) and np.isfinite(best_high):
            overlap_mask = (mean - ci <= best_high) & (mean + ci >= best_low)
            overlap_count = int(max(overlap_mask.sum() - 1, 0))
            if overlap_count > 0:
                print(
                    f"  Note: {overlap_count} other configuration(s) have overlapping ROC-AUC CIs with this point."
                )
    if "avg_ks_stat" in df.columns:
        print(
            f"  Quality: KS={best['avg_ks_stat']:.4f}, Corr={best['corr_frobenius']:.4f}, Range={best['range_violation_pct']:.2f}%"
        )

    print("\nSmallest gap from real model:")
    df["gap"] = abs(df["synth_roc_auc"] - df["real_roc_auc"])
    closest = df.loc[df["gap"].idxmin()]
    print(f"  nt={int(closest['nt'])}, n_noise={int(closest['n_noise'])}")
    print(f"  Synth ROC-AUC: {closest['synth_roc_auc']:.4f}")
    print(f"  Real ROC-AUC:  {closest['real_roc_auc']:.4f}")
    print(f"  Gap: {closest['synth_roc_auc'] - closest['real_roc_auc']:+.4f}")

    if "train_time_sec" in df.columns:
        print("\nAverage training time:")
        print(f"  {df['train_time_sec'].mean() / 60:.1f} minutes per run")

    print("\n" + "=" * 60)


def main():
    print("=" * 60)
    print("Sweep Results Visualization")
    print("=" * 60)

    results_path = Path("results/sweep_results.csv")
    output_dir = Path("results/sweep_plots")

    if not results_path.exists():
        print(f"\nError: {results_path} not found.")
        print("Run run_sweep.py first to generate results.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading results from {results_path}...")
    df = load_results(results_path)
    print(f"  Loaded {len(df)} valid results")

    if len(df) < 2:
        print("\nNot enough data to plot. Run more sweep configurations.")
        sys.exit(1)

    # Generate plots
    print("\nGenerating plots...")
    plot_heatmap(df, output_dir)
    plot_line_charts(df, output_dir)
    plot_f1_charts(df, output_dir)
    plot_accuracy_charts(df, output_dir)
    plot_balanced_accuracy_charts(df, output_dir)
    plot_training_time(df, output_dir)
    plot_quality_metrics(df, output_dir)
    plot_uncertainty_heatmaps(df, output_dir)

    if len(df) >= 4:
        plot_extrapolation(df, output_dir)
    else:
        print("  Skipping extrapolation (need at least 4 data points)")

    # Print summary
    print_summary(df)

    print(f"\nAll plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
