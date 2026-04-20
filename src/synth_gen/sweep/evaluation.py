"""TSTR + quality + privacy + trajectory evaluation for a sweep cell.

One ``evaluate_tstr`` call per (trajectory-sampling seed, sweep cell) produces
the full per-seed metric bundle consumed by ``metrics.average_sweep_metrics``
and ``metrics.compute_seed_uncertainty``.
"""

from __future__ import annotations

import warnings
from io import StringIO

import numpy as np
import pandas as pd

from synth_gen.data import TabularPreprocessor
from synth_gen.evaluation import (
    LOSTask,
    MortalityTask,
    compute_privacy_metrics,
    compute_quality_metrics,
    evaluate_tstr_multi_task,
)
from synth_gen.evaluation.trajectory_metrics import compute_trajectory_metrics

from .generation import create_flat_dataframe, drop_id_columns


def evaluate_tstr(
    X_synth_target: np.ndarray,
    X_synth_cond: np.ndarray,
    preprocessor: TabularPreprocessor,
    target_indices: list[int],
    condition_indices: list[int],
    real_test_path: str,
    real_quality_df: pd.DataFrame | None = None,
    real_holdout_df: pd.DataFrame | None = None,
    compute_privacy: bool = True,
    privacy_max_rows: int = 2000,
    subject_id: np.ndarray | None = None,
    hours_in: np.ndarray | None = None,
    eval_seed: int = 42,
) -> dict:
    """Train classifiers on synthetic, test on real. Returns flat metrics dict."""
    print("    Creating synthetic DataFrame...")
    synth_df = create_flat_dataframe(
        X_synth_target,
        X_synth_cond,
        preprocessor,
        target_indices,
        condition_indices,
        subject_id=subject_id,
        hours_in=hours_in,
    )

    # In-memory CSV roundtrip normalizes dtypes (e.g. integer binary columns)
    # without touching disk, which avoids clobbering between concurrent sweeps.
    buf = StringIO()
    synth_df.to_csv(buf, index=False)
    buf.seek(0)

    print("    Loading datasets for multi-task TSTR...")
    synth_df_raw = pd.read_csv(buf)
    real_df_full = pd.read_csv(real_test_path, low_memory=False)
    if len(real_df_full) > 50000:
        real_df_raw = real_df_full.sample(n=50000, random_state=eval_seed).reset_index(
            drop=True
        )
    else:
        real_df_raw = real_df_full

    metrics = _run_tstr_tasks(synth_df_raw, real_df_raw, eval_seed=eval_seed)
    _merge_trajectory_metrics(metrics, synth_df_raw, real_df_raw)

    real_quality_source = (
        real_quality_df if real_quality_df is not None else real_df_raw
    )
    _merge_quality_metrics(metrics, real_quality_source, synth_df_raw)

    if compute_privacy:
        _merge_privacy_metrics(
            metrics,
            real_quality_source,
            synth_df_raw,
            privacy_max_rows=privacy_max_rows,
            holdout_df=real_holdout_df,
            eval_seed=eval_seed,
        )

    metrics["degenerate_flag"] = _degenerate_flag(metrics)
    return metrics


def _run_tstr_tasks(
    synth_df: pd.DataFrame, real_df: pd.DataFrame, eval_seed: int = 42
) -> dict[str, float]:
    print("    Evaluating multi-task TSTR (Mortality, LOS)...")
    tasks = [
        MortalityTask(
            random_state=eval_seed, use_sequence_model=True, sequence_max_patients=2500
        ),
        LOSTask(
            random_state=eval_seed, use_sequence_model=True, sequence_max_patients=2500
        ),
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Mean of empty slice", category=RuntimeWarning
        )
        task_results = evaluate_tstr_multi_task(
            synth_df, real_df, tasks, test_size=0.3, verbose=False
        )

    metrics: dict[str, float] = {}
    for task_name, task_metrics in task_results.items():
        for metric_name, value in task_metrics.items():
            metrics[f"{task_name}_{metric_name}"] = value

    print("    TSTR Multi-Task Results:")
    mort_synth = metrics.get("mortality_synth_roc_auc", np.nan)
    mort_real = metrics.get("mortality_real_roc_auc", np.nan)
    if np.isfinite(mort_synth):
        print(
            f"      Mortality - Synth AUROC: {mort_synth:.4f}, "
            f"Real AUROC: {mort_real:.4f}"
        )
    los_synth = metrics.get("los_synth_macro_f1", np.nan)
    los_real = metrics.get("los_real_macro_f1", np.nan)
    if np.isfinite(los_synth):
        print(
            f"      LOS - Synth Macro-F1: {los_synth:.4f}, "
            f"Real Macro-F1: {los_real:.4f}"
        )
    return metrics


def _merge_trajectory_metrics(
    metrics: dict, synth_df: pd.DataFrame, real_df: pd.DataFrame
) -> None:
    print("    Computing trajectory metrics (autocorrelation, transitions, etc.)...")
    try:
        if "subject_id" in synth_df.columns and "hours_in" in synth_df.columns:
            traj_metrics = compute_trajectory_metrics(real_df, synth_df)
            metrics.update(traj_metrics)
            for tk, tv in traj_metrics.items():
                if np.isfinite(tv):
                    print(f"      {tk}: {tv:.4f}")
    except Exception as exc:
        print(f"      Warning: Trajectory metric computation failed: {exc}")


def _merge_quality_metrics(
    metrics: dict, real_df: pd.DataFrame, synth_df: pd.DataFrame
) -> None:
    print("    Computing quality metrics (KS, correlation, ranges)...")
    quality_metrics = compute_quality_metrics(
        drop_id_columns(real_df), drop_id_columns(synth_df)
    )
    metrics.update(quality_metrics)

    ks = quality_metrics.get("avg_ks_stat", np.nan)
    if np.isfinite(ks):
        print(f"      KS stat: {ks:.4f}")
    corr = quality_metrics.get("corr_frobenius", np.nan)
    if np.isfinite(corr):
        print(f"      Corr Frobenius: {corr:.4f}")
    rv = quality_metrics.get("range_violation_pct", np.nan)
    if np.isfinite(rv):
        print(f"      Range violations: {rv:.2f}%")


def _merge_privacy_metrics(
    metrics: dict,
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    privacy_max_rows: int,
    holdout_df: pd.DataFrame | None = None,
    eval_seed: int = 42,
) -> None:
    print("    Computing privacy metrics (DCR + membership inference)...")
    try:
        holdout_no_ids = drop_id_columns(holdout_df) if holdout_df is not None else None
        privacy_report = compute_privacy_metrics(
            real_df=drop_id_columns(real_df),
            synthetic_df=drop_id_columns(synth_df),
            holdout_df=holdout_no_ids,
            max_rows=privacy_max_rows,
            random_state=eval_seed,
        )
        dcr_stats = privacy_report.get("dcr_stats", {})
        dcr_baseline = privacy_report.get("dcr_baseline_protection", {})
        dcr_overfit = privacy_report.get("dcr_overfitting_protection", {})
        mia = privacy_report.get("membership_inference_domias_like", {})
        privacy_metrics = {
            "dcr_median": dcr_stats.get("dcr_median", np.nan),
            "dcr_p05": dcr_stats.get("dcr_p05", np.nan),
            "dcr_exact_match_rate": dcr_stats.get("exact_match_rate", np.nan),
            "dcr_baseline_protection": dcr_baseline.get("score", np.nan),
            "dcr_overfitting_protection": dcr_overfit.get("score", np.nan),
            "dcr_closer_to_training_pct": dcr_overfit.get(
                "closer_to_training_pct", np.nan
            ),
            "mia_roc_auc": mia.get("roc_auc", np.nan),
            "mia_average_precision": mia.get("average_precision", np.nan),
            "mia_attacker_advantage": mia.get("attacker_advantage", np.nan),
        }
        metrics.update(privacy_metrics)
        if np.isfinite(privacy_metrics["dcr_median"]):
            print(f"      DCR median: {privacy_metrics['dcr_median']:.4f}")
        if np.isfinite(privacy_metrics["dcr_baseline_protection"]):
            print(
                "      DCR baseline protection: "
                f"{privacy_metrics['dcr_baseline_protection']:.4f}"
            )
        if np.isfinite(privacy_metrics["mia_roc_auc"]):
            print(f"      MIA ROC-AUC: {privacy_metrics['mia_roc_auc']:.4f}")
    except Exception as exc:
        print(f"      Warning: Privacy metric computation failed: {exc}")


def _degenerate_flag(metrics: dict) -> int:
    mortality_auc = metrics.get("mortality_synth_roc_auc", np.nan)
    los_f1 = metrics.get("los_synth_macro_f1", np.nan)
    is_degenerate = (
        np.isfinite(mortality_auc)
        and np.isfinite(los_f1)
        and mortality_auc <= 0.55
        and los_f1 <= 0.15
    )
    if is_degenerate:
        print(
            "      Warning: Degenerate TSTR metrics detected "
            "(near-random utility across tasks)."
        )
    return 1 if is_degenerate else 0


__all__ = ["evaluate_tstr"]
