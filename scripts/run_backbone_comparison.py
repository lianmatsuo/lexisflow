#!/usr/bin/env python3
"""Run a single matched-backbone comparison cell (HS3F vs ForestFlow vs CTGAN).

All heavy lifting — data preparation, generator training, trajectory sampling,
TSTR/quality/privacy evaluation, per-seed uncertainty aggregation, and CSV row
assembly — lives in :mod:`synth_gen.sweep`. This driver only orchestrates the
three backbones, packages the delta/winner comparison, and writes the compact
thesis artefacts.

Example:
    uv run python scripts/run_backbone_comparison.py \\
        --nt 5 --n-noise 5 --train-rows 50000 --synth-samples 50000
"""

from __future__ import annotations

import os

# Prevent OpenMP re-init crashes when CTGAN (torch) runs after long multi-process
# joblib/XGBoost phases in the same Python process. Must be set before torch/OMP
# are imported. Leaves BLAS thread counts unconstrained for forest backbones.
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import argparse
import csv
import gc
from datetime import datetime
from pathlib import Path

import numpy as np

from synth_gen.sweep import (
    AutoregressiveInputs,
    Hour0Inputs,
    SWEEP_RESULT_COLUMNS,
    TSTR_TRAJECTORY_SAMPLING_SEEDS,
    add_ctgan_arguments,
    average_sweep_metrics,
    build_result_row,
    compute_seed_uncertainty,
    ctgan_params_from_args,
    evaluate_tstr,
    format_ctgan_params,
    format_time,
    generate_synthetic_data,
    load_autoregressive_inputs,
    load_hour0_inputs,
    parse_int_list,
    train_autoregressive,
    train_hour0,
)


# Comparison driver adds a model_type column per row. All other metric/std/CI95
# columns come from the canonical sweep schema.
RESULT_COLUMNS: list[str] = list(dict.fromkeys(SWEEP_RESULT_COLUMNS + ["model_type"]))

COMPARISON_COLUMNS = [
    "nt",
    "n_noise",
    "train_rows",
    "synth_samples",
    "trajectory_seed_count",
    "hour0_train_time_sec_hs3f",
    "hour0_train_time_sec_forest_flow",
    "hour0_train_time_delta_ff_minus_hs3f",
    "autoregressive_train_time_sec_hs3f",
    "autoregressive_train_time_sec_forest_flow",
    "autoregressive_train_time_delta_ff_minus_hs3f",
    "total_train_time_sec_hs3f",
    "total_train_time_sec_forest_flow",
    "total_train_time_delta_ff_minus_hs3f",
    "total_train_time_speedup_ratio_hs3f_over_ff",
    "synth_accuracy_hs3f",
    "synth_accuracy_forest_flow",
    "synth_accuracy_delta_ff_minus_hs3f",
    "synth_accuracy_combined_ci95",
    "synth_accuracy_significant",
    "synth_accuracy_winner",
    "synth_roc_auc_hs3f",
    "synth_roc_auc_forest_flow",
    "synth_roc_auc_delta_ff_minus_hs3f",
    "synth_roc_auc_combined_ci95",
    "synth_roc_auc_significant",
    "synth_roc_auc_winner",
    "timestamp",
]

SUMMARY_COLUMNS = [
    "metric",
    "label",
    "higher_is_better",
    "hs3f_value",
    "forest_flow_value",
    "delta_ff_minus_hs3f",
    "combined_ci95",
    "significant",
    "winner",
]

BASELINE_SUMMARY_COLUMNS = [
    "nt",
    "n_noise",
    "train_rows",
    "synth_samples",
    "trajectory_seed_count",
    "hour0_train_time_sec_hs3f",
    "hour0_train_time_sec_forest_flow",
    "hour0_train_time_sec_ctgan",
    "autoregressive_train_time_sec_hs3f",
    "autoregressive_train_time_sec_forest_flow",
    "autoregressive_train_time_sec_ctgan",
    "total_train_time_sec_hs3f",
    "total_train_time_sec_forest_flow",
    "total_train_time_sec_ctgan",
    "synth_accuracy_hs3f",
    "synth_accuracy_forest_flow",
    "synth_accuracy_ctgan",
    "synth_roc_auc_hs3f",
    "synth_roc_auc_forest_flow",
    "synth_roc_auc_ctgan",
    "los_synth_macro_f1_hs3f",
    "los_synth_macro_f1_forest_flow",
    "los_synth_macro_f1_ctgan",
    "avg_ks_stat_hs3f",
    "avg_ks_stat_forest_flow",
    "avg_ks_stat_ctgan",
    "dcr_baseline_protection_hs3f",
    "dcr_baseline_protection_forest_flow",
    "dcr_baseline_protection_ctgan",
    "mia_roc_auc_hs3f",
    "mia_roc_auc_forest_flow",
    "mia_roc_auc_ctgan",
    "best_train_time_model",
    "best_synth_accuracy_model",
    "best_synth_roc_auc_model",
    "best_los_synth_macro_f1_model",
    "best_avg_ks_stat_model",
    "best_dcr_baseline_protection_model",
    "best_mia_roc_auc_model",
    "timestamp",
]


def _model_slug(model_type: str) -> str:
    return model_type.replace("-", "_")


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_round(value: object, digits: int) -> float:
    out = _safe_float(value)
    return round(out, digits) if np.isfinite(out) else float("nan")


def _write_csv_row(path: Path, row: dict, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, np.nan) for k in columns})


def _load_existing_result_row(
    path: Path, nt: int, n_noise: int, model_type: str | None = None
) -> dict | None:
    """Load the latest matching per-model row for resumable runs."""
    if not path.exists():
        return None

    latest: dict | None = None
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row_nt = int(row.get("nt", ""))
                    row_n_noise = int(row.get("n_noise", ""))
                except (TypeError, ValueError):
                    continue
                if row_nt == nt and row_n_noise == n_noise:
                    latest = row
    except Exception:
        return None

    if latest is None:
        return None

    out = {col: latest.get(col, np.nan) for col in RESULT_COLUMNS}
    if model_type is not None and not out.get("model_type"):
        out["model_type"] = model_type
    return out


def _significance(delta: float, ci_a: float, ci_b: float) -> tuple[float, bool]:
    if not (np.isfinite(delta) and np.isfinite(ci_a) and np.isfinite(ci_b)):
        return float("nan"), False
    combined = float(np.sqrt((ci_a**2) + (ci_b**2)))
    return combined, bool(abs(delta) > combined)


def _winner(delta: float, higher_is_better: bool, significant: bool) -> str:
    if not np.isfinite(delta):
        return "unclear"
    if not significant:
        return "unclear"
    if higher_is_better:
        return "forest-flow" if delta > 0 else "hs3f"
    return "forest-flow" if delta < 0 else "hs3f"


def _best_model(
    rows_by_model: dict[str, dict], metric_key: str, higher_is_better: bool
) -> str:
    candidates: list[tuple[str, float]] = []
    for model_type, row in rows_by_model.items():
        val = _safe_float(row.get(metric_key, np.nan))
        if np.isfinite(val):
            candidates.append((model_type, val))
    if not candidates:
        return "unclear"
    if higher_is_better:
        return max(candidates, key=lambda item: item[1])[0]
    return min(candidates, key=lambda item: item[1])[0]


def _run_backbone(
    model_type: str,
    nt: int,
    n_noise: int,
    synth_samples: int,
    n_jobs: int,
    compute_privacy: bool,
    privacy_max_rows: int,
    trajectory_seeds: list[int],
    hour0_inputs: Hour0Inputs,
    ar_inputs: AutoregressiveInputs,
    output_path: Path,
    ctgan_params: dict | None = None,
) -> dict:
    """Train + evaluate a single backbone, write its CSV row, return the row."""
    print("\n" + "=" * 80)
    print(f"Backbone run: {model_type} (nt={nt}, n_noise={n_noise})")
    print("=" * 80)

    hour0_model, hour0_train_time = train_hour0(
        hour0_inputs.X_hour0,
        nt,
        n_noise,
        hour0_inputs.feature_types,
        n_jobs=n_jobs,
        model_type=model_type,
        ctgan_params=ctgan_params,
    )
    del hour0_model
    gc.collect()

    model, train_time = train_autoregressive(
        ar_inputs.X_train,
        ar_inputs.X_cond_train,
        nt,
        n_noise,
        ar_inputs.feature_types,
        n_jobs=n_jobs,
        model_type=model_type,
        ctgan_params=ctgan_params,
    )

    seed_metrics: list[dict] = []
    for idx, seed in enumerate(trajectory_seeds, start=1):
        print(
            f"\n  [{idx}/{len(trajectory_seeds)}] "
            f"Generate + evaluate with trajectory_seed={seed}"
        )
        X_synth, X_cond_synth, synth_sid, synth_hours = generate_synthetic_data(
            model,
            ar_inputs.X_cond_train,
            synth_samples,
            trajectory_seed=seed,
        )
        metrics = evaluate_tstr(
            X_synth,
            X_cond_synth,
            ar_inputs.preprocessor,
            ar_inputs.target_indices,
            ar_inputs.condition_indices,
            ar_inputs.real_test_path,
            real_quality_df=ar_inputs.real_quality_df,
            real_holdout_df=ar_inputs.real_holdout_df,
            compute_privacy=compute_privacy,
            privacy_max_rows=privacy_max_rows,
            subject_id=synth_sid,
            hours_in=synth_hours,
            eval_seed=seed,
        )
        seed_metrics.append(metrics)

    averaged = average_sweep_metrics(seed_metrics)
    uncertainty = compute_seed_uncertainty(seed_metrics)
    row = build_result_row(
        nt=nt,
        n_noise=n_noise,
        hour0_train_time=hour0_train_time,
        autoregressive_train_time=train_time,
        metrics=averaged,
        uncertainty=uncertainty,
        trajectory_seed_count=len(trajectory_seeds),
    )
    row["model_type"] = model_type
    _write_csv_row(output_path, row, RESULT_COLUMNS)

    print("\n  Result snapshot:")
    print(
        "    Train time (hour0/ar/total): "
        f"{format_time(hour0_train_time)} / {format_time(train_time)} / "
        f"{format_time(hour0_train_time + train_time)}"
    )
    print(
        f"    Mortality synth accuracy={row['synth_accuracy']}, "
        f"synth AUROC={row['synth_roc_auc']}"
    )
    print(f"    Saved run row -> {output_path}")

    del model
    gc.collect()
    return row


def _build_baseline_summary_row(
    rows_by_model: dict[str, dict],
    train_rows: int,
    synth_samples: int,
) -> dict:
    row = {col: np.nan for col in BASELINE_SUMMARY_COLUMNS}
    reference_row = next(iter(rows_by_model.values()))
    row.update(
        {
            "nt": int(_safe_float(reference_row.get("nt", np.nan))),
            "n_noise": int(_safe_float(reference_row.get("n_noise", np.nan))),
            "train_rows": int(train_rows),
            "synth_samples": int(synth_samples),
            "trajectory_seed_count": int(
                _safe_float(reference_row.get("trajectory_seed_count", np.nan))
            ),
            "timestamp": datetime.now().isoformat(),
        }
    )

    model_order = ["hs3f", "forest-flow", "ctgan"]
    metric_keys = [
        "hour0_train_time_sec",
        "autoregressive_train_time_sec",
        "total_train_time_sec",
        "synth_accuracy",
        "synth_roc_auc",
        "los_synth_macro_f1",
        "avg_ks_stat",
        "dcr_baseline_protection",
        "mia_roc_auc",
    ]
    for model_type in model_order:
        row_data = rows_by_model.get(model_type, {})
        slug = _model_slug(model_type)
        for metric_key in metric_keys:
            row[f"{metric_key}_{slug}"] = row_data.get(metric_key, np.nan)

    row["best_train_time_model"] = _best_model(
        rows_by_model, "total_train_time_sec", higher_is_better=False
    )
    row["best_synth_accuracy_model"] = _best_model(
        rows_by_model, "synth_accuracy", higher_is_better=True
    )
    row["best_synth_roc_auc_model"] = _best_model(
        rows_by_model, "synth_roc_auc", higher_is_better=True
    )
    row["best_los_synth_macro_f1_model"] = _best_model(
        rows_by_model, "los_synth_macro_f1", higher_is_better=True
    )
    row["best_avg_ks_stat_model"] = _best_model(
        rows_by_model, "avg_ks_stat", higher_is_better=False
    )
    row["best_dcr_baseline_protection_model"] = _best_model(
        rows_by_model, "dcr_baseline_protection", higher_is_better=True
    )
    row["best_mia_roc_auc_model"] = _best_model(
        rows_by_model, "mia_roc_auc", higher_is_better=False
    )
    return row


def _build_comparison_row(
    hs3f: dict,
    forest_flow: dict,
    train_rows: int,
    synth_samples: int,
) -> tuple[dict, list[dict]]:
    delta_hour0 = _safe_float(forest_flow["hour0_train_time_sec"]) - _safe_float(
        hs3f["hour0_train_time_sec"]
    )
    delta_ar = _safe_float(forest_flow["autoregressive_train_time_sec"]) - _safe_float(
        hs3f["autoregressive_train_time_sec"]
    )
    delta_total = _safe_float(forest_flow["total_train_time_sec"]) - _safe_float(
        hs3f["total_train_time_sec"]
    )

    hs_total = _safe_float(hs3f["total_train_time_sec"])
    ff_total = _safe_float(forest_flow["total_train_time_sec"])
    speedup_ratio = (
        hs_total / ff_total if np.isfinite(hs_total) and ff_total > 0 else float("nan")
    )

    delta_acc = _safe_float(forest_flow["synth_accuracy"]) - _safe_float(
        hs3f["synth_accuracy"]
    )
    acc_ci95, acc_sig = _significance(
        delta_acc,
        _safe_float(hs3f.get("synth_accuracy_ci95", np.nan)),
        _safe_float(forest_flow.get("synth_accuracy_ci95", np.nan)),
    )

    delta_auc = _safe_float(forest_flow["synth_roc_auc"]) - _safe_float(
        hs3f["synth_roc_auc"]
    )
    auc_ci95, auc_sig = _significance(
        delta_auc,
        _safe_float(hs3f.get("synth_roc_auc_ci95", np.nan)),
        _safe_float(forest_flow.get("synth_roc_auc_ci95", np.nan)),
    )

    comparison_row = {
        "nt": int(hs3f["nt"]),
        "n_noise": int(hs3f["n_noise"]),
        "train_rows": train_rows,
        "synth_samples": synth_samples,
        "trajectory_seed_count": int(hs3f["trajectory_seed_count"]),
        "hour0_train_time_sec_hs3f": hs3f["hour0_train_time_sec"],
        "hour0_train_time_sec_forest_flow": forest_flow["hour0_train_time_sec"],
        "hour0_train_time_delta_ff_minus_hs3f": _safe_round(delta_hour0, 4),
        "autoregressive_train_time_sec_hs3f": hs3f["autoregressive_train_time_sec"],
        "autoregressive_train_time_sec_forest_flow": forest_flow[
            "autoregressive_train_time_sec"
        ],
        "autoregressive_train_time_delta_ff_minus_hs3f": _safe_round(delta_ar, 4),
        "total_train_time_sec_hs3f": hs3f["total_train_time_sec"],
        "total_train_time_sec_forest_flow": forest_flow["total_train_time_sec"],
        "total_train_time_delta_ff_minus_hs3f": _safe_round(delta_total, 4),
        "total_train_time_speedup_ratio_hs3f_over_ff": _safe_round(speedup_ratio, 6),
        "synth_accuracy_hs3f": hs3f["synth_accuracy"],
        "synth_accuracy_forest_flow": forest_flow["synth_accuracy"],
        "synth_accuracy_delta_ff_minus_hs3f": _safe_round(delta_acc, 6),
        "synth_accuracy_combined_ci95": _safe_round(acc_ci95, 6),
        "synth_accuracy_significant": bool(acc_sig),
        "synth_accuracy_winner": _winner(
            delta_acc, higher_is_better=True, significant=acc_sig
        ),
        "synth_roc_auc_hs3f": hs3f["synth_roc_auc"],
        "synth_roc_auc_forest_flow": forest_flow["synth_roc_auc"],
        "synth_roc_auc_delta_ff_minus_hs3f": _safe_round(delta_auc, 6),
        "synth_roc_auc_combined_ci95": _safe_round(auc_ci95, 6),
        "synth_roc_auc_significant": bool(auc_sig),
        "synth_roc_auc_winner": _winner(
            delta_auc, higher_is_better=True, significant=auc_sig
        ),
        "timestamp": datetime.now().isoformat(),
    }

    summary_rows = [
        {
            "metric": "total_train_time_sec",
            "label": "Total Training Time (sec)",
            "higher_is_better": False,
            "hs3f_value": hs3f["total_train_time_sec"],
            "forest_flow_value": forest_flow["total_train_time_sec"],
            "delta_ff_minus_hs3f": _safe_round(delta_total, 6),
            "combined_ci95": np.nan,
            "significant": np.nan,
            "winner": "forest-flow" if delta_total < 0 else "hs3f",
        },
        {
            "metric": "synth_accuracy",
            "label": "Mortality Accuracy (TSTR)",
            "higher_is_better": True,
            "hs3f_value": hs3f["synth_accuracy"],
            "forest_flow_value": forest_flow["synth_accuracy"],
            "delta_ff_minus_hs3f": _safe_round(delta_acc, 6),
            "combined_ci95": _safe_round(acc_ci95, 6),
            "significant": bool(acc_sig),
            "winner": _winner(delta_acc, higher_is_better=True, significant=acc_sig),
        },
        {
            "metric": "synth_roc_auc",
            "label": "Mortality AUROC (TSTR)",
            "higher_is_better": True,
            "hs3f_value": hs3f["synth_roc_auc"],
            "forest_flow_value": forest_flow["synth_roc_auc"],
            "delta_ff_minus_hs3f": _safe_round(delta_auc, 6),
            "combined_ci95": _safe_round(auc_ci95, 6),
            "significant": bool(auc_sig),
            "winner": _winner(delta_auc, higher_is_better=True, significant=auc_sig),
        },
    ]

    return comparison_row, summary_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-cell matched backbone comparison (HS3F vs ForestFlow vs CTGAN)."
        )
    )
    parser.add_argument(
        "--nt", type=int, default=5, help="Flow time levels (default: 5)"
    )
    parser.add_argument(
        "--n-noise", type=int, default=5, help="Noise samples per cell (default: 5)"
    )
    parser.add_argument(
        "--train-rows",
        type=int,
        default=50000,
        help="Training rows for both backbones (default: 50000)",
    )
    parser.add_argument(
        "--synth-samples",
        type=int,
        default=50000,
        help="Synthetic rows generated per seed (default: 50000)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Parallel jobs for tree models (default: 8)",
    )
    parser.add_argument(
        "--trajectory-seeds",
        type=parse_int_list,
        default=list(TSTR_TRAJECTORY_SAMPLING_SEEDS),
        help=(
            "Comma-separated trajectory generation seeds "
            f"(default: {','.join(str(v) for v in TSTR_TRAJECTORY_SAMPLING_SEEDS)})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/backbone_comparison"),
        help="Directory for per-backbone + comparison CSVs",
    )
    parser.add_argument(
        "--refresh-transformed-cache",
        action="store_true",
        help="Rebuild transformed autoregressive cache",
    )
    parser.add_argument(
        "--skip-privacy-metrics",
        action="store_true",
        help="Skip privacy metrics to reduce runtime",
    )
    parser.add_argument(
        "--privacy-max-rows",
        type=int,
        default=2000,
        help="Max rows for privacy metrics (default: 2000)",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete existing output CSVs before running",
    )
    add_ctgan_arguments(parser)
    return parser.parse_args()


def _shutdown_joblib_pool() -> None:
    """Release loky workers before CTGAN (torch) re-enters OpenMP.

    Prevents ``OMP: Error #179 pthread_mutex_init`` caused by semaphore/thread
    exhaustion when long XGBoost joblib phases leave worker pools alive.
    """
    try:
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True)
    except Exception:  # pragma: no cover - best-effort cleanup
        pass
    gc.collect()


def main() -> None:
    args = _parse_args()

    compute_privacy = not args.skip_privacy_metrics
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    hs3f_path = output_dir / "sweep_results_hs3f.csv"
    forest_flow_path = output_dir / "sweep_results_forest_flow.csv"
    ctgan_path = output_dir / "sweep_results_ctgan.csv"
    comparison_path = output_dir / "backbone_matched_comparison.csv"
    summary_path = output_dir / "backbone_matched_summary.csv"
    baseline_summary_path = output_dir / "backbone_baseline_comparison.csv"

    if args.overwrite_output:
        for p in (
            hs3f_path,
            forest_flow_path,
            ctgan_path,
            comparison_path,
            summary_path,
            baseline_summary_path,
        ):
            if p.exists():
                p.unlink()

    print("=" * 80)
    print("Matched Backbone Comparison: HS3F vs ForestFlow vs CTGAN")
    print("=" * 80)
    print(
        "Config: "
        f"nt={args.nt}, n_noise={args.n_noise}, train_rows={args.train_rows:,}, "
        f"synth_samples={args.synth_samples:,}, n_jobs={args.n_jobs}"
    )
    print(
        "Resume completed rows: "
        f"{'disabled' if args.overwrite_output else 'enabled (default)'}"
    )
    print(f"Trajectory seeds: {args.trajectory_seeds}")
    print(
        f"Privacy metrics: {'enabled' if compute_privacy else 'disabled'} "
        f"(max_rows={args.privacy_max_rows})"
    )

    ctgan_params = ctgan_params_from_args(args)
    print(f"CTGAN params: {format_ctgan_params(ctgan_params)}")

    hour0_inputs = load_hour0_inputs(args.train_rows)
    ar_inputs = load_autoregressive_inputs(
        args.train_rows,
        refresh_cache=args.refresh_transformed_cache,
    )

    def _run_or_resume(model_type: str, output_path: Path) -> tuple[dict, bool]:
        if not args.overwrite_output:
            cached_row = _load_existing_result_row(
                output_path,
                nt=args.nt,
                n_noise=args.n_noise,
                model_type=model_type,
            )
            if cached_row is not None:
                print(
                    f"\n[SKIP] {model_type} nt={args.nt}, "
                    f"n_noise={args.n_noise} (already completed)"
                )
                return cached_row, True

        return (
            _run_backbone(
                model_type=model_type,
                nt=args.nt,
                n_noise=args.n_noise,
                synth_samples=args.synth_samples,
                n_jobs=args.n_jobs,
                compute_privacy=compute_privacy,
                privacy_max_rows=args.privacy_max_rows,
                trajectory_seeds=args.trajectory_seeds,
                hour0_inputs=hour0_inputs,
                ar_inputs=ar_inputs,
                output_path=output_path,
                ctgan_params=ctgan_params,
            ),
            False,
        )

    hs3f_row, hs3f_cached = _run_or_resume("hs3f", hs3f_path)
    forest_flow_row, forest_flow_cached = _run_or_resume(
        "forest-flow", forest_flow_path
    )

    _shutdown_joblib_pool()

    ctgan_row, ctgan_cached = _run_or_resume("ctgan", ctgan_path)

    comparison_row, summary_rows = _build_comparison_row(
        hs3f_row,
        forest_flow_row,
        train_rows=args.train_rows,
        synth_samples=args.synth_samples,
    )
    baseline_summary_row = _build_baseline_summary_row(
        rows_by_model={
            "hs3f": hs3f_row,
            "forest-flow": forest_flow_row,
            "ctgan": ctgan_row,
        },
        train_rows=args.train_rows,
        synth_samples=args.synth_samples,
    )
    all_from_cache = hs3f_cached and forest_flow_cached and ctgan_cached
    if all_from_cache and not args.overwrite_output:
        print(
            "\nAll model rows loaded from cache; skipping duplicate comparison writes."
        )
    else:
        _write_csv_row(comparison_path, comparison_row, COMPARISON_COLUMNS)
        for row in summary_rows:
            _write_csv_row(summary_path, row, SUMMARY_COLUMNS)
        _write_csv_row(
            baseline_summary_path, baseline_summary_row, BASELINE_SUMMARY_COLUMNS
        )

    print("\n" + "=" * 80)
    print("Comparison complete.")
    print("=" * 80)
    print(f"HS3F row:        {hs3f_path}")
    print(f"ForestFlow row:  {forest_flow_path}")
    print(f"CTGAN row:       {ctgan_path}")
    print(f"Matched compare: {comparison_path}")
    print(f"Summary table:   {summary_path}")
    print(f"Baseline table:  {baseline_summary_path}")
    print("\nKey deltas (forest-flow minus hs3f):")
    print(
        "  total_train_time_sec = "
        f"{comparison_row['total_train_time_delta_ff_minus_hs3f']} "
        f"(speedup ratio hs3f/ff: {comparison_row['total_train_time_speedup_ratio_hs3f_over_ff']})"
    )
    print(
        "  synth_accuracy       = "
        f"{comparison_row['synth_accuracy_delta_ff_minus_hs3f']} "
        f"(winner: {comparison_row['synth_accuracy_winner']})"
    )
    print(
        "  synth_roc_auc        = "
        f"{comparison_row['synth_roc_auc_delta_ff_minus_hs3f']} "
        f"(winner: {comparison_row['synth_roc_auc_winner']})"
    )
    print("\nCTGAN best-of summary:")
    print(
        f"  best_train_time_model        = {baseline_summary_row['best_train_time_model']}"
    )
    print(
        f"  best_synth_accuracy_model    = {baseline_summary_row['best_synth_accuracy_model']}"
    )
    print(
        f"  best_synth_roc_auc_model     = {baseline_summary_row['best_synth_roc_auc_model']}"
    )
    print(
        f"  best_los_synth_macro_f1_model= {baseline_summary_row['best_los_synth_macro_f1_model']}"
    )


if __name__ == "__main__":
    main()
