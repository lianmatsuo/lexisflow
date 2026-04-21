#!/usr/bin/env python3
"""Hyperparameter sweep for tabular flow-matching generators.

Trains BOTH hour-0 (IID) and autoregressive models with the same hyperparameters.
Each (nt, n_noise) combination trains both models for complete synthetic
generation, then aggregates TSTR + quality + privacy + trajectory metrics over
``TSTR_TRAJECTORY_SAMPLING_SEEDS`` per sweep cell (mean + std + CI95).

This file is intentionally thin: all heavy lifting (data loading, generator
training, trajectory sampling, evaluation, schema management) lives in
:mod:`lexisflow.sweep`. Importing those helpers keeps the sweep driver and the
matched-backbone comparison driver in lockstep.

Usage:
    uv run python scripts/mimic/run_sweep.py
    uv run python scripts/mimic/run_sweep.py --train-rows 10000 --synth-samples 5000
"""

from __future__ import annotations

import argparse
import gc
import pickle
import time
from pathlib import Path

from tqdm import tqdm

from lexisflow.config import get_dataset_config

from lexisflow.sweep import (
    DEFAULT_N_JOBS,
    SEQUENCE_TIMESTEPS,
    TSTR_TRAJECTORY_SAMPLING_SEEDS,
    add_ctgan_arguments,
    append_result,
    average_sweep_metrics,
    build_error_row,
    build_result_row,
    compute_seed_uncertainty,
    ctgan_params_from_args,
    ensure_results_schema,
    evaluate_tstr,
    format_ctgan_params,
    format_time,
    generate_synthetic_data,
    load_autoregressive_inputs,
    load_completed_runs,
    load_hour0_inputs,
    parse_int_list,
    train_autoregressive,
    train_hour0,
)


CFG = get_dataset_config("mimic")
DEFAULT_PROFILE = "full"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter sweep for tabular flow-matching models. Trains both "
            "hour-0 (IID) and autoregressive generators per (nt, n_noise) cell."
        )
    )
    parser.add_argument(
        "--profile",
        choices=sorted(CFG.sweep_profiles),
        default=DEFAULT_PROFILE,
        help=f"Sweep profile preset (default: {DEFAULT_PROFILE}).",
    )
    parser.add_argument("--train-rows", type=int, default=None)
    parser.add_argument("--synth-samples", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--nt-values", type=parse_int_list, default=None)
    parser.add_argument("--noise-values", type=parse_int_list, default=None)
    parser.add_argument(
        "--model-type",
        choices=["forest-flow", "hs3f", "ctgan"],
        default="hs3f",
    )
    parser.add_argument("--refresh-transformed-cache", action="store_true")
    parser.add_argument("--skip-privacy-metrics", action="store_true")
    parser.add_argument("--privacy-max-rows", type=int, default=None)
    add_ctgan_arguments(parser)
    return parser


def _run_sweep_cell(
    nt: int,
    n_noise: int,
    *,
    model_type: str,
    ctgan_params: dict,
    n_jobs: int,
    synth_samples: int,
    compute_privacy: bool,
    privacy_max_rows: int,
    hour0_inputs,
    autoregressive_inputs,
    models_dir: Path,
) -> tuple[dict, float]:
    """Run one (nt, n_noise) sweep cell and return (result row, wall time)."""
    run_start = time.time()
    n_traj_seeds = len(TSTR_TRAJECTORY_SAMPLING_SEEDS)
    total_sweep_steps = 2 + n_traj_seeds
    run_progress = tqdm(
        total=total_sweep_steps,
        desc=f"Run nt={nt}, noise={n_noise}",
        unit="step",
        leave=False,
    )

    model = None
    try:
        print(f"\n  [1/{total_sweep_steps}] Training Hour-0 IID model...")
        hour0_model, hour0_train_time = train_hour0(
            hour0_inputs.X_hour0,
            nt,
            n_noise,
            hour0_inputs.feature_types,
            n_jobs=n_jobs,
            model_type=model_type,
            ctgan_params=ctgan_params,
        )
        run_progress.update(1)

        hour0_model_path = models_dir / f"hour0_nt{nt}_noise{n_noise}.pkl"
        with open(hour0_model_path, "wb") as f:
            pickle.dump(
                {
                    "model": hour0_model,
                    "preprocessor_path": hour0_inputs.preprocessor_path,
                    "all_cols": hour0_inputs.all_cols,
                    "n_features": hour0_inputs.preprocessor.n_features,
                    "training_samples": hour0_inputs.X_hour0.shape[0],
                    "nt": nt,
                    "n_noise": n_noise,
                    "training_time_seconds": hour0_train_time,
                    "model_type": model_type,
                },
                f,
            )
        print(f"    Saved to: {hour0_model_path}")
        del hour0_model
        gc.collect()

        print(f"\n  [2/{total_sweep_steps}] Training Autoregressive model...")
        model, train_time = train_autoregressive(
            autoregressive_inputs.X_train,
            autoregressive_inputs.X_cond_train,
            nt,
            n_noise,
            autoregressive_inputs.feature_types,
            n_jobs=n_jobs,
            model_type=model_type,
            ctgan_params=ctgan_params,
        )
        print(f"    Autoregressive training: {format_time(train_time)}")
        run_progress.update(1)

        autoregressive_model_path = (
            models_dir / f"autoregressive_nt{nt}_noise{n_noise}.pkl"
        )
        with open(autoregressive_model_path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "all_cols": autoregressive_inputs.all_cols,
                    "target_indices": autoregressive_inputs.target_indices,
                    "condition_indices": autoregressive_inputs.condition_indices,
                    "training_samples": autoregressive_inputs.X_train.shape[0],
                    "nt": nt,
                    "n_noise": n_noise,
                    "training_time_seconds": train_time,
                    "model_type": model_type,
                },
                f,
            )
        print(f"    Saved to: {autoregressive_model_path}")

        seed_metrics: list[dict] = []
        for si, traj_seed in enumerate(TSTR_TRAJECTORY_SAMPLING_SEEDS):
            print(
                f"\n  [{3 + si}/{total_sweep_steps}] "
                f"Generating ~{synth_samples:,} synthetic trajectory rows "
                f"(trajectory_seed={traj_seed}, draw {si + 1}/{n_traj_seeds})..."
            )
            X_synth, X_cond_synth, synth_sid, synth_hours = generate_synthetic_data(
                model,
                autoregressive_inputs.X_cond_train,
                synth_samples,
                trajectory_seed=traj_seed,
            )
            print(
                f"\n  TSTR evaluation (trajectory_seed={traj_seed}, "
                f"{si + 1}/{n_traj_seeds})..."
            )
            seed_metrics.append(
                evaluate_tstr(
                    X_synth,
                    X_cond_synth,
                    autoregressive_inputs.preprocessor,
                    autoregressive_inputs.target_indices,
                    autoregressive_inputs.condition_indices,
                    autoregressive_inputs.real_test_path,
                    real_quality_df=autoregressive_inputs.real_quality_df,
                    real_holdout_df=autoregressive_inputs.real_holdout_df,
                    compute_privacy=compute_privacy,
                    privacy_max_rows=privacy_max_rows,
                    subject_id=synth_sid,
                    hours_in=synth_hours,
                    eval_seed=traj_seed,
                )
            )
            run_progress.update(1)

        metrics = average_sweep_metrics(seed_metrics)
        uncertainty = compute_seed_uncertainty(seed_metrics)
        print(
            f"\n  Recorded sweep row = mean/std/CI95 over trajectory seeds "
            f"{TSTR_TRAJECTORY_SAMPLING_SEEDS} ({n_traj_seeds} evaluations)."
        )

        result = build_result_row(
            nt=nt,
            n_noise=n_noise,
            hour0_train_time=hour0_train_time,
            autoregressive_train_time=train_time,
            metrics=metrics,
            uncertainty=uncertainty,
            trajectory_seed_count=n_traj_seeds,
        )
        return result, time.time() - run_start

    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        return (
            build_error_row(nt, n_noise, exc, trajectory_seed_count=n_traj_seeds),
            time.time() - run_start,
        )

    finally:
        if model is not None:
            del model
        gc.collect()
        run_progress.close()


def _print_result_snapshot(
    result: dict, run_time: float, compute_privacy: bool
) -> None:
    print("\n  Results:")
    print(
        f"    Training: Hour-0={format_time(result['hour0_train_time_sec'])}, "
        f"Autoregressive={format_time(result['autoregressive_train_time_sec'])}, "
        f"Total={format_time(result['total_train_time_sec'])}"
    )
    print(
        f"    Mortality - Synth ROC-AUC: {result['synth_roc_auc']}, "
        f"Real ROC-AUC: {result['real_roc_auc']}"
    )
    print(
        f"    LOS - Synth F1: {result['los_synth_macro_f1']}, "
        f"Real F1: {result['los_real_macro_f1']}"
    )
    print(
        f"    Quality: KS={result['avg_ks_stat']}, "
        f"Corr={result['corr_frobenius']}, "
        f"RangeViol={result['range_violation_pct']}%"
    )
    dcr = result.get("dcr_median")
    if compute_privacy and dcr is not None and dcr == dcr:  # not NaN
        print(
            "    Privacy: "
            f"DCRmed={result['dcr_median']}, "
            f"DCRbase={result['dcr_baseline_protection']}, "
            f"MIA-AUC={result['mia_roc_auc']}"
        )
    print(f"    Run time: {format_time(run_time)}")


def main() -> None:
    args = _build_argparser().parse_args()
    sweep_defaults = CFG.get_sweep_defaults(args.profile)
    if args.train_rows is None:
        args.train_rows = sweep_defaults.max_train_rows
    if args.synth_samples is None:
        args.synth_samples = sweep_defaults.n_synth_samples
    if args.nt_values is None:
        args.nt_values = list(sweep_defaults.nt_values)
    if args.noise_values is None:
        args.noise_values = list(sweep_defaults.noise_values)
    if args.privacy_max_rows is None:
        args.privacy_max_rows = sweep_defaults.privacy_max_rows

    ctgan_params = ctgan_params_from_args(args)

    print("=" * 70)
    print("Forest-Flow Hyperparameter Sweep (IID + Autoregressive)")
    print("=" * 70)
    print(
        f"Config: train_rows={args.train_rows:,}, "
        f"synth_samples={args.synth_samples:,}, n_jobs={args.n_jobs}"
    )
    print(f"Sweep profile: {args.profile}")
    print(f"Model type: {args.model_type}")
    if args.model_type == "ctgan":
        print(f"CTGAN params: {format_ctgan_params(ctgan_params)}")
    print(f"nt values: {args.nt_values}")
    print(f"noise values: {args.noise_values}")

    compute_privacy = not args.skip_privacy_metrics
    print(
        "TSTR mode: patient-level sequence classifiers "
        f"(mortality, LOS; {SEQUENCE_TIMESTEPS}h trajectories); "
        f"utility/quality/privacy metrics aggregated over trajectory seeds "
        f"{TSTR_TRAJECTORY_SAMPLING_SEEDS} (mean + std + CI95 for key metrics)"
    )
    print(
        f"privacy metrics: {'enabled' if compute_privacy else 'disabled'} "
        f"(max_rows={args.privacy_max_rows})"
    )
    print("Note: Each (nt, n_noise) trains BOTH hour-0 and autoregressive models")

    results_path = CFG.results_csv
    models_dir = CFG.sweep_models_dir
    results_path.parent.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    ensure_results_schema(results_path)

    print("\n" + "=" * 70)
    print("Loading Hour-0 Data")
    print("=" * 70)
    hour0_inputs = load_hour0_inputs(
        args.train_rows,
        dataset=CFG.name,
    )

    print("\n" + "=" * 70)
    print("Loading Autoregressive Data")
    print("=" * 70)
    completed = load_completed_runs(results_path)
    total_runs = len(args.nt_values) * len(args.noise_values)
    print(f"\nCompleted: {len(completed)}/{total_runs} runs")

    ar_inputs = load_autoregressive_inputs(
        args.train_rows,
        dataset=CFG.name,
        refresh_cache=args.refresh_transformed_cache,
    )

    print("\n" + "=" * 70)
    print("Starting sweep...")
    print("=" * 70)
    sweep_start = time.time()

    sweep_progress = tqdm(total=total_runs, desc="Sweep configurations", unit="config")
    if completed:
        sweep_progress.update(len(completed))

    for nt in args.nt_values:
        for n_noise in args.noise_values:
            if (nt, n_noise) in completed:
                print(f"\n[SKIP] nt={nt}, n_noise={n_noise} (already completed)")
                sweep_progress.update(1)
                continue

            print("\n" + "=" * 70)
            print(f"[RUN] nt={nt}, n_noise={n_noise}")
            print("=" * 70)

            result, run_time = _run_sweep_cell(
                nt=nt,
                n_noise=n_noise,
                model_type=args.model_type,
                ctgan_params=ctgan_params,
                n_jobs=args.n_jobs,
                synth_samples=args.synth_samples,
                compute_privacy=compute_privacy,
                privacy_max_rows=args.privacy_max_rows,
                hour0_inputs=hour0_inputs,
                autoregressive_inputs=ar_inputs,
                models_dir=models_dir,
            )
            append_result(results_path, result)
            completed.add((nt, n_noise))

            if not result.get("error"):
                _print_result_snapshot(result, run_time, compute_privacy)

            sweep_progress.update(1)

    sweep_progress.close()

    total_time = time.time() - sweep_start
    print("\n" + "=" * 70)
    print("Sweep Complete!")
    print("=" * 70)
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Completed runs: {len(completed)}/{total_runs}")
    print(f"  Results saved to: {results_path}")
    print(f"  Model artifacts saved to: {models_dir}/")
    print("    Files: hour0_nt*_noise*.pkl, autoregressive_nt*_noise*.pkl")
    print("\nNext step:")
    print("  • Run scripts/common/analyze_sweep.py to visualize results")


if __name__ == "__main__":
    main()
