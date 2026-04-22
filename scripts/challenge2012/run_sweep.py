#!/usr/bin/env python3
"""Internal Challenge 2012 sweep driver (HS3F / ForestFlow / CTGAN).

This script is called by ``scripts/run_sweep.py --dataset challenge2012`` and
wires Challenge-specific paths so the resulting CSV can be reported alongside
the MIMIC sweep as a public, reproducible benchmark (no credentialing required).

All sweep helpers (``_run_sweep_cell``, metric averaging, result schema) are
reused from :mod:`lexisflow.sweep`; only the data-loading paths differ.

Outputs:
    results/challenge2012_sweep_results.csv
    artifacts/challenge2012/sweep/{hour0,autoregressive}_nt*_noise*.pkl

Usage:
    uv run python scripts/run_sweep.py --dataset challenge2012
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


CFG = get_dataset_config("challenge2012")
DEFAULT_PROFILE = "full"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hyperparameter sweep for Challenge 2012 tabular flow-matching. "
            "Trains both hour-0 (IID) and autoregressive generators per cell."
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

        seed_metrics: list[dict] = []
        for si, traj_seed in enumerate(TSTR_TRAJECTORY_SAMPLING_SEEDS):
            print(
                f"\n  [{3 + si}/{total_sweep_steps}] "
                f"Generating ~{synth_samples:,} synthetic rows "
                f"(trajectory_seed={traj_seed}, draw {si + 1}/{n_traj_seeds})..."
            )
            X_synth, X_cond_synth, synth_sid, synth_hours = generate_synthetic_data(
                model,
                autoregressive_inputs.X_cond_train,
                synth_samples,
                trajectory_seed=traj_seed,
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
        f"Autoregressive={format_time(result['autoregressive_train_time_sec'])}"
    )
    print(
        f"    Mortality — Synth ROC-AUC: {result['synth_roc_auc']}, "
        f"Real ROC-AUC: {result['real_roc_auc']}"
    )
    print(
        f"    LOS — Synth F1: {result['los_synth_macro_f1']}, "
        f"Real F1: {result['los_real_macro_f1']}"
    )
    print(
        f"    Quality: KS={result['avg_ks_stat']}, "
        f"Corr={result['corr_frobenius']}, "
        f"RangeViol={result['range_violation_pct']}%"
    )
    dcr = result.get("dcr_median")
    if compute_privacy and dcr is not None and dcr == dcr:
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

    for p in (
        CFG.hour0_data,
        CFG.hour0_preprocessor,
        CFG.autoregressive_data,
        CFG.real_test,
        CFG.real_holdout,
        CFG.autoregressive_preprocessor,
    ):
        if not p.exists():
            raise SystemExit(
                f"ERROR: missing {p}. Run scripts/challenge2012/prepare_* and "
                "fit_* scripts first (see docs/challenge2012.md)."
            )

    print("=" * 70)
    print("Challenge 2012 Hyperparameter Sweep (IID + Autoregressive)")
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
        f"utility/quality/privacy averaged over seeds "
        f"{TSTR_TRAJECTORY_SAMPLING_SEEDS} (mean + std + CI95)."
    )
    print(
        f"Privacy metrics: {'enabled' if compute_privacy else 'disabled'} "
        f"(max_rows={args.privacy_max_rows})"
    )

    CFG.results_csv.parent.mkdir(parents=True, exist_ok=True)
    CFG.sweep_models_dir.mkdir(parents=True, exist_ok=True)
    ensure_results_schema(CFG.results_csv)

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
    completed = load_completed_runs(CFG.results_csv)
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
                models_dir=CFG.sweep_models_dir,
            )
            append_result(CFG.results_csv, result)
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
    print(f"  Results saved to: {CFG.results_csv}")
    print(f"  Model artifacts: {CFG.sweep_models_dir}/")


if __name__ == "__main__":
    main()
