#!/usr/bin/env python3
"""Single entrypoint for dataset sweep pipelines.

This command runs the full pipeline for one dataset:
1) Prepare hour-0 + autoregressive CSVs
2) Fit hour-0 + autoregressive preprocessors
3) Execute the dataset sweep

Usage:
    uv run python scripts/run_sweep.py --dataset mimic
    uv run python scripts/run_sweep.py --dataset challenge2012 --reset
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path

from lexisflow.config import DATASET_CONFIGS, get_dataset_config


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full sweep pipeline for one dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        default="mimic",
        help="Dataset pipeline to run (default: mimic).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete generated dataset artifacts/results before running.",
    )
    parser.add_argument(
        "--profile",
        default="full",
        help=(
            "Sweep profile forwarded to dataset run_sweep.py "
            "(for example: full, smoke)."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Override parallel worker count for the dataset sweep step. "
            "If omitted, the dataset/profile default is used."
        ),
    )
    return parser


def _delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        print(f"  reset: removed directory {path}")
    elif path.exists():
        path.unlink()
        print(f"  reset: removed file {path}")


def _reset_dataset_outputs(dataset: str) -> None:
    cfg = get_dataset_config(dataset)
    print(f"\nResetting generated outputs for dataset='{dataset}'...")
    for path in cfg.reset_targets:
        _delete_path(path)
    print("Reset complete.\n")


def _run_script(script_path: Path, extra_args: list[str] | None = None) -> None:
    if not script_path.exists():
        raise SystemExit(f"Missing script: {script_path}")
    original_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path)] + (extra_args or [])
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = original_argv


def _has_all_outputs(paths: tuple[Path, ...]) -> bool:
    return all(path.exists() for path in paths)


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = get_dataset_config(args.dataset)
    cfg.get_sweep_defaults(args.profile)

    if args.reset:
        _reset_dataset_outputs(args.dataset)

    steps = [
        (
            "prepare hour-0 data",
            cfg.script_dir / "prepare_hour0.py",
            (cfg.hour0_data,),
        ),
        (
            "prepare autoregressive data",
            cfg.script_dir / "prepare_autoregressive.py",
            (cfg.autoregressive_data, cfg.real_test, cfg.real_holdout),
        ),
        (
            "fit hour-0 preprocessor",
            cfg.script_dir / "fit_hour0_preprocessor.py",
            (cfg.hour0_preprocessor,),
        ),
        (
            "fit autoregressive preprocessor",
            cfg.script_dir / "fit_autoregressive_preprocessor.py",
            (cfg.autoregressive_preprocessor,),
        ),
        ("run sweep", cfg.script_dir / "run_sweep.py", ()),
    ]

    print("=" * 72)
    print(f"Running full sweep pipeline (dataset={args.dataset})")
    print(f"Sweep profile: {args.profile}")
    if args.n_jobs is not None:
        print(f"Parallel workers override: n_jobs={args.n_jobs}")
    print("=" * 72)
    for idx, (label, script_path, expected_outputs) in enumerate(steps, start=1):
        print(f"\n[{idx}/{len(steps)}] {label}: {script_path}")
        if not args.reset and expected_outputs and _has_all_outputs(expected_outputs):
            print("  cache: outputs already exist, skipping step")
            continue
        if label == "run sweep":
            sweep_args = ["--profile", args.profile]
            if args.n_jobs is not None:
                sweep_args.extend(["--n-jobs", str(args.n_jobs)])
            _run_script(script_path, extra_args=sweep_args)
        else:
            _run_script(script_path)

    print("\nPipeline complete.")
    print(f"Results CSV: {cfg.results_csv}")
    print(f"Sweep artifacts: {cfg.sweep_models_dir}")


if __name__ == "__main__":
    main()
