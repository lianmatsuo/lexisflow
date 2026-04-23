#!/usr/bin/env python3
"""Evaluate sweep hour-0 models in isolation against real hour-0 rows.

This script closes the "hour-0 diagnostic gap" by evaluating the initial-state
generator directly (without autoregressive rollout or TSTR tasks).

For each hour-0 checkpoint in the sweep directory (filtered by nt / n_noise),
it:
1) Samples synthetic hour-0 rows from the IID model.
2) Samples an equally sized real hour-0 reference set.
3) Computes tabular two-sample fidelity metrics.
4) Writes a ranked CSV report.

Default sample sizes:
    synthetic rows per model = 5000
    real rows per comparison  = 5000
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
import types
from pathlib import Path
from typing import Any

import lexisflow
import lexisflow.data
import lexisflow.data.transformers as _transformers
import lexisflow.models
import lexisflow.models.forest_flow as _forest_flow
import lexisflow.models.hs3f as _hs3f
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from lexisflow.config import DATASET_CONFIGS, get_dataset_config
from lexisflow.evaluation.quality_metrics import compute_quality_metrics


MODEL_RE = re.compile(r"hour0_nt(?P<nt>\d+)_noise(?P<nnoise>\d+)\.pkl$")


def _install_legacy_pickle_shims() -> None:
    """Map legacy module paths used in older model pickles."""
    # Older artifacts may reference pre-rename modules:
    # - packages.models.*
    # - src.forest_flow.*
    # - synth_gen.*
    for legacy, target in [
        ("packages", types.ModuleType("packages")),
        ("packages.models", types.ModuleType("packages.models")),
        ("packages.models.hs3f", _hs3f),
        ("packages.models.forest_flow", _forest_flow),
        ("src", types.ModuleType("src")),
        ("src.forest_flow", types.ModuleType("src.forest_flow")),
        ("src.forest_flow.preprocessing", _transformers),
        ("src.forest_flow.model", _forest_flow),
        ("src.forest_flow.hs3f", _hs3f),
        ("synth_gen", lexisflow),
        ("synth_gen.data", lexisflow.data),
        ("synth_gen.data.transformers", _transformers),
        ("synth_gen.models", lexisflow.models),
        ("synth_gen.models.forest_flow", _forest_flow),
        ("synth_gen.models.hs3f", _hs3f),
    ]:
        sys.modules.setdefault(legacy, target)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate hour-0 sweep checkpoints directly against real hour-0 rows."
        )
    )
    parser.add_argument(
        "--dataset",
        default="mimic",
        choices=sorted(DATASET_CONFIGS),
        help="Dataset preset for default paths (default: mimic).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory containing hour0_nt*_noise*.pkl artifacts.",
    )
    parser.add_argument(
        "--hour0-data",
        type=Path,
        default=None,
        help="Real hour-0 CSV path (defaults to dataset hour0_data).",
    )
    parser.add_argument(
        "--hour0-preprocessor",
        type=Path,
        default=None,
        help="Hour-0 preprocessor pickle path (defaults to dataset config).",
    )
    parser.add_argument(
        "--min-nt",
        type=int,
        default=1,
        help="Minimum nt to include (default: 1).",
    )
    parser.add_argument(
        "--min-nnoise",
        type=int,
        default=1,
        help="Minimum n_noise to include (default: 1).",
    )
    parser.add_argument(
        "--n-synth",
        type=int,
        default=5000,
        help="Synthetic hour-0 rows generated per model (default: 5000).",
    )
    parser.add_argument(
        "--n-real",
        type=int,
        default=5000,
        help="Real hour-0 rows sampled per model comparison (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for fixed real-sample selection (default: 42).",
    )
    parser.add_argument(
        "--synth-seeds",
        type=str,
        default="42,11,50",
        help=(
            "Comma-separated synthetic sampling seeds reused for every model "
            "(default: 42,11,50). Metrics are averaged across these seeds."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: results/<dataset>_hour0_diagnostics.csv).",
    )
    return parser


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _discover_hour0_models(
    models_dir: Path, min_nt: int, min_nnoise: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(models_dir.glob("hour0_nt*_noise*.pkl")):
        m = MODEL_RE.match(p.name)
        if m is None:
            continue
        nt = int(m.group("nt"))
        nnoise = int(m.group("nnoise"))
        if nt < min_nt or nnoise < min_nnoise:
            continue
        rows.append({"path": p, "nt": nt, "n_noise": nnoise})
    rows.sort(key=lambda r: (r["nt"], r["n_noise"]))
    return rows


def _fmt_metric(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{digits}f}"


def _parse_seed_list(raw: str) -> list[int]:
    seeds: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("synth seed list is empty")
    # Preserve order but drop duplicates.
    return list(dict.fromkeys(seeds))


def _sample_real_hour0(
    real_df: pd.DataFrame,
    n_rows: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, bool]:
    replace = len(real_df) < n_rows
    idx = rng.choice(len(real_df), size=n_rows, replace=replace)
    return real_df.iloc[idx].reset_index(drop=True), idx, replace


def _sample_synth_hour0(model: Any, n_rows: int, seed: int) -> np.ndarray:
    try:
        sampled = model.sample(n_samples=n_rows, X_condition=None, random_state=seed)
    except TypeError:
        sampled = model.sample(n_samples=n_rows, X_condition=None)
    return np.asarray(sampled)


def _numeric_distribution_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_cols: list[str],
) -> dict[str, float]:
    ks_vals: list[float] = []
    wass_vals: list[float] = []
    wass_scaled_vals: list[float] = []

    for col in numeric_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        real_vals = pd.to_numeric(real_df[col], errors="coerce").dropna().to_numpy()
        synth_vals = pd.to_numeric(synth_df[col], errors="coerce").dropna().to_numpy()
        if len(real_vals) < 2 or len(synth_vals) < 2:
            continue

        ks_stat, _ = stats.ks_2samp(real_vals, synth_vals)
        ks_vals.append(float(ks_stat))

        wdist = float(stats.wasserstein_distance(real_vals, synth_vals))
        wass_vals.append(wdist)

        real_scale = float(np.nanstd(real_vals))
        wass_scaled_vals.append(wdist / real_scale if real_scale > 1e-8 else wdist)

    def _summary(vals: list[float], prefix: str) -> dict[str, float]:
        if not vals:
            return {
                f"{prefix}_mean": float("nan"),
                f"{prefix}_median": float("nan"),
                f"{prefix}_p95": float("nan"),
            }
        arr = np.asarray(vals, dtype=float)
        return {
            f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_median": float(np.median(arr)),
            f"{prefix}_p95": float(np.quantile(arr, 0.95)),
        }

    out = {"numeric_features_evaluated": float(len(ks_vals))}
    out.update(_summary(ks_vals, "ks"))
    out.update(_summary(wass_vals, "wasserstein"))
    out.update(_summary(wass_scaled_vals, "wasserstein_scaled"))
    return out


def _categorical_distribution_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    categorical_cols: list[str],
) -> dict[str, float]:
    tv_vals: list[float] = []
    js_vals: list[float] = []

    for col in categorical_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            continue
        r = real_df[col].fillna("__NAN__").astype(str).value_counts(normalize=True)
        s = synth_df[col].fillna("__NAN__").astype(str).value_counts(normalize=True)
        categories = sorted(set(r.index) | set(s.index))
        if not categories:
            continue

        r_probs = np.asarray([r.get(cat, 0.0) for cat in categories], dtype=float)
        s_probs = np.asarray([s.get(cat, 0.0) for cat in categories], dtype=float)

        tv = 0.5 * np.abs(r_probs - s_probs).sum()
        tv_vals.append(float(tv))

        js = float(jensenshannon(r_probs, s_probs, base=2))
        js_vals.append(js)

    def _summary(vals: list[float], prefix: str) -> dict[str, float]:
        if not vals:
            return {
                f"{prefix}_mean": float("nan"),
                f"{prefix}_median": float("nan"),
                f"{prefix}_p95": float("nan"),
            }
        arr = np.asarray(vals, dtype=float)
        return {
            f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_median": float(np.median(arr)),
            f"{prefix}_p95": float(np.quantile(arr, 0.95)),
        }

    out = {"categorical_features_evaluated": float(len(tv_vals))}
    out.update(_summary(tv_vals, "categorical_tv"))
    out.update(_summary(js_vals, "categorical_js"))
    return out


def _aggregate_seed_metrics(seed_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-seed metrics into mean/std/ci95 per key."""
    if not seed_metrics:
        return {}

    all_keys = set().union(*(m.keys() for m in seed_metrics))
    out: dict[str, float] = {}
    n_seeds = len(seed_metrics)
    out["seed_count"] = float(n_seeds)

    for key in sorted(all_keys):
        vals: list[float] = []
        for m in seed_metrics:
            v = m.get(key, np.nan)
            if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)):
                vals.append(float(v))
        if not vals:
            out[key] = float("nan")
            out[f"{key}_std"] = float("nan")
            out[f"{key}_ci95"] = float("nan")
            out[f"{key}_n"] = 0.0
            continue

        arr = np.asarray(vals, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci95 = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
        out[key] = mean
        out[f"{key}_std"] = std
        out[f"{key}_ci95"] = ci95
        out[f"{key}_n"] = float(len(arr))

    return out


def _evaluate_single_model(
    model_info: dict[str, Any],
    *,
    preprocessor: Any,
    all_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    real_sample: pd.DataFrame,
    real_with_replacement: bool,
    real_sample_seed: int,
    n_synth: int,
    synth_seeds: list[int],
    verbose: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_path": str(model_info["path"]),
        "nt": model_info["nt"],
        "n_noise": model_info["n_noise"],
    }

    t0 = time.time()
    artifact = _load_pickle(model_info["path"])
    model = (
        artifact["model"]
        if isinstance(artifact, dict) and "model" in artifact
        else artifact
    )
    row["model_type"] = (
        artifact.get("model_type", "unknown")
        if isinstance(artifact, dict)
        else "unknown"
    )

    if verbose:
        print(
            f"  -> Loading model: {model_info['path'].name} "
            f"(nt={model_info['nt']}, n_noise={model_info['n_noise']}, "
            f"model_type={row['model_type']})"
        )
    row["n_real"] = len(real_sample)
    row["n_synth"] = n_synth
    row["real_sample_seed"] = float(real_sample_seed)
    row["real_with_replacement"] = float(real_with_replacement)
    row["seed_count"] = float(len(synth_seeds))
    row["synth_seeds"] = ",".join(str(s) for s in synth_seeds)

    seed_metric_rows: list[dict[str, float]] = []
    for synth_seed in synth_seeds:
        if verbose:
            print(
                f"     Sampling synthetic hour-0 rows (n={n_synth:,}, seed={synth_seed})"
            )
        X_synth = _sample_synth_hour0(model, n_synth, synth_seed)
        synth_sample = preprocessor.inverse_transform(X_synth)[all_cols].reset_index(
            drop=True
        )

        metrics = compute_quality_metrics(real_sample, synth_sample)
        metrics.update(
            _numeric_distribution_metrics(real_sample, synth_sample, numeric_cols)
        )
        metrics.update(
            _categorical_distribution_metrics(
                real_sample, synth_sample, categorical_cols
            )
        )
        seed_metric_rows.append(metrics)

    row.update(_aggregate_seed_metrics(seed_metric_rows))

    row["elapsed_sec"] = time.time() - t0
    if verbose:
        print(
            "     Done in "
            f"{row['elapsed_sec']:.2f}s | "
            f"KS={_fmt_metric(row.get('avg_ks_stat'))}"
            f" ±{_fmt_metric(row.get('avg_ks_stat_ci95'))} | "
            f"CorrF={_fmt_metric(row.get('corr_frobenius'))}"
            f" ±{_fmt_metric(row.get('corr_frobenius_ci95'))} | "
            f"Wass(z)={_fmt_metric(row.get('wasserstein_scaled_mean'))}"
            f" ±{_fmt_metric(row.get('wasserstein_scaled_mean_ci95'))}"
        )
    return row


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = get_dataset_config(args.dataset)
    _install_legacy_pickle_shims()
    synth_seeds = _parse_seed_list(args.synth_seeds)

    models_dir = args.models_dir or cfg.sweep_models_dir
    hour0_data_path = args.hour0_data or cfg.hour0_data
    hour0_preprocessor_path = args.hour0_preprocessor or cfg.hour0_preprocessor

    if args.output_csv is not None:
        output_csv = args.output_csv
    else:
        output_csv = Path("results") / f"{args.dataset}_hour0_diagnostics.csv"

    if not models_dir.exists():
        raise SystemExit(f"ERROR: models directory not found: {models_dir}")
    if not hour0_data_path.exists():
        raise SystemExit(f"ERROR: hour-0 data not found: {hour0_data_path}")
    if not hour0_preprocessor_path.exists():
        raise SystemExit(
            f"ERROR: hour-0 preprocessor not found: {hour0_preprocessor_path}"
        )

    model_rows = _discover_hour0_models(models_dir, args.min_nt, args.min_nnoise)
    if not model_rows:
        raise SystemExit(
            f"ERROR: no matching hour-0 model files in {models_dir} "
            f"for nt>={args.min_nt}, n_noise>={args.min_nnoise}."
        )

    prep_payload = _load_pickle(hour0_preprocessor_path)
    preprocessor = prep_payload["preprocessor"]
    all_cols = list(prep_payload["all_cols"])
    numeric_cols = list(getattr(preprocessor, "numeric_cols", [])) + list(
        getattr(preprocessor, "binary_cols", [])
    )
    categorical_cols = list(getattr(preprocessor, "categorical_cols", []))

    real_hour0_df = pd.read_csv(hour0_data_path, low_memory=False)
    missing_cols = [c for c in all_cols if c not in real_hour0_df.columns]
    if missing_cols:
        raise SystemExit(
            "ERROR: hour-0 data is missing columns required by preprocessor: "
            f"{missing_cols[:10]}"
        )
    real_hour0_df = real_hour0_df[all_cols].reset_index(drop=True)
    rng_real = np.random.default_rng(args.seed)
    real_sample, real_idx, real_with_replacement = _sample_real_hour0(
        real_hour0_df, args.n_real, rng_real
    )

    print("=" * 80)
    print("Hour-0 model diagnostics (IID, no rollout, no TSTR)")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print("Legacy pickle compatibility shims: enabled")
    print(f"Models dir: {models_dir}")
    print("Model filename pattern: hour0_nt*_noise*.pkl (non-recursive)")
    print(
        f"Model filters: nt>={args.min_nt}, n_noise>={args.min_nnoise} "
        "(autoregressive artifacts are excluded by filename)"
    )
    print(f"Hour-0 data: {hour0_data_path}")
    print(f"Hour-0 preprocessor: {hour0_preprocessor_path}")
    print(f"Models selected: {len(model_rows)}")
    for i, info in enumerate(model_rows, start=1):
        print(
            f"  [{i:02d}] nt={info['nt']:<3d} n_noise={info['n_noise']:<3d} {info['path'].name}"
        )
    print(f"Per-model samples: synth={args.n_synth:,}, real={len(real_sample):,}")
    print(f"Fixed real sample seed: {args.seed}")
    print(f"Synthetic seeds reused for all models: {synth_seeds}")
    print(
        "Feature schema: "
        f"{len(all_cols)} total columns | "
        f"{len(numeric_cols)} numeric/binary | "
        f"{len(categorical_cols)} categorical"
    )
    print(
        f"Real hour-0 pool rows available: {len(real_hour0_df):,} "
        f"(sampling {'with' if real_with_replacement else 'without'} replacement)"
    )
    print(
        "Fixed real sample index preview: "
        f"{real_idx[:10].tolist() if len(real_idx) >= 10 else real_idx.tolist()}"
    )
    print("-" * 80)

    results: list[dict[str, Any]] = []
    interrupted = False
    for run_idx, info in enumerate(
        tqdm(model_rows, desc="Evaluating hour-0 models", unit="model"), start=1
    ):
        print(
            f"\n[{run_idx}/{len(model_rows)}] "
            f"Evaluating nt={info['nt']}, n_noise={info['n_noise']}"
        )
        try:
            result = _evaluate_single_model(
                info,
                preprocessor=preprocessor,
                all_cols=all_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                real_sample=real_sample,
                real_with_replacement=real_with_replacement,
                real_sample_seed=args.seed,
                n_synth=args.n_synth,
                synth_seeds=synth_seeds,
                verbose=True,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted by user (Ctrl+C). Saving partial results...")
            break
        except Exception as exc:
            result = {
                "model_path": str(info["path"]),
                "nt": info["nt"],
                "n_noise": info["n_noise"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"  !! Failed nt={info['nt']}, n_noise={info['n_noise']}: "
                f"{result['error']}"
            )
        else:
            results.append(result)
            continue
        results.append(result)

    df = pd.DataFrame(results)
    if "error" in df.columns:
        ok_df = df[df["error"].isna()] if df["error"].notna().any() else df
    else:
        ok_df = df
    if not ok_df.empty:
        sort_cols = [
            c for c in ("avg_ks_stat", "wasserstein_scaled_mean") if c in ok_df.columns
        ]
        if sort_cols:
            ok_df = ok_df.sort_values(sort_cols, ascending=[True, True]).reset_index(
                drop=True
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ok_df.to_csv(output_csv, index=False)
    print(f"\nSaved diagnostics: {output_csv}")
    if interrupted:
        print("Run status: interrupted early; output contains partial results.")
    print(
        f"Successful models: {len(ok_df):,} / {len(df):,} "
        f"(failed: {len(df) - len(ok_df):,})"
    )

    if "error" in df.columns and df["error"].notna().any():
        errors_csv = output_csv.with_name(output_csv.stem + "_errors.csv")
        df[df["error"].notna()].to_csv(errors_csv, index=False)
        print(f"Saved errors: {errors_csv}")

    if not ok_df.empty:
        if "elapsed_sec" in ok_df.columns:
            print(
                f"Average per-model runtime: "
                f"{float(ok_df['elapsed_sec'].mean()):.2f}s"
            )
        show_cols = [
            c
            for c in [
                "nt",
                "n_noise",
                "avg_ks_stat",
                "corr_frobenius",
                "range_violation_pct",
                "wasserstein_scaled_mean",
                "categorical_tv_mean",
                "elapsed_sec",
            ]
            if c in ok_df.columns
        ]
        print("\nTop configurations (lower is better):")
        print(ok_df[show_cols].head(10).to_string(index=False))
        print("\nBottom configurations (highest rank loss):")
        print(ok_df[show_cols].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
