"""Hour-0 and autoregressive data loading shared by sweep drivers.

Both ``run_sweep.py`` and ``run_backbone_comparison.py`` reload the
same preprocessor artefacts and transformed autoregressive cache. Keeping a
single implementation here guarantees both drivers train on byte-identical
inputs and share the cache signature logic.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cache import build_cache_signature, load_transformed_cache, save_transformed_cache


DEFAULT_HOUR0_DATA_PATH = Path("data/processed/hour0_data.csv")
DEFAULT_HOUR0_PREPROCESSOR_PATH = Path("artifacts/hour0_preprocessor.pkl")
DEFAULT_AR_PREPROCESSOR_PATH = Path("artifacts/preprocessor_full.pkl")
DEFAULT_AR_CSV_PATH = Path("data/processed/autoregressive_data.csv")
DEFAULT_REAL_TEST_CSV_PATH = Path("data/processed/real_test.csv")
DEFAULT_REAL_HOLDOUT_CSV_PATH = Path("data/processed/real_holdout.csv")
DEFAULT_TRANSFORMED_CACHE_DIR = Path("artifacts/sweep_cache/transformed_autoregressive")
DEFAULT_QUALITY_SAMPLE_SIZE = 50000


@dataclass
class Hour0Inputs:
    X_hour0: np.ndarray
    preprocessor: object
    preprocessor_path: str
    all_cols: list[str]
    feature_types: list[str]


@dataclass
class AutoregressiveInputs:
    preprocessor: object
    target_indices: list[int]
    condition_indices: list[int]
    feature_types: list[str]
    X_train: np.ndarray
    X_cond_train: np.ndarray
    real_quality_df: pd.DataFrame
    real_holdout_df: pd.DataFrame | None
    n_rows: int
    real_test_path: str = str(DEFAULT_REAL_TEST_CSV_PATH)
    all_cols: list[str] = field(default_factory=list)


def load_hour0_inputs(
    train_rows: int,
    *,
    data_path: Path = DEFAULT_HOUR0_DATA_PATH,
    preprocessor_path: Path = DEFAULT_HOUR0_PREPROCESSOR_PATH,
    random_state: int = 42,
) -> Hour0Inputs:
    """Load the hour-0 preprocessor and subsample the training matrix."""
    if not data_path.exists():
        print(f"\nError: {data_path} not found!")
        print("Run: uv run python scripts/prepare_hour0.py")
        sys.exit(1)
    if not preprocessor_path.exists():
        print(f"\nError: {preprocessor_path} not found!")
        print("Run: uv run python scripts/fit_hour0_preprocessor.py")
        sys.exit(1)

    print(f"\nLoading hour-0 preprocessor from {preprocessor_path}...")
    with open(preprocessor_path, "rb") as f:
        payload = pickle.load(f)
    preprocessor = payload["preprocessor"]
    all_cols = payload["all_cols"]
    print(
        f"  ✓ Hour-0 preprocessor: {len(all_cols)} features → "
        f"{preprocessor.n_features} encoded"
    )

    print(f"\nLoading hour-0 data from {data_path}...")
    df_hour0 = pd.read_csv(data_path, low_memory=False)
    print(f"  ✓ Loaded: {df_hour0.shape[0]:,} rows × {df_hour0.shape[1]} columns")

    X_full = preprocessor.transform(df_hour0[all_cols])
    rows_available = len(X_full)
    take = min(train_rows, rows_available)
    print(
        "  Subsampling hour-0 to match autoregressive sample size: "
        f"{take:,} / {rows_available:,}"
    )
    rng = np.random.default_rng(random_state)
    idx = rng.choice(rows_available, size=take, replace=False)
    idx.sort()
    X_hour0 = X_full[idx]
    print(f"  ✓ Hour-0 training matrix: {X_hour0.shape}")

    feature_types = preprocessor.get_feature_types()
    return Hour0Inputs(
        X_hour0=X_hour0,
        preprocessor=preprocessor,
        preprocessor_path=str(preprocessor_path),
        all_cols=all_cols,
        feature_types=feature_types,
    )


def load_autoregressive_inputs(
    train_rows: int,
    *,
    refresh_cache: bool = False,
    preprocessor_path: Path = DEFAULT_AR_PREPROCESSOR_PATH,
    csv_path: Path = DEFAULT_AR_CSV_PATH,
    real_test_csv_path: Path = DEFAULT_REAL_TEST_CSV_PATH,
    real_holdout_csv_path: Path = DEFAULT_REAL_HOLDOUT_CSV_PATH,
    transformed_cache_dir: Path = DEFAULT_TRANSFORMED_CACHE_DIR,
    quality_sample_size: int = DEFAULT_QUALITY_SAMPLE_SIZE,
    random_state: int = 42,
    show_progress: bool = True,
) -> AutoregressiveInputs:
    """Load preprocessor, transform autoregressive dataset, and subsample rows."""
    if not preprocessor_path.exists():
        print(f"\nError: {preprocessor_path} not found!")
        print("▶ Run: uv run python scripts/fit_autoregressive_preprocessor.py")
        sys.exit(1)

    with open(preprocessor_path, "rb") as f:
        payload = pickle.load(f)
    preprocessor = payload["preprocessor"]
    target_cols = payload["target_cols"]
    condition_cols = payload["condition_cols"]
    all_cols = payload["all_cols"]
    target_indices = payload.get("target_indices")
    condition_indices = payload.get("condition_indices")
    if target_indices is None or condition_indices is None:
        target_indices, condition_indices = preprocessor.split_indices(
            target_cols, condition_cols
        )

    n_target = len(target_indices)
    full_feature_types = preprocessor.get_feature_types()
    feature_types = [
        full_feature_types[i] for i in (target_indices + condition_indices)
    ]
    print(
        f"  ✓ Feature types: {feature_types.count('q')} quantitative, "
        f"{feature_types.count('c')} categorical"
    )
    if target_indices != list(range(n_target)):
        print(
            "  ⚠️  Target columns are non-contiguous in transformed space; "
            "using index-based target/condition split"
        )

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        print("Run scripts/prepare_autoregressive.py first.")
        sys.exit(1)

    cache_sig = build_cache_signature(
        csv_path,
        all_cols,
        n_target,
        target_indices,
        condition_indices,
        preprocessor_path=preprocessor_path,
    )
    cached = (
        None
        if refresh_cache
        else load_transformed_cache(transformed_cache_dir, cache_sig)
    )

    if cached is not None:
        X_target_full, X_condition_full, real_quality_df, n_rows = cached
        print(f"\nLoaded transformed cache from {transformed_cache_dir}")
        print(f"  Cached arrays: {X_target_full.shape}, {X_condition_full.shape}")
        print(f"  Cached quality sample: {real_quality_df.shape}")
    else:
        X_target_full, X_condition_full, real_quality_df, n_rows = (
            _rebuild_transformed_cache(
                csv_path=csv_path,
                real_test_csv_path=real_test_csv_path,
                preprocessor=preprocessor,
                all_cols=all_cols,
                target_indices=target_indices,
                condition_indices=condition_indices,
                cache_dir=transformed_cache_dir,
                cache_sig=cache_sig,
                quality_sample_size=quality_sample_size,
                show_progress=show_progress,
            )
        )

    # Load privacy holdout from disk — never falls through the transformed cache
    # because it is not derived from the training matrix.
    real_holdout_df: pd.DataFrame | None = None
    if real_holdout_csv_path.exists():
        real_holdout_raw = pd.read_csv(real_holdout_csv_path, low_memory=False)
        avail_cols = [c for c in all_cols if c in real_holdout_raw.columns]
        real_holdout_df = real_holdout_raw[avail_cols].reset_index(drop=True)
        print(
            f"\nLoaded privacy holdout from {real_holdout_csv_path}: {real_holdout_df.shape}"
        )
    else:
        print(
            f"\n  Warning: {real_holdout_csv_path} not found. "
            "Privacy metrics will use an internal split of the quality sample "
            "(potentially overlapping with training data). "
            "Re-run scripts/prepare_autoregressive.py to generate a true holdout."
        )

    print(f"\nSubsampling {train_rows:,} / {n_rows:,} rows for training...")
    rng = np.random.default_rng(random_state)
    train_idx = rng.choice(n_rows, size=min(train_rows, n_rows), replace=False)
    train_idx.sort()
    X_train = np.ascontiguousarray(X_target_full[train_idx])
    X_cond_train = np.ascontiguousarray(X_condition_full[train_idx])
    print(f"  X_train: {X_train.shape}")
    print(f"  X_cond_train: {X_cond_train.shape}")

    # Release the mmap-backed full arrays; only the materialized subsamples
    # are needed downstream.
    del X_target_full, X_condition_full

    real_test_path = (
        str(real_test_csv_path) if real_test_csv_path.exists() else str(csv_path)
    )
    if not real_test_csv_path.exists():
        print(
            f"\n  Warning: {real_test_csv_path} not found. "
            "TSTR will read from the training CSV — results will be optimistic. "
            "Re-run scripts/prepare_autoregressive.py to generate a disjoint test split."
        )

    return AutoregressiveInputs(
        preprocessor=preprocessor,
        target_indices=target_indices,
        condition_indices=condition_indices,
        feature_types=feature_types,
        X_train=X_train,
        X_cond_train=X_cond_train,
        real_quality_df=real_quality_df.reset_index(drop=True),
        real_holdout_df=real_holdout_df,
        n_rows=n_rows,
        real_test_path=real_test_path,
        all_cols=all_cols,
    )


def _rebuild_transformed_cache(
    *,
    csv_path: Path,
    real_test_csv_path: Path,
    preprocessor: object,
    all_cols: list[str],
    target_indices: list[int],
    condition_indices: list[int],
    cache_dir: Path,
    cache_sig: dict,
    quality_sample_size: int,
    show_progress: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    prep_bar: Optional[tqdm] = None
    if show_progress:
        prep_bar = tqdm(total=4, desc="AR data preparation", unit="step")
    print(f"\nLoading and transforming full dataset from {csv_path}...")
    print("  (This will take a few minutes for 2.2M rows)")
    df_full = pd.read_csv(csv_path, low_memory=False)
    n_rows = len(df_full)
    print(f"  Loaded: {n_rows:,} rows")
    if prep_bar is not None:
        prep_bar.update(1)

    X_full = preprocessor.transform(df_full[all_cols])
    X_target_full = X_full[:, target_indices]
    X_condition_full = X_full[:, condition_indices]
    print(f"  Transformed: {X_target_full.shape}, {X_condition_full.shape}")
    if prep_bar is not None:
        prep_bar.update(1)

    # Load real_quality_df from the patient-disjoint test split so that fidelity
    # metrics (KS, Frobenius correlation, range violations) are not measured
    # against training rows.  Falls back to sampling from training data if the
    # test CSV is absent (e.g. user hasn't re-run prepare_autoregressive.py).
    if real_test_csv_path.exists():
        real_quality_raw = pd.read_csv(real_test_csv_path, low_memory=False)
        avail_cols = [c for c in all_cols if c in real_quality_raw.columns]
        if len(real_quality_raw) > quality_sample_size:
            real_quality_df = (
                real_quality_raw[avail_cols]
                .sample(quality_sample_size, random_state=42)
                .reset_index(drop=True)
            )
        else:
            real_quality_df = real_quality_raw[avail_cols].reset_index(drop=True)
        print(
            f"  Loaded real_quality_df from {real_test_csv_path}: {real_quality_df.shape}"
        )
    else:
        print(
            f"  Warning: {real_test_csv_path} not found — "
            "sampling real_quality_df from training data (results will be optimistic)."
        )
        quality_idx = np.random.default_rng(42).choice(
            n_rows, size=min(quality_sample_size, n_rows), replace=False
        )
        quality_idx.sort()
        real_quality_df = df_full.iloc[quality_idx][all_cols].reset_index(drop=True)
    if prep_bar is not None:
        prep_bar.update(1)

    print(f"  Saving transformed cache to {cache_dir} ...")
    save_transformed_cache(
        cache_dir,
        cache_sig,
        X_target_full,
        X_condition_full,
        real_quality_df,
        n_rows,
    )
    print("  ✓ Cache saved")
    if prep_bar is not None:
        prep_bar.update(1)
        prep_bar.close()
    return X_target_full, X_condition_full, real_quality_df, n_rows


__all__ = [
    "Hour0Inputs",
    "AutoregressiveInputs",
    "load_hour0_inputs",
    "load_autoregressive_inputs",
    "DEFAULT_HOUR0_DATA_PATH",
    "DEFAULT_HOUR0_PREPROCESSOR_PATH",
    "DEFAULT_AR_PREPROCESSOR_PATH",
    "DEFAULT_AR_CSV_PATH",
    "DEFAULT_REAL_TEST_CSV_PATH",
    "DEFAULT_REAL_HOLDOUT_CSV_PATH",
    "DEFAULT_TRANSFORMED_CACHE_DIR",
]
