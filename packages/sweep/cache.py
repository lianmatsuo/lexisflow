"""Transformed-data caching for the sweep pipeline.

The sweep repeatedly trains on the same preprocessed autoregressive matrix.
Rather than reloading and re-transforming the 2M+ row CSV every invocation,
we cache the transformed target/condition arrays plus a held-out real-data
sample used for quality metrics, keyed on a cheap signature derived from the
source CSV and preprocessor column layout.
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def build_cache_signature(
    csv_path: Path,
    all_cols: list[str],
    n_target: int,
    target_indices: list[int],
    condition_indices: list[int],
) -> dict:
    """Build a lightweight signature for transformed-data cache compatibility."""
    st = csv_path.stat()
    cols_hash = hashlib.sha256("||".join(all_cols).encode("utf-8")).hexdigest()
    target_idx_hash = hashlib.sha256(
        ",".join(str(i) for i in target_indices).encode("utf-8")
    ).hexdigest()
    condition_idx_hash = hashlib.sha256(
        ",".join(str(i) for i in condition_indices).encode("utf-8")
    ).hexdigest()
    return {
        "csv_path": str(csv_path),
        "csv_size": int(st.st_size),
        "csv_mtime_ns": int(st.st_mtime_ns),
        "n_cols": len(all_cols),
        "n_target": int(n_target),
        "cols_hash": cols_hash,
        "target_idx_hash": target_idx_hash,
        "condition_idx_hash": condition_idx_hash,
    }


_REQUIRED_SIG_KEYS = (
    "csv_path",
    "csv_size",
    "csv_mtime_ns",
    "n_cols",
    "n_target",
    "cols_hash",
    "target_idx_hash",
    "condition_idx_hash",
)


def load_transformed_cache(
    cache_dir: Path,
    expected_sig: dict,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int] | None:
    """Return cached (X_target, X_condition, real_quality_df, n_rows) or None."""
    meta_path = cache_dir / "metadata.pkl"
    x_target_path = cache_dir / "X_target_full.npy"
    x_cond_path = cache_dir / "X_condition_full.npy"
    quality_path = cache_dir / "real_quality_df.pkl"

    if not (
        meta_path.exists()
        and x_target_path.exists()
        and x_cond_path.exists()
        and quality_path.exists()
    ):
        return None

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    sig = meta.get("signature", {})
    if any(sig.get(k) != expected_sig.get(k) for k in _REQUIRED_SIG_KEYS):
        return None

    X_target_full = np.load(x_target_path, mmap_mode="r")
    X_condition_full = np.load(x_cond_path, mmap_mode="r")
    real_quality_df = pd.read_pickle(quality_path)
    n_rows = int(meta["n_rows"])
    return X_target_full, X_condition_full, real_quality_df, n_rows


def save_transformed_cache(
    cache_dir: Path,
    signature: dict,
    X_target_full: np.ndarray,
    X_condition_full: np.ndarray,
    real_quality_df: pd.DataFrame,
    n_rows: int,
) -> None:
    """Persist transformed arrays and quality sample for future sweeps."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "X_target_full.npy", X_target_full)
    np.save(cache_dir / "X_condition_full.npy", X_condition_full)
    real_quality_df.to_pickle(cache_dir / "real_quality_df.pkl")
    metadata = {
        "signature": signature,
        "n_rows": int(n_rows),
        "saved_at": datetime.now().isoformat(),
    }
    with open(cache_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


__all__ = [
    "build_cache_signature",
    "load_transformed_cache",
    "save_transformed_cache",
]
