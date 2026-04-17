"""CTGAN adapter with ForestFlow-compatible fit/sample interface.

This adapter intentionally keeps a minimal API surface used by sweep scripts:
- fit(X_target, X_condition=None, feature_types=None)
- sample(n_samples, X_condition=None, random_state=None) -> X_target samples

For baseline parity, this implementation models the joint transformed table and
returns only target columns during sampling. Per-row conditioning input passed
to sample is accepted for interface compatibility but not enforced.
"""

from __future__ import annotations

import os

# Constrain BLAS/OMP threads BEFORE importing torch/ctgan so OpenMP picks these up
# at first initialisation. Fixes `OMP: Error #179 pthread_mutex_init` caused by OS
# thread/semaphore exhaustion on macOS after long multi-process joblib runs.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from ctgan import CTGAN
except ImportError:  # pragma: no cover - environment dependent optional dependency
    CTGAN = None

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent optional dependency
    torch = None


def _limit_thread_pools(num_threads: int = 1) -> None:
    """Reassert torch thread caps (env vars are set at module import time)."""
    num_threads = max(1, int(num_threads))
    if torch is not None:
        try:
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
        except (RuntimeError, AttributeError):
            pass


@dataclass
class _DiscreteColumnStats:
    """Metadata to robustly map sampled discrete values back to numerics."""

    numeric_fallback: float


class CTGANAdapter:
    """CTGAN baseline model wrapped for existing sweep/training pipelines."""

    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        generator_dim: tuple[int, ...] = (256, 256),
        discriminator_dim: tuple[int, ...] = (256, 256),
        embedding_dim: int = 128,
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_decay: float = 1e-6,
        pac: int = 10,
        log_frequency: bool = True,
        verbose: bool = False,
        sample_chunk_size: int = 10000,
        random_state: int = 42,
    ):
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.generator_dim = tuple(int(v) for v in generator_dim)
        self.discriminator_dim = tuple(int(v) for v in discriminator_dim)
        self.embedding_dim = int(embedding_dim)
        self.generator_lr = float(generator_lr)
        self.discriminator_lr = float(discriminator_lr)
        self.generator_decay = float(generator_decay)
        self.discriminator_decay = float(discriminator_decay)
        requested_pac = max(1, int(pac))
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.batch_size % requested_pac == 0:
            self.pac = requested_pac
        else:
            # CTGAN requires batch_size % pac == 0.
            # Pick the largest valid pac <= requested_pac.
            valid_pac = 1
            for candidate in range(min(requested_pac, self.batch_size), 0, -1):
                if self.batch_size % candidate == 0:
                    valid_pac = candidate
                    break
            self.pac = valid_pac
        self.log_frequency = bool(log_frequency)
        self.verbose = bool(verbose)
        self.sample_chunk_size = int(sample_chunk_size)
        if self.sample_chunk_size <= 0:
            raise ValueError("sample_chunk_size must be positive.")
        self.random_state = int(random_state)

        self.model_: object | None = None
        self.d_target_: int | None = None
        self.d_condition_: int = 0
        self.is_conditional_: bool = False
        # Public capability flag used by sweep sampling code.
        # This baseline accepts X_condition for interface compatibility but does
        # not enforce per-row conditioning during generation.
        self.uses_row_conditioning_ = False
        self.feature_types_: list[str] | None = None

        self._target_col_names: list[str] = []
        self._condition_col_names: list[str] = []
        self._all_col_names: list[str] = []
        self._discrete_col_names: list[str] = []
        self._discrete_stats: dict[str, _DiscreteColumnStats] = {}
        self._warned_condition_ignored = False

    def _build_column_names(
        self, d_target: int, d_condition: int
    ) -> tuple[list[str], list[str]]:
        target_cols = [f"t_{i}" for i in range(d_target)]
        condition_cols = [f"c_{i}" for i in range(d_condition)]
        return target_cols, condition_cols

    def fit(
        self,
        X: np.ndarray,
        X_condition: np.ndarray | None = None,
        feature_types: list[str] | None = None,
    ) -> "CTGANAdapter":
        """Fit CTGAN on transformed target/condition features."""
        if CTGAN is None:
            raise ImportError(
                "ctgan is required for CTGANAdapter. Install with `pip install ctgan`."
            )

        _limit_thread_pools(1)

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        self.d_target_ = int(X.shape[1])
        self.is_conditional_ = X_condition is not None

        if X_condition is not None:
            X_condition = np.asarray(X_condition, dtype=np.float32)
            if X_condition.ndim != 2:
                raise ValueError("X_condition must be a 2D array.")
            if len(X_condition) != len(X):
                raise ValueError("X and X_condition must have the same number of rows.")
            self.d_condition_ = int(X_condition.shape[1])
            X_full = np.hstack([X, X_condition]).astype(np.float32, copy=False)
        else:
            self.d_condition_ = 0
            X_full = X

        self._target_col_names, self._condition_col_names = self._build_column_names(
            self.d_target_,
            self.d_condition_,
        )
        self._all_col_names = self._target_col_names + self._condition_col_names

        if feature_types is not None:
            expected = self.d_target_ + self.d_condition_
            if len(feature_types) != expected:
                raise ValueError(
                    f"feature_types length {len(feature_types)} does not match expected {expected}."
                )
            self.feature_types_ = list(feature_types)
        else:
            self.feature_types_ = ["q"] * (self.d_target_ + self.d_condition_)

        df = pd.DataFrame(X_full, columns=self._all_col_names)

        self._discrete_col_names = [
            col_name
            for col_name, feature_type in zip(self._all_col_names, self.feature_types_)
            if feature_type == "c"
        ]

        self._discrete_stats = {}
        for col_name in self._discrete_col_names:
            vals = pd.to_numeric(df[col_name], errors="coerce")
            if vals.notna().any():
                fallback = float(np.nanmedian(vals.values))
            else:
                fallback = 0.0
            self._discrete_stats[col_name] = _DiscreteColumnStats(
                numeric_fallback=fallback
            )
            # CTGAN handles discrete columns robustly when represented as strings.
            df[col_name] = df[col_name].round().astype("Int64").astype(str)

        self.model_ = CTGAN(
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=self.generator_lr,
            generator_decay=self.generator_decay,
            discriminator_lr=self.discriminator_lr,
            discriminator_decay=self.discriminator_decay,
            batch_size=self.batch_size,
            epochs=self.epochs,
            pac=self.pac,
            log_frequency=self.log_frequency,
            verbose=self.verbose,
        )
        fit_start = time.time()
        if self.verbose:
            print(
                "    CTGAN fit: "
                f"rows={len(df):,}, columns={len(self._all_col_names)}, "
                f"discrete={len(self._discrete_col_names)}, epochs={self.epochs}, "
                f"batch_size={self.batch_size}, pac={self.pac}"
            )
        self.model_.fit(df, discrete_columns=self._discrete_col_names)
        if self.verbose:
            print(f"    CTGAN fit complete in {time.time() - fit_start:.1f}s")
        return self

    def sample(
        self,
        n_samples: int,
        X_condition: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Sample target features; condition rows are accepted but ignored."""
        if self.model_ is None or self.d_target_ is None:
            raise RuntimeError("CTGANAdapter is not fitted.")

        _limit_thread_pools(1)

        if X_condition is not None and not self._warned_condition_ignored:
            # Keep logging minimal while making baseline behavior explicit.
            print(
                "    Note: CTGAN baseline currently ignores per-row X_condition during sampling."
            )
            self._warned_condition_ignored = True

        if random_state is not None:
            np.random.seed(int(random_state))

        sample_start = time.time()
        n_samples_int = int(n_samples)
        if self.verbose and n_samples_int >= max(1000, self.batch_size):
            print(f"    CTGAN sample: drawing {n_samples_int:,} rows...")

        if n_samples_int > self.sample_chunk_size:
            chunks: list[pd.DataFrame] = []
            n_done = 0
            chunk_idx = 0
            while n_done < n_samples_int:
                chunk_idx += 1
                curr = min(self.sample_chunk_size, n_samples_int - n_done)
                if self.verbose:
                    print(
                        "      CTGAN sample chunk "
                        f"{chunk_idx}: {curr:,} rows "
                        f"(done {n_done:,}/{n_samples_int:,})"
                    )
                chunks.append(self.model_.sample(curr))
                n_done += curr
            sampled = pd.concat(chunks, axis=0, ignore_index=True)
        else:
            sampled = self.model_.sample(n_samples_int)
        sampled = sampled.reindex(columns=self._all_col_names)

        for col_name in self._all_col_names:
            if col_name in self._discrete_col_names:
                stats = self._discrete_stats[col_name]
                numeric = pd.to_numeric(sampled[col_name], errors="coerce")
                numeric = numeric.fillna(stats.numeric_fallback)
                sampled[col_name] = np.round(numeric).astype(np.float32)
            else:
                sampled[col_name] = pd.to_numeric(
                    sampled[col_name], errors="coerce"
                ).fillna(0.0)

        sampled_values = sampled.to_numpy(dtype=np.float32, copy=False)
        if self.verbose and int(n_samples) >= max(1000, self.batch_size):
            print(f"    CTGAN sample complete in {time.time() - sample_start:.1f}s")
        return sampled_values[:, : self.d_target_]
