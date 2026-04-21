"""Generator training helpers shared by sweep and backbone-comparison drivers.

One ``build_generator`` factory instantiates the three sweep backbones
(HS3F, ForestFlow, CTGAN) with consistent defaults, and two thin wrappers
(`train_autoregressive`, `train_hour0`) record wall-clock time alongside the
fit. Keeping these in a single module removes the cross-script import
(``from run_sweep import train_forest_flow``) the comparison driver used to
rely on.
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import numpy as np

from lexisflow.models import CTGANAdapter, ForestFlow, HS3F

from .config import SWEEP_XGB_PARAMS, format_time


DEFAULT_BATCH_SIZE = 5000
DEFAULT_N_JOBS = 8


def build_generator(
    model_type: str,
    nt: int,
    n_noise: int,
    n_jobs: int,
    ctgan_params: Optional[dict] = None,
    *,
    legacy_iterator: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Instantiate the requested generator backbone with shared defaults."""
    if model_type == "hs3f":
        return HS3F(
            nt=nt,
            n_noise=n_noise,
            n_jobs=n_jobs,
            random_state=42,
            solver="rk4",
            xgb_params=SWEEP_XGB_PARAMS,
        )
    if model_type == "ctgan":
        return CTGANAdapter(
            **(ctgan_params or {}),
            random_state=42,
        )
    return ForestFlow(
        nt=nt,
        n_noise=n_noise,
        n_jobs=n_jobs,
        random_state=42,
        use_data_iterator=legacy_iterator,
        batch_size=batch_size,
        xgb_params=SWEEP_XGB_PARAMS,
    )


def train_autoregressive(
    X_train: np.ndarray,
    X_cond_train: np.ndarray,
    nt: int,
    n_noise: int,
    feature_types: Optional[list[str]] = None,
    n_jobs: int = DEFAULT_N_JOBS,
    model_type: str = "hs3f",
    ctgan_params: Optional[dict] = None,
) -> tuple[object, float]:
    """Train the autoregressive (conditional) generator for a sweep cell."""
    start = time.time()
    model = build_generator(
        model_type, nt, n_noise, n_jobs, ctgan_params, legacy_iterator=True
    )
    print("    Fitting model...")
    sys.stdout.flush()
    model.fit(X_train, X_cond_train, feature_types=feature_types)
    print("    Fit complete, computing time...")
    sys.stdout.flush()
    return model, time.time() - start


def train_hour0(
    X_hour0: np.ndarray,
    nt: int,
    n_noise: int,
    feature_types: Optional[list[str]] = None,
    n_jobs: int = DEFAULT_N_JOBS,
    model_type: str = "hs3f",
    ctgan_params: Optional[dict] = None,
) -> tuple[object, float]:
    """Train the hour-0 IID generator (no condition matrix)."""
    print(f"  Training Hour-0 IID model (nt={nt}, n_noise={n_noise})...")
    start = time.time()
    model = build_generator(
        model_type, nt, n_noise, n_jobs, ctgan_params, legacy_iterator=True
    )
    model.fit(X_hour0, X_condition=None, feature_types=feature_types)
    train_time = time.time() - start
    print(f"    Hour-0 training: {format_time(train_time)}")
    return model, train_time


__all__ = [
    "build_generator",
    "train_autoregressive",
    "train_hour0",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_N_JOBS",
]
