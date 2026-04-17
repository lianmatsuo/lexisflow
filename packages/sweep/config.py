"""Configuration constants shared by sweep and backbone-comparison drivers."""

from __future__ import annotations


# Trajectory length for autoregressive generation (hours)
SEQUENCE_TIMESTEPS = 48

# Trajectory-sampling RNG seeds used to aggregate TSTR/quality/privacy metrics
# per sweep cell (mean + std + CI95 across seeds).
TSTR_TRAJECTORY_SAMPLING_SEEDS = (42, 11, 50)

# XGBoost parameters used by both hour-0 and autoregressive models during sweep
SWEEP_XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# CTGAN baseline defaults (kept modest for laptop-friendliness)
CTGAN_DEFAULT_EPOCHS = 300
CTGAN_DEFAULT_BATCH_SIZE = 500
CTGAN_DEFAULT_GENERATOR_DIM = (256, 256)
CTGAN_DEFAULT_DISCRIMINATOR_DIM = (256, 256)
CTGAN_DEFAULT_EMBEDDING_DIM = 128
CTGAN_DEFAULT_LR = 2e-4
CTGAN_DEFAULT_DECAY = 1e-6
CTGAN_DEFAULT_PAC = 10


def format_time(seconds: float) -> str:
    """Render a duration as e.g. ``12.3s``, ``4.5m``, or ``1.2h``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


__all__ = [
    "SEQUENCE_TIMESTEPS",
    "TSTR_TRAJECTORY_SAMPLING_SEEDS",
    "SWEEP_XGB_PARAMS",
    "CTGAN_DEFAULT_EPOCHS",
    "CTGAN_DEFAULT_BATCH_SIZE",
    "CTGAN_DEFAULT_GENERATOR_DIM",
    "CTGAN_DEFAULT_DISCRIMINATOR_DIM",
    "CTGAN_DEFAULT_EMBEDDING_DIM",
    "CTGAN_DEFAULT_LR",
    "CTGAN_DEFAULT_DECAY",
    "CTGAN_DEFAULT_PAC",
    "format_time",
]
