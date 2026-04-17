"""Forest-Flow generative models for time series synthesis.

This package contains:
- ForestFlow: Main conditional flow matching model with XGBoost
- FlowMatchingDataIterator: Memory-efficient data iterator
- Trajectory sampling functions for autoregressive generation
"""

from .forest_flow import ForestFlow
from .hs3f import HS3F
from .ctgan_adapter import CTGANAdapter
from .iterator import FlowMatchingDataIterator
from .sampling import (
    sample_trajectory,
    sample_trajectory_with_initial_sampling,
    prepare_training_data_from_trajectories,
)

__all__ = [
    # Core model
    "ForestFlow",
    "HS3F",
    "CTGANAdapter",
    "FlowMatchingDataIterator",
    # Sampling
    "sample_trajectory",
    "sample_trajectory_with_initial_sampling",
    "prepare_training_data_from_trajectories",
]
