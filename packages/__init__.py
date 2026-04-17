"""Synthetic data generation package for healthcare time series.

This package provides tools for generating synthetic electronic health records
using tree-based flow matching (ForestFlow, HS3F).

Main components:
- data: Data loading, cleaning, and preprocessing
- models: Generative backbones (ForestFlow, HS3F, CTGAN adapter)
- evaluation: Quality, trajectory, privacy, and TSTR metrics
"""

__version__ = "0.1.0"

from .data import TabularPreprocessor, prepare_autoregressive_data
from .models import ForestFlow, HS3F, sample_trajectory
from .evaluation import compute_quality_metrics

__all__ = [
    "TabularPreprocessor",
    "prepare_autoregressive_data",
    "ForestFlow",
    "HS3F",
    "sample_trajectory",
    "compute_quality_metrics",
]
