"""Shared sweep pipeline helpers.

Modules in ``packages.sweep`` are the single source of truth for:

- :mod:`config`: sweep-wide constants (XGB params, CTGAN defaults, sequence length)
- :mod:`cache`: transformed-data cache keyed on preprocessor signature
- :mod:`schema`: declarative result-row schema and CSV I/O
- :mod:`metrics`: seed-level uncertainty aggregation
- :mod:`training`: generator factory + hour-0 / autoregressive trainers
- :mod:`generation`: autoregressive sampling and inverse-transform helpers
- :mod:`evaluation`: per-seed TSTR / quality / privacy / trajectory evaluation
- :mod:`data_prep`: preprocessor + transformed-cache loading
- :mod:`cli`: argparse helpers (CTGAN options, integer-list parsing)

Drivers (``scripts/run_sweep.py``, ``scripts/run_matched_backbone_comparison.py``)
import from here rather than duplicating these helpers.
"""

from .config import (
    SEQUENCE_TIMESTEPS,
    TSTR_TRAJECTORY_SAMPLING_SEEDS,
    SWEEP_XGB_PARAMS,
    CTGAN_DEFAULT_EPOCHS,
    CTGAN_DEFAULT_BATCH_SIZE,
    CTGAN_DEFAULT_GENERATOR_DIM,
    CTGAN_DEFAULT_DISCRIMINATOR_DIM,
    CTGAN_DEFAULT_EMBEDDING_DIM,
    CTGAN_DEFAULT_LR,
    CTGAN_DEFAULT_DECAY,
    CTGAN_DEFAULT_PAC,
    format_time,
)
from .cache import (
    build_cache_signature,
    load_transformed_cache,
    save_transformed_cache,
)
from .schema import (
    SweepField,
    SWEEP_RESULT_COLUMNS,
    SEED_STAT_METRIC_MAP,
    build_result_row,
    build_error_row,
    load_completed_runs,
    append_result,
    ensure_results_schema,
)
from .metrics import (
    metric_stats,
    average_sweep_metrics,
    compute_seed_uncertainty,
)
from .training import (
    build_generator,
    train_autoregressive,
    train_hour0,
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_JOBS,
)
from .generation import (
    BINARY_OUTPUT_COLUMNS,
    ID_COLUMNS_TO_DROP,
    generate_synthetic_data,
    create_flat_dataframe,
    drop_id_columns,
)
from .evaluation import evaluate_tstr
from .data_prep import (
    Hour0Inputs,
    AutoregressiveInputs,
    load_hour0_inputs,
    load_autoregressive_inputs,
)
from .cli import (
    parse_int_list,
    parse_int_tuple,
    add_ctgan_arguments,
    ctgan_params_from_args,
    format_ctgan_params,
)

__all__ = [
    # config
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
    # cache
    "build_cache_signature",
    "load_transformed_cache",
    "save_transformed_cache",
    # schema
    "SweepField",
    "SWEEP_RESULT_COLUMNS",
    "SEED_STAT_METRIC_MAP",
    "build_result_row",
    "build_error_row",
    "load_completed_runs",
    "append_result",
    "ensure_results_schema",
    # metrics
    "metric_stats",
    "average_sweep_metrics",
    "compute_seed_uncertainty",
    # training
    "build_generator",
    "train_autoregressive",
    "train_hour0",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_N_JOBS",
    # generation
    "BINARY_OUTPUT_COLUMNS",
    "ID_COLUMNS_TO_DROP",
    "generate_synthetic_data",
    "create_flat_dataframe",
    "drop_id_columns",
    # evaluation
    "evaluate_tstr",
    # data_prep
    "Hour0Inputs",
    "AutoregressiveInputs",
    "load_hour0_inputs",
    "load_autoregressive_inputs",
    # cli
    "parse_int_list",
    "parse_int_tuple",
    "add_ctgan_arguments",
    "ctgan_params_from_args",
    "format_ctgan_params",
]
