"""Dataset presets for sweep orchestration and path defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class SweepDefaults:
    nt_values: tuple[int, ...]
    noise_values: tuple[int, ...]
    max_train_rows: int
    n_synth_samples: int
    privacy_max_rows: int = 2000


@dataclass(frozen=True)
class SplitConfig:
    test_fraction: float
    holdout_fraction: float
    shuffle_seed: int


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    script_dir: Path
    hour0_data: Path
    hour0_preprocessor: Path
    autoregressive_data: Path
    real_test: Path
    real_holdout: Path
    autoregressive_preprocessor: Path
    transformed_cache_dir: Path
    results_csv: Path
    sweep_models_dir: Path
    sweep_profiles: Mapping[str, SweepDefaults]
    split: SplitConfig

    @property
    def reset_targets(self) -> tuple[Path, ...]:
        """Artifacts recreated by running the full sweep pipeline."""
        return (
            self.hour0_data,
            self.autoregressive_data,
            self.real_test,
            self.real_holdout,
            self.hour0_preprocessor,
            self.autoregressive_preprocessor,
            self.transformed_cache_dir,
            self.sweep_models_dir,
            self.results_csv,
        )

    def get_sweep_defaults(self, profile: str = "full") -> SweepDefaults:
        try:
            return self.sweep_profiles[profile]
        except KeyError as exc:
            known = ", ".join(sorted(self.sweep_profiles))
            raise ValueError(
                f"Unknown sweep profile '{profile}' for dataset '{self.name}'. "
                f"Expected one of: {known}"
            ) from exc


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "mimic": DatasetConfig(
        name="mimic",
        script_dir=Path("scripts/mimic"),
        hour0_data=Path("data/processed/hour0_data.csv"),
        hour0_preprocessor=Path("artifacts/hour0_preprocessor.pkl"),
        autoregressive_data=Path("data/processed/autoregressive_data.csv"),
        real_test=Path("data/processed/real_test.csv"),
        real_holdout=Path("data/processed/real_holdout.csv"),
        autoregressive_preprocessor=Path("artifacts/preprocessor_full.pkl"),
        transformed_cache_dir=Path("artifacts/sweep_cache/transformed_autoregressive"),
        results_csv=Path("results/sweep_results.csv"),
        sweep_models_dir=Path("artifacts/sweep"),
        sweep_profiles={
            "full": SweepDefaults(
                nt_values=(1, 2, 3, 5, 8, 13, 21, 34),
                noise_values=(1, 2, 3, 5, 8, 13, 21, 34),
                max_train_rows=100000,
                n_synth_samples=100000,
                privacy_max_rows=2000,
            ),
            "smoke": SweepDefaults(
                nt_values=(1, 3),
                noise_values=(1, 3),
                max_train_rows=10000,
                n_synth_samples=5000,
                privacy_max_rows=1000,
            ),
        },
        split=SplitConfig(test_fraction=0.10, holdout_fraction=0.10, shuffle_seed=0),
    ),
    "challenge2012": DatasetConfig(
        name="challenge2012",
        script_dir=Path("scripts/challenge2012"),
        hour0_data=Path("data/challenge2012/processed/hour0_data.csv"),
        hour0_preprocessor=Path("artifacts/challenge2012/hour0_preprocessor.pkl"),
        autoregressive_data=Path(
            "data/challenge2012/processed/autoregressive_data.csv"
        ),
        real_test=Path("data/challenge2012/processed/real_test.csv"),
        real_holdout=Path("data/challenge2012/processed/real_holdout.csv"),
        autoregressive_preprocessor=Path(
            "artifacts/challenge2012/autoregressive_preprocessor.pkl"
        ),
        transformed_cache_dir=Path(
            "artifacts/challenge2012/sweep_cache/transformed_autoregressive"
        ),
        results_csv=Path("results/challenge2012_sweep_results.csv"),
        sweep_models_dir=Path("artifacts/challenge2012/sweep"),
        sweep_profiles={
            "full": SweepDefaults(
                nt_values=(1, 2, 3, 5, 8, 13, 21),
                noise_values=(1, 2, 3, 5, 8, 13, 21),
                max_train_rows=100000,
                n_synth_samples=50000,
                privacy_max_rows=2000,
            ),
            "smoke": SweepDefaults(
                nt_values=(1, 3),
                noise_values=(1, 3),
                max_train_rows=20000,
                n_synth_samples=10000,
                privacy_max_rows=1000,
            ),
        },
        split=SplitConfig(test_fraction=0.10, holdout_fraction=0.10, shuffle_seed=0),
    ),
}


def get_dataset_config(dataset: str) -> DatasetConfig:
    try:
        return DATASET_CONFIGS[dataset]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(
            f"Unknown dataset '{dataset}'. Expected one of: {known}"
        ) from exc
