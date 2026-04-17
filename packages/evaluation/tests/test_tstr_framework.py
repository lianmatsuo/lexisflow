from __future__ import annotations

import numpy as np
import pandas as pd

from packages.evaluation.tstr_framework import MortalityTask, VasopressorTask, LOSTask


def _build_synthetic_patient_rows(n_patients: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []

    for patient_id in range(n_patients):
        label = int(rng.random() < 0.35)
        los_bucket = patient_id % 3
        if los_bucket == 0:
            los_days = float(np.clip(rng.normal(1.3, 0.25), 0.5, 1.9))
        elif los_bucket == 1:
            los_days = float(np.clip(rng.normal(4.2, 0.8), 2.2, 6.8))
        else:
            los_days = float(np.clip(rng.normal(9.5, 1.5), 7.3, 14.0))
        length = int(max(4, min(24, np.ceil(los_days * 2.0))))
        gender = "F" if rng.random() < 0.5 else "M"
        patient_needs_vaso = int(label or rng.random() < 0.25)

        for hour in range(length):
            hr_trend = 95 + 0.7 * hour if label else 78 - 0.1 * hour
            map_trend = 62 - 0.4 * hour if label else 75 - 0.1 * hour
            vaso_now = int(patient_needs_vaso and hour >= max(1, length // 3))
            rows.append(
                {
                    "subject_id": patient_id,
                    "hours_in": hour,
                    "hospital_expire_flag": label,
                    "heart_rate": float(hr_trend + rng.normal(0, 2.5)),
                    "map": float(map_trend + rng.normal(0, 2.0)),
                    "lactate": float((2.8 if label else 1.5) + rng.normal(0, 0.4)),
                    "vent": int(label and hour > 2),
                    "vaso": vaso_now,
                    "los_icu": los_days,
                    "gender": gender,
                }
            )

    return pd.DataFrame(rows)


def test_mortality_sequence_tstr_runs_end_to_end():
    synth_df = _build_synthetic_patient_rows(n_patients=70, seed=11)
    real_df = _build_synthetic_patient_rows(n_patients=90, seed=99)

    task = MortalityTask(
        random_state=7,
        use_sequence_model=True,
        sequence_max_patients=None,
    )

    metrics = task.evaluate(
        synth_df=synth_df, real_df=real_df, test_size=0.3, verbose=False
    )

    expected_keys = {
        "synth_accuracy",
        "synth_balanced_accuracy",
        "synth_f1",
        "synth_roc_auc",
        "real_accuracy",
        "real_balanced_accuracy",
        "real_f1",
        "real_roc_auc",
    }
    assert expected_keys.issubset(metrics.keys())
    assert np.isfinite(metrics["synth_accuracy"])
    assert np.isfinite(metrics["synth_balanced_accuracy"])
    assert np.isfinite(metrics["real_accuracy"])
    assert np.isfinite(metrics["real_balanced_accuracy"])
    assert np.isfinite(metrics["synth_f1"])
    assert np.isfinite(metrics["real_f1"])


def test_vasopressor_sequence_tstr_runs_end_to_end():
    synth_df = _build_synthetic_patient_rows(n_patients=80, seed=17)
    real_df = _build_synthetic_patient_rows(n_patients=100, seed=33)

    task = VasopressorTask(
        random_state=9,
        use_sequence_model=True,
        sequence_max_patients=None,
    )
    metrics = task.evaluate(
        synth_df=synth_df, real_df=real_df, test_size=0.3, verbose=False
    )

    assert np.isfinite(metrics["synth_accuracy"])
    assert np.isfinite(metrics["synth_balanced_accuracy"])
    assert np.isfinite(metrics["real_accuracy"])
    assert np.isfinite(metrics["real_balanced_accuracy"])
    assert np.isfinite(metrics["synth_f1"])
    assert np.isfinite(metrics["real_f1"])


def test_los_sequence_tstr_runs_end_to_end():
    synth_df = _build_synthetic_patient_rows(n_patients=90, seed=23)
    real_df = _build_synthetic_patient_rows(n_patients=110, seed=44)

    task = LOSTask(
        random_state=5,
        use_sequence_model=True,
        sequence_max_patients=None,
    )
    metrics = task.evaluate(
        synth_df=synth_df, real_df=real_df, test_size=0.3, verbose=False
    )

    assert np.isfinite(metrics["synth_accuracy"])
    assert np.isfinite(metrics["synth_balanced_accuracy"])
    assert np.isfinite(metrics["real_accuracy"])
    assert np.isfinite(metrics["real_balanced_accuracy"])
    assert np.isfinite(metrics["synth_macro_f1"])
    assert np.isfinite(metrics["real_macro_f1"])


def test_mortality_row_tstr_mode_still_supported():
    synth_df = _build_synthetic_patient_rows(n_patients=60, seed=123).drop(
        columns=["subject_id", "hours_in"]
    )
    real_df = _build_synthetic_patient_rows(n_patients=80, seed=321).drop(
        columns=["subject_id", "hours_in"]
    )

    task = MortalityTask(random_state=21, use_sequence_model=False)
    metrics = task.evaluate(
        synth_df=synth_df, real_df=real_df, test_size=0.3, verbose=False
    )

    assert np.isfinite(metrics["synth_accuracy"])
    assert np.isfinite(metrics["synth_balanced_accuracy"])
    assert np.isfinite(metrics["real_accuracy"])
    assert np.isfinite(metrics["real_balanced_accuracy"])
