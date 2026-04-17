"""Unit tests for privacy metrics."""

import numpy as np
import pandas as pd

from packages.evaluation.privacy_metrics import (
    compute_distance_to_closest_record,
    compute_dcr_overfitting_protection,
    compute_domias_like_membership_inference,
    compute_privacy_metrics,
)


def _make_real_train(n: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(20, 90, size=n),
            "heart_rate": rng.normal(82, 12, size=n),
            "spo2": rng.normal(96, 2, size=n),
            "sex": rng.choice(["M", "F"], size=n),
        }
    )


def _make_holdout(n: int = 120, seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(25, 95, size=n),
            "heart_rate": rng.normal(95, 10, size=n),
            "spo2": rng.normal(92, 2, size=n),
            "sex": rng.choice(["M", "F"], size=n),
        }
    )


def test_dcr_exact_duplicates_have_zero_distance():
    """When synthetic rows clone real rows, DCR median should be near zero."""
    real_train = _make_real_train(n=80, seed=7)
    synthetic = real_train.sample(n=40, random_state=7).reset_index(drop=True)

    metrics = compute_distance_to_closest_record(
        real_train_df=real_train,
        synthetic_df=synthetic,
        max_rows=None,
        random_state=7,
    )

    assert metrics["dcr_median"] <= 1e-12
    assert metrics["exact_match_rate"] > 0.95


def test_dcr_overfitting_protection_penalizes_train_clones():
    """Overfitting score should drop when synthetic data copies training rows."""
    real_train = _make_real_train(n=120, seed=13)
    holdout = _make_holdout(n=120, seed=17)
    synthetic = real_train.sample(n=80, random_state=13).reset_index(drop=True)

    metrics = compute_dcr_overfitting_protection(
        real_train_df=real_train,
        real_holdout_df=holdout,
        synthetic_df=synthetic,
        max_rows=None,
        random_state=13,
    )

    assert metrics["closer_to_training_pct"] > metrics["closer_to_holdout_pct"]
    assert metrics["score"] < 0.5


def test_domias_like_attack_detects_memorization_signal():
    """DOMIAS-like attack should exceed random guess under memorization."""
    real_train = _make_real_train(n=180, seed=19)
    holdout = _make_holdout(n=180, seed=29)
    synthetic = real_train.sample(n=120, random_state=19).reset_index(drop=True)

    metrics = compute_domias_like_membership_inference(
        real_train_df=real_train,
        real_holdout_df=holdout,
        synthetic_df=synthetic,
        max_rows=None,
        random_state=19,
    )

    assert metrics["roc_auc"] > 0.6
    assert metrics["attacker_advantage"] > 0.05


def test_compute_privacy_metrics_has_expected_sections():
    """Integrated report should expose all privacy metric blocks."""
    real_train = _make_real_train(n=100, seed=31)
    holdout = _make_holdout(n=100, seed=37)
    synthetic = real_train.sample(n=90, random_state=31).reset_index(drop=True)

    report = compute_privacy_metrics(
        real_df=real_train,
        holdout_df=holdout,
        synthetic_df=synthetic,
        max_rows=None,
        random_state=31,
    )

    assert "dcr_stats" in report
    assert "dcr_baseline_protection" in report
    assert "dcr_overfitting_protection" in report
    assert "membership_inference_domias_like" in report
