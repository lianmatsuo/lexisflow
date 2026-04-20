"""Unit tests for HS3F model."""

import numpy as np
import pytest

from synth_gen.models.hs3f import HS3F


def _small_xgb_params() -> dict:
    return {
        "n_estimators": 10,
        "max_depth": 3,
        "learning_rate": 0.2,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }


class TestHS3FBasic:
    """Test core HS3F behavior."""

    def test_iid_mixed_targets(self):
        rng = np.random.default_rng(42)
        X = np.column_stack(
            [
                rng.normal(0.0, 1.0, 80),
                rng.integers(0, 4, 80),  # categorical
                rng.normal(5.0, 2.0, 80),
            ]
        ).astype(np.float32)

        model = HS3F(
            nt=2,
            n_noise=3,
            n_jobs=1,
            random_state=42,
            xgb_params=_small_xgb_params(),
        )
        model.fit(X, feature_types=["q", "c", "q"])

        X_synth = model.sample(n_samples=16, random_state=123)
        assert X_synth.shape == (16, 3)
        assert X_synth.dtype == np.float32

        # Categorical feature should be integer-coded class IDs
        cat_values = X_synth[:, 1]
        assert np.allclose(cat_values, np.round(cat_values))
        assert np.min(cat_values) >= 0
        assert np.max(cat_values) <= 3

    def test_conditional_mixed_targets(self):
        rng = np.random.default_rng(7)
        X_target = np.column_stack(
            [
                rng.normal(0.0, 1.0, 100),
                rng.integers(0, 3, 100),  # categorical
            ]
        ).astype(np.float32)
        X_condition = rng.normal(0.0, 1.0, (100, 2)).astype(np.float32)

        feature_types = ["q", "c", "q", "q"]
        model = HS3F(
            nt=2,
            n_noise=3,
            n_jobs=1,
            random_state=11,
            xgb_params=_small_xgb_params(),
        )
        model.fit(X_target, X_condition=X_condition, feature_types=feature_types)

        X_cond_new = rng.normal(0.0, 1.0, (12, 2)).astype(np.float32)
        X_synth = model.sample(n_samples=12, X_condition=X_cond_new, random_state=99)

        assert X_synth.shape == (12, 2)
        assert np.allclose(X_synth[:, 1], np.round(X_synth[:, 1]))

    def test_error_paths(self):
        model = HS3F(nt=2, n_noise=2, n_jobs=1, xgb_params=_small_xgb_params())

        with pytest.raises(RuntimeError):
            model.sample(5)

        rng = np.random.default_rng(1)
        X_target = rng.normal(size=(20, 2)).astype(np.float32)
        X_condition = rng.normal(size=(20, 1)).astype(np.float32)
        model.fit(X_target, X_condition=X_condition, feature_types=["q", "q", "q"])

        with pytest.raises(ValueError, match="must provide X_condition"):
            model.sample(5)

        with pytest.raises(ValueError, match="shape mismatch"):
            model.sample(5, X_condition=rng.normal(size=(5, 2)).astype(np.float32))

    def test_sparse_categorical_codes_are_supported(self):
        """Subsampled sparse label codes should train and sample correctly."""
        rng = np.random.default_rng(123)

        # Categorical target has sparse codes (non-contiguous), e.g. from subsampling.
        sparse_codes = np.array([0, 2, 3, 5, 8, 11], dtype=np.float32)
        X_target = np.column_stack(
            [
                rng.normal(0.0, 1.0, 120),
                rng.choice(sparse_codes, size=120),
            ]
        ).astype(np.float32)
        X_condition = rng.normal(0.0, 1.0, (120, 2)).astype(np.float32)

        model = HS3F(
            nt=2,
            n_noise=2,
            n_jobs=1,
            random_state=123,
            xgb_params=_small_xgb_params(),
        )
        model.fit(X_target, X_condition=X_condition, feature_types=["q", "c", "q", "q"])

        X_cond_new = rng.normal(0.0, 1.0, (20, 2)).astype(np.float32)
        X_synth = model.sample(n_samples=20, X_condition=X_cond_new, random_state=456)

        sampled_codes = np.unique(np.round(X_synth[:, 1]).astype(int))
        assert set(sampled_codes).issubset(set(sparse_codes.astype(int)))

    def test_level_parallel_training_runs(self):
        rng = np.random.default_rng(55)
        X = np.column_stack(
            [
                rng.normal(0.0, 1.0, 60),
                rng.integers(0, 3, 60),
                rng.normal(5.0, 1.0, 60),
            ]
        ).astype(np.float32)

        model = HS3F(
            nt=3,
            n_noise=2,
            n_jobs=2,
            random_state=55,
            xgb_params=_small_xgb_params(),
        )
        model.fit(X, feature_types=["q", "c", "q"])

        X_synth = model.sample(n_samples=8, random_state=77)
        assert X_synth.shape == (8, 3)
