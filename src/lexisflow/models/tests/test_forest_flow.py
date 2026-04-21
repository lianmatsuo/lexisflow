"""Unit tests for ForestFlow model."""

import numpy as np
import pandas as pd
import pytest

from lexisflow.models.forest_flow import ForestFlow


class TestForestFlowBasic:
    """Test basic ForestFlow functionality."""

    def test_initialization(self):
        """Test model initialization with various parameters."""
        model = ForestFlow(nt=5, n_noise=10, random_state=42)

        assert model.nt == 5
        assert model.n_noise == 10
        assert model.random_state == 42
        assert model.models_ == {}
        assert model.t_levels_ is None

    def test_iid_mode_training(self):
        """Test training in IID mode (no conditioning)."""
        # Create simple 2D synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 2).astype(np.float32)

        model = ForestFlow(nt=3, n_noise=5, n_jobs=1, random_state=42)
        model.fit(X)

        # Check model is fitted
        assert len(model.models_) == 3  # nt=3
        assert model.d_target_ == 2
        assert model.d_condition_ == 0
        assert not model.is_conditional_

        # Check t_levels
        expected_levels = [1 / 3, 2 / 3, 3 / 3]
        np.testing.assert_array_almost_equal(model.t_levels_, expected_levels)

    def test_conditional_mode_training(self):
        """Test training in conditional mode."""
        np.random.seed(42)
        X_target = np.random.randn(100, 2).astype(np.float32)
        X_condition = np.random.randn(100, 3).astype(np.float32)

        model = ForestFlow(nt=3, n_noise=5, n_jobs=1, random_state=42)
        model.fit(X_target, X_condition=X_condition)

        # Check model is fitted
        assert len(model.models_) == 3
        assert model.d_target_ == 2
        assert model.d_condition_ == 3
        assert model.is_conditional_

    def test_iid_sampling(self):
        """Test sampling in IID mode."""
        np.random.seed(42)
        X = np.random.randn(50, 2).astype(np.float32)

        model = ForestFlow(nt=3, n_noise=5, n_jobs=1, random_state=42)
        model.fit(X)

        # Sample
        X_synth = model.sample(n_samples=10, random_state=42)

        # Check shape
        assert X_synth.shape == (10, 2)

        # Check dtype
        assert X_synth.dtype == np.float32

        # Check samples are not all identical
        assert not np.allclose(X_synth[0], X_synth[1])

    def test_conditional_sampling(self):
        """Test sampling in conditional mode."""
        np.random.seed(42)
        X_target = np.random.randn(50, 2).astype(np.float32)
        X_condition = np.random.randn(50, 3).astype(np.float32)

        model = ForestFlow(nt=3, n_noise=5, n_jobs=1, random_state=42)
        model.fit(X_target, X_condition=X_condition)

        # Sample with new conditions
        X_cond_new = np.random.randn(10, 3).astype(np.float32)
        X_synth = model.sample(n_samples=10, X_condition=X_cond_new, random_state=42)

        # Check shape
        assert X_synth.shape == (10, 2)

    def test_sample_without_fit_error(self):
        """Test that sampling before fit raises error."""
        model = ForestFlow(nt=3, n_noise=5)

        with pytest.raises(RuntimeError, match="must be fit"):
            model.sample(10)

    def test_conditional_mismatch_errors(self):
        """Test errors when conditioning is mismatched."""
        np.random.seed(42)
        X_target = np.random.randn(50, 2).astype(np.float32)
        X_condition = np.random.randn(50, 3).astype(np.float32)

        # Train conditionally
        model = ForestFlow(nt=2, n_noise=3, n_jobs=1, random_state=42)
        model.fit(X_target, X_condition=X_condition)

        # Error 1: Sample without condition
        with pytest.raises(ValueError, match="must provide X_condition"):
            model.sample(10)

        # Error 2: Wrong condition shape
        X_cond_wrong = np.random.randn(10, 5).astype(np.float32)  # Wrong dims
        with pytest.raises(ValueError, match="shape mismatch"):
            model.sample(10, X_condition=X_cond_wrong)

        # Error 3: Wrong number of samples
        X_cond_wrong_n = np.random.randn(5, 3).astype(np.float32)  # Wrong n
        with pytest.raises(ValueError, match="shape mismatch"):
            model.sample(10, X_condition=X_cond_wrong_n)


class TestForestFlowIntegration:
    """Integration tests with realistic scenarios."""

    def test_small_dataset_training(self):
        """Test training with small dataset (quick test mode)."""
        np.random.seed(42)

        # Simulated ICU data: 20 patients × 10 hours = 200 rows
        n_patients = 20
        n_hours = 10

        data = []
        for pid in range(n_patients):
            age = np.random.randint(30, 80)
            for hour in range(n_hours):
                data.append(
                    {
                        "hr": np.random.normal(80, 10),
                        "bp": np.random.normal(120, 15),
                        "age": age,
                        "hr_lag1": np.random.normal(80, 10) if hour > 0 else -1.0,
                        "bp_lag1": np.random.normal(120, 15) if hour > 0 else -1.0,
                    }
                )

        df = pd.DataFrame(data)

        # Target and condition
        X_target = df[["hr", "bp"]].values.astype(np.float32)
        X_condition = df[["age", "hr_lag1", "bp_lag1"]].values.astype(np.float32)

        # Train
        model = ForestFlow(nt=5, n_noise=5, n_jobs=1, random_state=42)
        model.fit(X_target, X_condition=X_condition)

        # Sample
        X_cond_new = np.random.randn(10, 3).astype(np.float32)
        X_synth = model.sample(10, X_condition=X_cond_new, random_state=42)

        # Check outputs are reasonable
        assert X_synth.shape == (10, 2)
        assert not np.isnan(X_synth).any()

    def test_reproducibility_with_seed(self):
        """Test that same seed gives same results."""
        np.random.seed(42)
        X = np.random.randn(50, 2).astype(np.float32)

        # Train two models with same seed
        model1 = ForestFlow(nt=3, n_noise=5, n_jobs=1, random_state=42)
        model1.fit(X)

        model2 = ForestFlow(nt=3, n_noise=5, n_jobs=1, random_state=42)
        model2.fit(X)

        # Sample with same seed
        X_synth1 = model1.sample(10, random_state=123)
        X_synth2 = model2.sample(10, random_state=123)

        # Should be identical (or very close due to XGBoost randomness)
        np.testing.assert_array_almost_equal(X_synth1, X_synth2, decimal=2)


class TestDataIteratorMode:
    """Test data iterator mode for memory efficiency."""

    def test_iterator_vs_legacy_equivalence(self):
        """Test that iterator mode gives similar results to legacy mode."""
        np.random.seed(42)
        X = np.random.randn(100, 2).astype(np.float32)

        # Train with legacy mode
        model_legacy = ForestFlow(
            nt=3, n_noise=5, n_jobs=1, use_data_iterator=False, random_state=42
        )
        model_legacy.fit(X)
        X_synth_legacy = model_legacy.sample(20, random_state=123)

        # Train with iterator mode
        model_iter = ForestFlow(
            nt=3,
            n_noise=5,
            n_jobs=1,
            use_data_iterator=True,
            batch_size=50,
            random_state=42,
        )
        model_iter.fit(X)
        X_synth_iter = model_iter.sample(20, random_state=123)

        # Results should be similar (not exact due to batching)
        # Check distributions are similar
        assert X_synth_legacy.mean() == pytest.approx(X_synth_iter.mean(), abs=0.5)
        assert X_synth_legacy.std() == pytest.approx(X_synth_iter.std(), abs=0.5)


class TestCategoricalSupport:
    """Test XGBoost categorical feature support."""

    def test_categorical_in_condition(self):
        """Test categorical features in condition (typical use case)."""
        np.random.seed(42)

        # Simulate: 2 dynamic features (target) + 1 numeric + 1 categorical (condition)
        # Target: continuous values (can be noised)
        X_target = np.random.randn(100, 2).astype(np.float32)

        # Condition: 1 numeric + 1 categorical (encoded as integers)
        X_condition = np.zeros((100, 2), dtype=np.float32)
        X_condition[:, 0] = np.random.randn(100)  # Numeric feature
        X_condition[:, 1] = np.random.randint(
            0, 3, size=100
        )  # Categorical (3 categories)

        # Feature types: [target_q, target_q, condition_q, condition_c]
        feature_types = ["q", "q", "q", "c"]

        # Train with categorical support
        model = ForestFlow(
            nt=3,
            n_noise=5,
            n_jobs=1,
            random_state=42,
            use_data_iterator=True,
            batch_size=50,
        )
        model.fit(X_target, X_condition=X_condition, feature_types=feature_types)

        # Check model fitted
        assert model.is_conditional_
        assert model.feature_types_ == feature_types

        # Sample
        X_cond_new = np.zeros((10, 2), dtype=np.float32)
        X_cond_new[:, 0] = np.random.randn(10)
        X_cond_new[:, 1] = np.random.randint(0, 3, size=10)

        X_synth = model.sample(n_samples=10, X_condition=X_cond_new, random_state=42)

        # Check shape
        assert X_synth.shape == (10, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
