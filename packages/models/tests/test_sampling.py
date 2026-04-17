"""Unit tests for trajectory sampling."""

import numpy as np
import pandas as pd
import pytest

from packages.models.forest_flow import ForestFlow
from packages.models.sampling import (
    sample_trajectory,
    prepare_training_data_from_trajectories,
)
from packages.data.transformers import TabularPreprocessor


class TestTrajectoryGeneration:
    """Test trajectory generation functionality."""

    def test_basic_trajectory_generation(self):
        """Test basic trajectory generation with simple data."""
        # Create and train simple model
        np.random.seed(42)
        X_target = np.random.randn(100, 2).astype(np.float32)
        X_condition = np.random.randn(100, 3).astype(np.float32)

        model = ForestFlow(nt=3, n_noise=5, n_jobs=1, random_state=42)
        model.fit(X_target, X_condition=X_condition)

        # Create simple preprocessor
        df_dummy = pd.DataFrame(
            {
                "hr": [70, 80, 90],
                "bp": [110, 120, 130],
                "age": [50, 60, 70],
            }
        )

        preprocessor = TabularPreprocessor(
            numeric_cols=["hr", "bp", "age"],
            categorical_cols=[],
        )
        preprocessor.fit(df_dummy)

        # Generate trajectories
        static_features = np.random.randn(5, 1).astype(np.float32)

        trajectories = sample_trajectory(
            model=model,
            preprocessor=preprocessor,
            static_features=static_features,
            n_trajectories=5,
            max_hours=10,
            target_cols=["hr", "bp"],
            condition_cols=["age", "hr_lag1", "bp_lag1"],
            random_state=42,
            verbose=False,
        )

        # Check output
        assert len(trajectories) == 5
        assert all(isinstance(traj, pd.DataFrame) for traj in trajectories)

        # Check each trajectory has timestep column
        for traj in trajectories:
            assert "timestep" in traj.columns
            assert len(traj) <= 10  # max_hours
            assert len(traj) > 0

    def test_trajectory_length_variability(self):
        """Test that trajectories can have variable lengths with discharge."""
        # This is a simplified test - full test would need discharge_col
        np.random.seed(42)
        X_target = np.random.randn(50, 1).astype(np.float32)  # 1D target
        X_condition = np.random.randn(50, 2).astype(np.float32)  # 2D condition

        model = ForestFlow(nt=2, n_noise=3, n_jobs=1, random_state=42)
        model.fit(X_target, X_condition=X_condition)

        df_dummy = pd.DataFrame({"hr": [70, 80, 90], "age": [50, 60, 70]})
        preprocessor = TabularPreprocessor(
            numeric_cols=["hr", "age"],
            categorical_cols=[],
        )
        preprocessor.fit(df_dummy)

        static_features = np.random.randn(3, 1).astype(np.float32)

        trajectories = sample_trajectory(
            model=model,
            preprocessor=preprocessor,
            static_features=static_features,
            n_trajectories=3,
            max_hours=5,
            target_cols=["hr"],
            condition_cols=["age", "hr_lag1"],
            random_state=42,
            verbose=False,
        )

        # Each trajectory should reach max_hours (no discharge)
        assert all(len(traj) == 5 for traj in trajectories)

    def test_static_features_constant(self):
        """Test that static features remain constant across timesteps."""
        np.random.seed(42)
        X_target = np.random.randn(50, 1).astype(np.float32)  # 1D target
        X_condition = np.random.randn(50, 2).astype(
            np.float32
        )  # 2D condition (static + lag)

        model = ForestFlow(nt=2, n_noise=3, n_jobs=1, random_state=42)
        model.fit(X_target, X_condition=X_condition)

        df_dummy = pd.DataFrame({"hr": [70, 80, 90], "age": [50, 60, 70]})
        preprocessor = TabularPreprocessor(
            numeric_cols=["hr", "age"],
            categorical_cols=[],
        )
        preprocessor.fit(df_dummy)

        # Static features: 1D (age)
        static_features = np.array([[0.5]]).astype(np.float32)

        trajectories = sample_trajectory(
            model=model,
            preprocessor=preprocessor,
            static_features=static_features,
            n_trajectories=1,
            max_hours=5,
            target_cols=["hr"],
            condition_cols=["age", "hr_lag1"],
            static_cols=["age"],
            random_state=42,
            verbose=False,
        )

        traj = trajectories[0]

        # Check static feature is present
        if "age" in traj.columns:
            # Static should be constant
            age_values = traj["age"].values
            assert np.allclose(
                age_values, age_values[0]
            ), f"Static feature 'age' should be constant, got {age_values}"


class TestPrepareTrainingData:
    """Test trajectory to DataFrame conversion."""

    def test_flatten_trajectories(self):
        """Test converting trajectory list to flat DataFrame."""
        # Create mock trajectories
        traj1 = pd.DataFrame(
            {
                "timestep": [0, 1, 2],
                "hr": [80, 82, 84],
                "bp": [120, 118, 116],
            }
        )

        traj2 = pd.DataFrame(
            {
                "timestep": [0, 1, 2, 3],
                "hr": [90, 92, 94, 96],
                "bp": [130, 128, 126, 124],
            }
        )

        df = prepare_training_data_from_trajectories([traj1, traj2])

        # Check structure
        assert "trajectory_id" in df.columns
        assert len(df) == 7  # 3 + 4 rows

        # Check IDs
        assert df[df["trajectory_id"] == 0].shape[0] == 3
        assert df[df["trajectory_id"] == 1].shape[0] == 4

        # Check all columns preserved
        assert "timestep" in df.columns
        assert "hr" in df.columns
        assert "bp" in df.columns

    def test_custom_id_column(self):
        """Test with custom ID column name."""
        traj1 = pd.DataFrame(
            {
                "timestep": [0, 1],
                "hr": [80, 82],
            }
        )

        df = prepare_training_data_from_trajectories(
            [traj1],
            id_col="patient_id",
        )

        assert "patient_id" in df.columns


class TestTrajectoryEdgeCases:
    """Test edge cases in trajectory generation."""

    def test_single_timestep_trajectory(self):
        """Test trajectory with single timestep (max_hours=1)."""
        np.random.seed(42)
        X_target = np.random.randn(30, 2).astype(np.float32)
        X_condition = np.random.randn(30, 2).astype(np.float32)

        model = ForestFlow(nt=2, n_noise=3, n_jobs=1, random_state=42)
        model.fit(X_target, X_condition=X_condition)

        df_dummy = pd.DataFrame({"hr": [70, 80], "age": [50, 60]})
        preprocessor = TabularPreprocessor(
            numeric_cols=["hr", "age"],
            categorical_cols=[],
        )
        preprocessor.fit(df_dummy)

        static_features = np.random.randn(1, 1).astype(np.float32)

        trajectories = sample_trajectory(
            model=model,
            preprocessor=preprocessor,
            static_features=static_features,
            n_trajectories=1,
            max_hours=1,  # Single timestep
            target_cols=["hr"],
            condition_cols=["age", "hr_lag1"],
            random_state=42,
            verbose=False,
        )

        # Should have 1 trajectory with 1 row
        assert len(trajectories) == 1
        assert len(trajectories[0]) == 1

    def test_zero_initial_strategy(self):
        """Test 'zero' initial strategy."""
        np.random.seed(42)
        X_target = np.random.randn(30, 1).astype(np.float32)  # 1D target
        X_condition = np.random.randn(30, 2).astype(np.float32)  # 2D condition

        model = ForestFlow(nt=2, n_noise=3, n_jobs=1, random_state=42)
        model.fit(X_target, X_condition=X_condition)

        df_dummy = pd.DataFrame({"hr": [70, 80, 90], "age": [50, 60, 70]})
        preprocessor = TabularPreprocessor(
            numeric_cols=["hr", "age"],
            categorical_cols=[],
        )
        preprocessor.fit(df_dummy)

        trajectories = sample_trajectory(
            model=model,
            preprocessor=preprocessor,
            n_trajectories=1,
            max_hours=3,
            target_cols=["hr"],
            condition_cols=["age", "hr_lag1"],
            initial_strategy="zero",  # Use zeros
            random_state=42,
            verbose=False,
        )

        assert len(trajectories) == 1
        assert len(trajectories[0]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
