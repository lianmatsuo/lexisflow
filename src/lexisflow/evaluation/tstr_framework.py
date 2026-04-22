"""Unified TSTR (Train on Synthetic, Test on Real) Framework.

This module provides an extensible framework for evaluating synthetic data
quality across multiple clinical prediction tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class _PaddedSequenceClassifier:
    """Patient-level classifier on padded time-ordered trajectories."""

    def __init__(
        self,
        random_state: int = 42,
        max_len: int | None = None,
        task_type: str = "binary",
    ):
        self.random_state = random_state
        self.max_len_: int | None = max_len
        self.feature_dim_: int | None = None
        self.scaler_: StandardScaler | None = None
        self.model_: LogisticRegression | None = None
        self.task_type = task_type

    def _vectorize(self, sequences: list[np.ndarray]) -> np.ndarray:
        if self.max_len_ is None or self.feature_dim_ is None:
            raise RuntimeError("Sequence model is not fitted.")

        n_samples = len(sequences)
        padded = np.zeros(
            (n_samples, self.max_len_, self.feature_dim_),
            dtype=np.float32,
        )
        mask = np.zeros((n_samples, self.max_len_, 1), dtype=np.float32)
        lengths = np.zeros((n_samples, 1), dtype=np.float32)

        for i, seq in enumerate(sequences):
            seq_arr = np.asarray(seq, dtype=np.float32)
            seq_len = min(seq_arr.shape[0], self.max_len_)
            padded[i, :seq_len, :] = seq_arr[:seq_len]
            mask[i, :seq_len, 0] = 1.0
            lengths[i, 0] = float(seq_len) / float(max(self.max_len_, 1))

        features = np.concatenate([padded, mask], axis=2).reshape(n_samples, -1)
        return np.hstack([features, lengths])

    def fit(
        self,
        sequences: list[np.ndarray],
        labels: np.ndarray,
    ) -> "_PaddedSequenceClassifier":
        if self.max_len_ is None:
            self.max_len_ = max(seq.shape[0] for seq in sequences)
        self.feature_dim_ = sequences[0].shape[1]
        X = self._vectorize(sequences)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        model_kwargs: dict[str, Any] = {
            "max_iter": 500,
            "random_state": self.random_state,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "C": 1.0,
        }
        self.model_ = LogisticRegression(**model_kwargs)
        self.model_.fit(X_scaled, labels)
        return self

    def predict(self, sequences: list[np.ndarray]) -> np.ndarray:
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("Sequence model is not fitted.")
        X = self._vectorize(sequences)
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def predict_proba(self, sequences: list[np.ndarray]) -> np.ndarray:
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("Sequence model is not fitted.")
        X = self._vectorize(sequences)
        X_scaled = self.scaler_.transform(X)
        proba = self.model_.predict_proba(X_scaled)
        if self.task_type == "binary":
            return proba[:, 1]
        return proba


def _extract_patient_sequences(
    features: np.ndarray,
    df: pd.DataFrame,
    label_fn: Callable[[pd.DataFrame], int | None],
    random_state: int,
    sequence_max_patients: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Group row-level features into patient trajectories with per-patient labels."""
    meta = pd.DataFrame(
        {
            "subject_id": df["subject_id"].values,
            "hours_in": pd.to_numeric(df["hours_in"], errors="coerce").fillna(0.0),
            "row_idx": np.arange(len(df)),
        }
    )
    meta = meta.dropna(subset=["subject_id"])

    if sequence_max_patients is not None:
        unique_subjects = meta["subject_id"].unique()
        if len(unique_subjects) > sequence_max_patients:
            rng = np.random.default_rng(random_state)
            sampled_subjects = rng.choice(
                unique_subjects,
                size=sequence_max_patients,
                replace=False,
            )
            meta = meta[meta["subject_id"].isin(sampled_subjects)]

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    grouped = meta.sort_values(["subject_id", "hours_in"]).groupby("subject_id")
    for _, group in grouped:
        row_idx = group["row_idx"].to_numpy(dtype=np.int64, copy=False)
        seq = features[row_idx]
        if seq.shape[0] == 0:
            continue
        patient_rows = df.iloc[row_idx]
        label = label_fn(patient_rows)
        if label is None:
            continue
        sequences.append(seq.astype(np.float32, copy=False))
        labels.append(int(label))

    return sequences, np.asarray(labels, dtype=np.int64)


def _evaluate_sequence_task(
    task: "TSTRTask",
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    label_fn: Callable[[pd.DataFrame], int | None],
    test_size: float,
    verbose: bool,
    sequence_max_patients: int | None,
    task_type: str = "binary",
) -> Dict[str, float]:
    """Shared sequence-level TSTR loop across patient-level tasks."""
    required_cols = {"subject_id", "hours_in"}
    if not required_cols.issubset(set(synth_df.columns)) or not required_cols.issubset(
        set(real_df.columns)
    ):
        if verbose:
            print("    Warning: Missing required trajectory columns for sequence TSTR")
        return task._empty_metrics()

    try:
        task_exclude_cols = task.get_exclude_cols()
        synth_rows, synth_feature_names = task.prepare_features(
            synth_df,
            task_exclude_cols,
        )
        real_rows, real_feature_names = task.prepare_features(
            real_df,
            task_exclude_cols,
        )

        synth_feature_df = pd.DataFrame(synth_rows, columns=synth_feature_names)
        real_feature_df = pd.DataFrame(real_rows, columns=real_feature_names)
        all_feature_names = sorted(
            set(synth_feature_df.columns).union(real_feature_df.columns)
        )

        synth_aligned = synth_feature_df.reindex(
            columns=all_feature_names, fill_value=0.0
        )
        real_aligned = real_feature_df.reindex(
            columns=all_feature_names, fill_value=0.0
        )

        scaler = StandardScaler()
        synth_scaled = scaler.fit_transform(synth_aligned.values)
        real_scaled = scaler.transform(real_aligned.values)

        synth_sequences, y_synth = _extract_patient_sequences(
            synth_scaled,
            synth_df,
            label_fn=label_fn,
            random_state=task.random_state,
            sequence_max_patients=sequence_max_patients,
        )
        real_sequences, y_real = _extract_patient_sequences(
            real_scaled,
            real_df,
            label_fn=label_fn,
            random_state=task.random_state,
            sequence_max_patients=sequence_max_patients,
        )

        if len(synth_sequences) == 0 or len(real_sequences) == 0:
            if verbose:
                print("    Warning: No valid patient trajectories for sequence TSTR")
            return task._empty_metrics()
        if len(np.unique(y_synth)) < 2:
            if verbose:
                print("    Warning: Synthetic trajectories contain one class only")
            return task._empty_metrics()

        real_idx = np.arange(len(real_sequences))
        train_idx, test_idx = train_test_split(
            real_idx,
            test_size=test_size,
            random_state=task.random_state,
            stratify=y_real if len(np.unique(y_real)) > 1 else None,
        )
        y_real_train = y_real[train_idx]
        y_real_test = y_real[test_idx]
        if len(np.unique(y_real_train)) < 2 or len(np.unique(y_real_test)) < 2:
            if verbose:
                print("    Warning: Real split has one class only for sequence TSTR")
            return task._empty_metrics()

        real_train_sequences = [real_sequences[i] for i in train_idx]
        real_test_sequences = [real_sequences[i] for i in test_idx]
        global_max_len = max(
            max(seq.shape[0] for seq in synth_sequences),
            max(seq.shape[0] for seq in real_sequences),
        )

        if verbose:
            print(
                "    Training sequence classifier on synthetic trajectories "
                f"({len(synth_sequences)} patients)..."
            )
        synth_model = _PaddedSequenceClassifier(
            random_state=task.random_state,
            max_len=global_max_len,
            task_type=task_type,
        ).fit(synth_sequences, y_synth)
        y_pred_synth = synth_model.predict(real_test_sequences)
        y_proba_synth = synth_model.predict_proba(real_test_sequences)
        synth_metrics = task.compute_metrics(y_real_test, y_pred_synth, y_proba_synth)

        if verbose:
            print(
                "    Training sequence baseline on real trajectories "
                f"({len(real_train_sequences)} patients)..."
            )
        real_model = _PaddedSequenceClassifier(
            random_state=task.random_state,
            max_len=global_max_len,
            task_type=task_type,
        ).fit(real_train_sequences, y_real_train)
        y_pred_real = real_model.predict(real_test_sequences)
        y_proba_real = real_model.predict_proba(real_test_sequences)
        real_metrics = task.compute_metrics(y_real_test, y_pred_real, y_proba_real)

        results: dict[str, float] = {}
        for key, val in synth_metrics.items():
            results[f"synth_{key}"] = val
        for key, val in real_metrics.items():
            results[f"real_{key}"] = val
        return results
    except Exception as e:
        if verbose:
            print(f"    ERROR during {task.name} sequence TSTR: {e}")
        return task._empty_metrics()


class TSTRTask(ABC):
    """Abstract base class for TSTR evaluation tasks.

    Each task defines:
    - How to prepare features and labels from raw data
    - Which model to use for the task
    - How to compute task-specific evaluation metrics
    """

    def __init__(self, name: str, random_state: int = 42):
        """Initialize TSTR task.

        Parameters
        ----------
        name : str
            Task name (e.g., 'mortality', 'vasopressor', 'los').
        random_state : int
            Random seed for reproducibility.
        """
        self.name = name
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    @abstractmethod
    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and prepare target variable from dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe with all columns.

        Returns
        -------
        y : np.ndarray
            Target labels.
        """
        pass

    def get_exclude_cols(self) -> set:
        """Get task-specific columns to exclude from features.

        Override this method to exclude task-specific target columns.

        Returns
        -------
        exclude_cols : set
            Set of column names to exclude.
        """
        return set()

    @abstractmethod
    def get_model(self) -> Any:
        """Create and return the model for this task.

        Returns
        -------
        model : sklearn estimator
            Initialized model (not fitted).
        """
        pass

    @abstractmethod
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute task-specific evaluation metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.
        y_proba : np.ndarray, optional
            Predicted probabilities (for ROC-AUC).

        Returns
        -------
        metrics : dict
            Dictionary of metric names and values.
        """
        pass

    def prepare_features(
        self, df: pd.DataFrame, exclude_cols: set = None
    ) -> np.ndarray:
        """Prepare feature matrix from dataframe.

        This is shared across all tasks - extracts features and handles
        categorical encoding.

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe.
        exclude_cols : set, optional
            Additional columns to exclude (task-specific).

        Returns
        -------
        X : np.ndarray
            Feature matrix.
        """
        # Base exclusions (IDs, timestamps, outcomes)
        base_exclude = {
            # IDs
            "subject_id",
            "hadm_id",
            "icustay_id",
            "trajectory_id",
            # Timestamps
            "timestep",
            "hours_in",
            "charttime",
            "admittime",
            "intime",
            "dischtime",
            "outtime",
            "deathtime",
            # Outcomes (leaky)
            "hospital_expire_flag",
            "mort_icu",
            "mort_hosp",
            "dnr_first_charttime",
            "timecmo_chart",
            "discharge_location",
            "hospstay_seq",
            "readmission_30",
            "los_icu",
            "los_hospital",
        }

        # Merge with task-specific exclusions
        if exclude_cols:
            base_exclude = base_exclude.union(exclude_cols)

        # Get available columns
        available_cols = [col for col in df.columns if col not in base_exclude]

        # Exclude aggregated statistics (these leak outcome information)
        feature_cols = []
        for col in available_cols:
            col_lower = str(col).lower()
            if any(
                agg in col_lower
                for agg in [
                    "count)",
                    "mean)",
                    "std)",
                    "min)",
                    "max)",
                    "median)",
                    "sum)",
                ]
            ):
                continue
            feature_cols.append(col)

        # Extract features
        X = df[feature_cols].copy()

        # Handle numeric vs categorical
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in feature_cols if col not in numeric_cols]

        # Fill missing values
        for col in numeric_cols:
            median_val = X[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            X[col] = X[col].fillna(median_val)

        for col in categorical_cols:
            mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else "UNKNOWN"
            X[col] = X[col].fillna(mode_val)

        # One-hot encode categorical
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        X = X.fillna(0)

        return X.values, X.columns.tolist()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TSTRTask":
        """Fit the task model on training data.

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training labels.

        Returns
        -------
        self : TSTRTask
            Fitted task.
        """
        # Check if we have valid labels
        unique_classes = np.unique(y[~np.isnan(y)])
        if len(unique_classes) < 2:
            raise ValueError(
                f"Task '{self.name}': Training data contains only one class: {unique_classes}. "
                "Cannot train classifier on single-class data."
            )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Create and fit model
        self.model = self.get_model()
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels.

        Parameters
        ----------
        X : np.ndarray
            Test features.

        Returns
        -------
        y_pred : np.ndarray
            Predicted labels.
        """
        if not self.is_fitted:
            raise RuntimeError(f"Task '{self.name}' not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (if applicable).

        Parameters
        ----------
        X : np.ndarray
            Test features.

        Returns
        -------
        y_proba : np.ndarray
            Predicted probabilities.
        """
        if not self.is_fitted:
            raise RuntimeError(f"Task '{self.name}' not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        # Check if model has predict_proba
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        else:
            raise NotImplementedError(
                f"Model for task '{self.name}' does not support predict_proba"
            )

    def evaluate(
        self,
        synth_df: pd.DataFrame,
        real_df: pd.DataFrame,
        test_size: float = 0.3,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Full TSTR evaluation for this task.

        Parameters
        ----------
        synth_df : pd.DataFrame
            Synthetic data (for training).
        real_df : pd.DataFrame
            Real data (for testing).
        test_size : float
            Proportion of real data to use for testing.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        metrics : dict
            Dictionary with synth_* and real_* metrics.
        """
        if verbose:
            print(f"\n  [{self.name.upper()}] Evaluating TSTR...")

        try:
            # Get task-specific exclusions
            task_exclude_cols = self.get_exclude_cols()

            # Prepare synthetic data
            if verbose:
                print("    Preparing synthetic features...")
            X_synth, synth_feature_names = self.prepare_features(
                synth_df, task_exclude_cols
            )
            y_synth = self.prepare_target(synth_df)

            # Prepare real data
            if verbose:
                print("    Preparing real features...")
            X_real, real_feature_names = self.prepare_features(
                real_df, task_exclude_cols
            )
            y_real = self.prepare_target(real_df)

            # Align feature spaces (synth to match real)
            if verbose:
                print("    Aligning feature spaces...")
            synth_df_features = pd.DataFrame(X_synth, columns=synth_feature_names)

            # Add missing columns
            missing_cols = [
                col
                for col in real_feature_names
                if col not in synth_df_features.columns
            ]
            if missing_cols:
                missing_data = pd.DataFrame(
                    0, index=synth_df_features.index, columns=missing_cols
                )
                synth_df_features = pd.concat([synth_df_features, missing_data], axis=1)

            # Keep only columns in real data (same order)
            synth_df_features = synth_df_features[real_feature_names]
            X_synth_aligned = synth_df_features.values

            # Remove NaN labels
            synth_valid_mask = ~np.isnan(y_synth)
            X_synth_aligned = X_synth_aligned[synth_valid_mask]
            y_synth = y_synth[synth_valid_mask]

            real_valid_mask = ~np.isnan(y_real)
            X_real = X_real[real_valid_mask]
            y_real = y_real[real_valid_mask]

            # Check if we have valid data
            if len(X_synth_aligned) == 0 or len(X_real) == 0:
                if verbose:
                    print(f"    Warning: No valid samples, skipping task '{self.name}'")
                return self._empty_metrics()

            if len(np.unique(y_synth)) < 2:
                if verbose:
                    print(
                        f"    Warning: Synthetic data has only one class, skipping task '{self.name}'"
                    )
                return self._empty_metrics()

            # Split real data
            X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
                X_real,
                y_real,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y_real if len(np.unique(y_real)) > 1 else None,
            )

            if len(np.unique(y_real_train)) < 2 or len(np.unique(y_real_test)) < 2:
                if verbose:
                    print(
                        f"    Warning: Real data split has only one class, skipping task '{self.name}'"
                    )
                return self._empty_metrics()

            # Train on synthetic data
            if verbose:
                print(
                    f"    Training on synthetic data ({len(X_synth_aligned)} samples)..."
                )
            synth_task = type(self)(random_state=self.random_state)
            synth_task.fit(X_synth_aligned, y_synth)

            # Evaluate on real test set
            if verbose:
                print(
                    f"    Evaluating on real test set ({len(X_real_test)} samples)..."
                )
            y_pred_synth = synth_task.predict(X_real_test)

            # Get probabilities if available
            try:
                y_proba_synth = synth_task.predict_proba(X_real_test)
            except Exception:
                y_proba_synth = None

            synth_metrics = self.compute_metrics(
                y_real_test, y_pred_synth, y_proba_synth
            )

            # Train on real data (baseline)
            if verbose:
                print(
                    f"    Training baseline on real data ({len(X_real_train)} samples)..."
                )
            real_task = type(self)(random_state=self.random_state)
            real_task.fit(X_real_train, y_real_train)

            y_pred_real = real_task.predict(X_real_test)
            try:
                y_proba_real = real_task.predict_proba(X_real_test)
            except Exception:
                y_proba_real = None

            real_metrics = self.compute_metrics(y_real_test, y_pred_real, y_proba_real)

            # Combine results with prefixes
            results = {}
            for key, val in synth_metrics.items():
                results[f"synth_{key}"] = val
            for key, val in real_metrics.items():
                results[f"real_{key}"] = val

            if verbose:
                print(f"    ✓ {self.name} TSTR complete!")

            return results

        except Exception as e:
            if verbose:
                print(f"    ERROR during {self.name} TSTR: {e}")
            return self._empty_metrics()

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dict (all NaN)."""
        base_metrics = ["accuracy", "f1", "roc_auc"]
        results = {}
        for prefix in ["synth", "real"]:
            for metric in base_metrics:
                results[f"{prefix}_{metric}"] = np.nan
        return results


class MortalityTask(TSTRTask):
    """Mortality prediction task (binary classification).

    Predicts in-hospital mortality from ICU features.
    """

    def __init__(
        self,
        random_state: int = 42,
        use_sequence_model: bool = False,
        sequence_max_patients: int | None = None,
    ):
        super().__init__(name="mortality", random_state=random_state)
        self.use_sequence_model = use_sequence_model
        self.sequence_max_patients = sequence_max_patients

    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Extract mortality labels."""
        if "hospital_expire_flag" not in df.columns:
            raise ValueError("Column 'hospital_expire_flag' not found")
        return df["hospital_expire_flag"].values

    def get_model(self) -> LogisticRegression:
        """Create logistic regression model."""
        return LogisticRegression(
            max_iter=500,
            random_state=self.random_state,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
        )

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute mortality metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        # Add ROC-AUC if probabilities available
        if y_proba is not None:
            # For binary classification, take probability of positive class
            if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except Exception:
                metrics["roc_auc"] = np.nan
        else:
            metrics["roc_auc"] = np.nan

        return metrics

    @staticmethod
    def _sequence_label_fn(patient_rows: pd.DataFrame) -> int | None:
        label_series = pd.to_numeric(
            patient_rows["hospital_expire_flag"],
            errors="coerce",
        ).dropna()
        if label_series.empty:
            return None
        return int(np.clip(np.round(label_series.iloc[0]), 0, 1))

    def evaluate(
        self,
        synth_df: pd.DataFrame,
        real_df: pd.DataFrame,
        test_size: float = 0.3,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Run row-level or patient-sequence TSTR evaluation for mortality."""
        if not self.use_sequence_model:
            return super().evaluate(
                synth_df=synth_df,
                real_df=real_df,
                test_size=test_size,
                verbose=verbose,
            )

        if verbose:
            print("\n  [MORTALITY] Evaluating sequence TSTR...")
        return _evaluate_sequence_task(
            task=self,
            synth_df=synth_df,
            real_df=real_df,
            label_fn=self._sequence_label_fn,
            test_size=test_size,
            verbose=verbose,
            sequence_max_patients=self.sequence_max_patients,
            task_type="binary",
        )


class VasopressorTask(TSTRTask):
    """Vasopressor requirement prediction (binary classification).

    Predicts if patient requires vasopressors during ICU stay.
    """

    def __init__(
        self,
        random_state: int = 42,
        use_sequence_model: bool = False,
        sequence_max_patients: int | None = None,
    ):
        super().__init__(name="vasopressor", random_state=random_state)
        self.use_sequence_model = use_sequence_model
        self.sequence_max_patients = sequence_max_patients

    def get_exclude_cols(self) -> set:
        """Exclude vaso column from features (it's the target)."""
        return {"vaso"}

    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Extract vasopressor usage labels."""
        if "vaso" not in df.columns:
            raise ValueError("Column 'vaso' not found")

        # vaso is binary (0 or 1) indicating if patient received vasopressors
        y = df["vaso"].values

        # Convert to integer (in case it's float)
        y = np.round(y).astype(int)

        return y

    def get_model(self) -> LogisticRegression:
        """Create logistic regression model."""
        return LogisticRegression(
            max_iter=500,
            random_state=self.random_state,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
        )

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute vasopressor metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None:
            if len(y_proba.shape) == 2 and y_proba.shape[1] == 2:
                y_proba = y_proba[:, 1]
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except Exception:
                metrics["roc_auc"] = np.nan
        else:
            metrics["roc_auc"] = np.nan

        return metrics

    @staticmethod
    def _sequence_label_fn(patient_rows: pd.DataFrame) -> int | None:
        vaso_series = pd.to_numeric(patient_rows["vaso"], errors="coerce").dropna()
        if vaso_series.empty:
            return None
        vaso_binary = np.clip(np.round(vaso_series.values), 0, 1).astype(int)
        return int(np.any(vaso_binary > 0))

    def evaluate(
        self,
        synth_df: pd.DataFrame,
        real_df: pd.DataFrame,
        test_size: float = 0.3,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Run row-level or patient-sequence TSTR evaluation for vasopressor."""
        if not self.use_sequence_model:
            return super().evaluate(
                synth_df=synth_df,
                real_df=real_df,
                test_size=test_size,
                verbose=verbose,
            )

        if verbose:
            print("\n  [VASOPRESSOR] Evaluating sequence TSTR...")
        return _evaluate_sequence_task(
            task=self,
            synth_df=synth_df,
            real_df=real_df,
            label_fn=self._sequence_label_fn,
            test_size=test_size,
            verbose=verbose,
            sequence_max_patients=self.sequence_max_patients,
            task_type="binary",
        )


class LOSTask(TSTRTask):
    """Length-of-Stay prediction (multi-class classification).

    Predicts ICU length-of-stay category:
    - 0: Short (0-2 days)
    - 1: Medium (3-7 days)
    - 2: Long (8+ days)
    """

    def __init__(
        self,
        random_state: int = 42,
        use_sequence_model: bool = False,
        sequence_max_patients: int | None = None,
    ):
        super().__init__(name="los", random_state=random_state)
        self.use_sequence_model = use_sequence_model
        self.sequence_max_patients = sequence_max_patients
        # LOS bin thresholds in days
        self.bin_edges = [0, 2, 7, np.inf]
        self.bin_labels = [0, 1, 2]  # short, medium, long

    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and bin LOS labels."""
        if "los_icu" not in df.columns:
            raise ValueError("Column 'los_icu' not found")

        # los_icu is in days
        los_days = df["los_icu"].values

        # Bin into categories
        y = np.digitize(los_days, bins=self.bin_edges[1:], right=False)

        return y

    def get_model(self) -> LogisticRegression:
        """Create multi-class logistic regression model."""
        return LogisticRegression(
            max_iter=500,
            random_state=self.random_state,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
        )

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute LOS metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        metrics["roc_auc"] = self._compute_los_roc_auc(y_true, y_proba)

        return metrics

    @staticmethod
    def _compute_los_roc_auc(
        y_true: np.ndarray, y_proba: Optional[np.ndarray]
    ) -> float:
        """Compute LOS ROC-AUC with binary fallback for missing classes."""
        if y_proba is None:
            return np.nan

        y_true_arr = np.asarray(y_true)
        classes = np.unique(y_true_arr)
        if len(classes) < 2:
            return np.nan

        y_score = np.asarray(y_proba)
        if y_score.ndim == 1:
            if len(classes) != 2:
                return np.nan
            pos_class = classes[-1]
            y_bin = (y_true_arr == pos_class).astype(int)
            return float(roc_auc_score(y_bin, y_score))

        n_cols = y_score.shape[1]
        try:
            # Usual multiclass path when labels and score columns align.
            if len(classes) > 2 and n_cols == len(classes):
                return float(
                    roc_auc_score(
                        y_true_arr, y_score, multi_class="ovr", average="macro"
                    )
                )

            # Binary fallback for sparse-label splits (e.g., LOS class 0 absent).
            if len(classes) == 2:
                pos_class = int(classes[-1])
                if n_cols == 2:
                    pos_score = y_score[:, 1]
                elif 0 <= pos_class < n_cols:
                    # LOS labels are canonical bins 0/1/2, so this aligns when
                    # model probabilities include all bins but y_true has only 2.
                    pos_score = y_score[:, pos_class]
                else:
                    pos_score = y_score[:, -1]
                y_bin = (y_true_arr == classes[-1]).astype(int)
                return float(roc_auc_score(y_bin, pos_score))

            # Fallback: try selecting columns by observed class IDs.
            class_cols = [int(c) for c in classes if 0 <= int(c) < n_cols]
            if len(classes) > 2 and len(class_cols) == len(classes):
                sub_scores = y_score[:, class_cols]
                remap = {int(c): i for i, c in enumerate(classes)}
                y_remapped = np.asarray([remap[int(c)] for c in y_true_arr], dtype=int)
                return float(
                    roc_auc_score(
                        y_remapped, sub_scores, multi_class="ovr", average="macro"
                    )
                )
        except Exception:
            return np.nan
        return np.nan

    def _sequence_label_fn(self, patient_rows: pd.DataFrame) -> int | None:
        if "los_icu" in patient_rows.columns:
            los_series = pd.to_numeric(
                patient_rows["los_icu"],
                errors="coerce",
            ).dropna()
        else:
            los_series = pd.Series(dtype=float)
        if not los_series.empty:
            los_days = float(los_series.iloc[0])
        else:
            if "hours_in" not in patient_rows.columns:
                return None
            hours_series = pd.to_numeric(
                patient_rows["hours_in"],
                errors="coerce",
            ).dropna()
            if hours_series.empty:
                return None
            los_days = float(max(hours_series.max(), 0.0) / 24.0)
        return int(np.digitize(los_days, bins=self.bin_edges[1:], right=False))

    def evaluate(
        self,
        synth_df: pd.DataFrame,
        real_df: pd.DataFrame,
        test_size: float = 0.3,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Run row-level or patient-sequence TSTR evaluation for LOS."""
        if not self.use_sequence_model:
            return super().evaluate(
                synth_df=synth_df,
                real_df=real_df,
                test_size=test_size,
                verbose=verbose,
            )

        if verbose:
            print("\n  [LOS] Evaluating sequence TSTR...")
        return _evaluate_sequence_task(
            task=self,
            synth_df=synth_df,
            real_df=real_df,
            label_fn=self._sequence_label_fn,
            test_size=test_size,
            verbose=verbose,
            sequence_max_patients=self.sequence_max_patients,
            task_type="multiclass",
        )


def evaluate_tstr_multi_task(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    tasks: list[TSTRTask],
    test_size: float = 0.3,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple TSTR tasks.

    Parameters
    ----------
    synth_df : pd.DataFrame
        Synthetic data (for training).
    real_df : pd.DataFrame
        Real data (for testing).
    tasks : list[TSTRTask]
        List of TSTR tasks to evaluate.
    test_size : float
        Proportion of real data for testing.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    results : dict
        Dictionary mapping task names to their metrics.
        Example: {'mortality': {'synth_accuracy': 0.85, ...}, ...}
    """
    results = {}

    if verbose:
        print(f"\n{'='*70}")
        print(f"TSTR Multi-Task Evaluation ({len(tasks)} tasks)")
        print(f"{'='*70}")

    for task in tasks:
        task_results = task.evaluate(
            synth_df, real_df, test_size=test_size, verbose=verbose
        )
        results[task.name] = task_results

        # Print summary for this task
        if verbose and not all(np.isnan(v) for v in task_results.values()):
            print(
                f"    Synth AUROC: {task_results.get('synth_roc_auc', np.nan):.4f}, "
                f"Real AUROC: {task_results.get('real_roc_auc', np.nan):.4f}"
            )

    if verbose:
        print(f"\n{'='*70}")
        print("Multi-Task TSTR Complete!")
        print(f"{'='*70}")

    return results
