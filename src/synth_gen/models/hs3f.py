"""HS3F model: sequential heterogeneous generation for tabular data.

This module implements a practical HS3F-style generator:
- Continuous target features are generated with 1D flow matching regressors
- Categorical target features are generated with XGBoost classifiers
- Features are generated sequentially to condition on previously generated values
"""

from __future__ import annotations

from contextlib import contextmanager
import numpy as np
from joblib import Parallel, delayed
import joblib
import xgboost as xgb
from tqdm import tqdm


class HS3F:
    """Heterogeneous Sequential Feature Forest-Flow model.

    The model keeps Forest-Flow style ODE-based generation for continuous target
    features while replacing categorical flow matching with direct categorical
    sampling from XGBoost class probabilities.
    """

    def __init__(
        self,
        nt: int = 50,
        n_noise: int = 100,
        n_jobs: int = -1,
        xgb_params: dict | None = None,
        random_state: int = 42,
        solver: str = "rk4",
    ):
        self.nt = nt
        self.n_noise = n_noise
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.solver = solver

        self.xgb_params = xgb_params or {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        }

        self.t_levels_: np.ndarray | None = None
        self.models_: dict[int, dict[int, object]] = {}
        self.feature_types_: list[str] | None = None

        self.is_conditional_ = False
        self.d_target_ = 0
        self.d_condition_ = 0

        self.continuous_target_indices_: list[int] = []
        self.categorical_target_indices_: list[int] = []
        self.target_feature_types_: list[str] = []

    def _base_regressor(self, n_jobs_override: int | None = None) -> xgb.XGBRegressor:
        n_jobs = self.n_jobs if n_jobs_override is None else n_jobs_override
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_jobs=n_jobs,
            random_state=self.random_state,
            **self.xgb_params,
        )

    def _base_classifier(self, n_classes: int) -> xgb.XGBClassifier:
        params = {
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "max_depth": self.xgb_params.get("max_depth", 6),
            "learning_rate": self.xgb_params.get("learning_rate", 0.1),
            "subsample": self.xgb_params.get("subsample", 1.0),
            "colsample_bytree": self.xgb_params.get("colsample_bytree", 1.0),
            "n_estimators": self.xgb_params.get("n_estimators", 100),
            "eval_metric": "mlogloss",
        }
        if n_classes <= 2:
            params["objective"] = "binary:logistic"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = int(n_classes)
        return xgb.XGBClassifier(**params)

    @staticmethod
    @contextmanager
    def _tqdm_joblib(tqdm_bar):
        """Update tqdm by completed joblib batches (not dispatch count)."""

        class _TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_bar.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = _TqdmBatchCallback
        try:
            yield tqdm_bar
        finally:
            joblib.parallel.BatchCompletionCallBack = old_callback
            tqdm_bar.close()

    def _train_continuous_level(
        self,
        X: np.ndarray,
        X_condition: np.ndarray | None,
        t: float,
        t_index: int,
        reg_n_jobs: int,
        show_dim_progress: bool = False,
    ) -> tuple[int, dict[int, object]]:
        """Train all continuous target regressors at one flow level."""
        level_models: dict[int, object] = {}
        cont_iter = self.continuous_target_indices_
        if show_dim_progress and len(self.continuous_target_indices_) > 0:
            cont_iter = tqdm(
                self.continuous_target_indices_,
                desc=f"  Continuous dims @ t={t:.3f}",
                unit="dim",
                leave=False,
            )

        for j in cont_iter:
            x_j = X[:, j]
            x_rep = np.repeat(x_j, self.n_noise)
            z = (
                np.random.default_rng(self.random_state + t_index * 1009 + j)
                .standard_normal(len(x_rep))
                .astype(np.float32)
            )

            x_t = (1.0 - t) * z + t * x_rep
            y_t = x_rep - z

            cond_rep = (
                np.repeat(X_condition, self.n_noise, axis=0)
                if X_condition is not None
                else None
            )
            prev_rep = np.repeat(X[:, :j], self.n_noise, axis=0) if j > 0 else None

            X_in = self._compose_inputs(
                current_feature=x_t,
                prev_features=prev_rep,
                X_condition=cond_rep,
            )

            reg = self._base_regressor(n_jobs_override=reg_n_jobs)
            reg.fit(X_in, y_t)
            level_models[j] = reg

        return t_index, level_models

    def _compose_inputs(
        self,
        current_feature: np.ndarray | None,
        prev_features: np.ndarray | None,
        X_condition: np.ndarray | None,
    ) -> np.ndarray:
        parts: list[np.ndarray] = []
        if current_feature is not None:
            parts.append(current_feature.reshape(-1, 1).astype(np.float32))
        if X_condition is not None:
            parts.append(X_condition.astype(np.float32))
        if prev_features is not None and prev_features.shape[1] > 0:
            parts.append(prev_features.astype(np.float32))
        if not parts:
            raise ValueError("HS3F requires at least one input feature")
        return np.concatenate(parts, axis=1)

    def fit(
        self,
        X: np.ndarray,
        X_condition: np.ndarray | None = None,
        feature_types: list[str] | None = None,
    ) -> "HS3F":
        """Fit HS3F on target features with optional conditioning features."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        self.is_conditional_ = X_condition is not None
        self.d_target_ = X.shape[1]

        if X_condition is not None:
            X_condition = np.asarray(X_condition, dtype=np.float32)
            if X_condition.ndim != 2:
                raise ValueError("X_condition must be a 2D array")
            if len(X_condition) != len(X):
                raise ValueError("X and X_condition must have same number of rows")
            self.d_condition_ = X_condition.shape[1]
        else:
            self.d_condition_ = 0

        if feature_types is not None:
            expected = self.d_target_ + self.d_condition_
            if len(feature_types) != expected:
                raise ValueError(
                    f"feature_types length {len(feature_types)} doesn't match total features {expected}"
                )
            self.feature_types_ = feature_types
            self.target_feature_types_ = feature_types[: self.d_target_]
        else:
            self.feature_types_ = None
            self.target_feature_types_ = ["q"] * self.d_target_

        self.continuous_target_indices_ = [
            j for j, ft in enumerate(self.target_feature_types_) if ft == "q"
        ]
        self.categorical_target_indices_ = [
            j for j, ft in enumerate(self.target_feature_types_) if ft == "c"
        ]

        self.t_levels_ = np.array(
            [(i + 1) / self.nt for i in range(self.nt)], dtype=np.float32
        )

        # Train categorical models once per target feature
        categorical_models: dict[int, object] = {}
        cat_iter = self.categorical_target_indices_
        if len(self.categorical_target_indices_) > 0:
            cat_iter = tqdm(
                self.categorical_target_indices_,
                desc="HS3F categorical models",
                unit="feature",
                leave=False,
            )
        for j in cat_iter:
            y_original = np.round(X[:, j]).astype(int)
            unique_original = np.unique(y_original)
            if len(unique_original) < 2:
                # Degenerate category (single class); keep a constant model payload
                categorical_models[j] = {"constant": int(y_original[0])}
                continue

            prev = X[:, :j] if j > 0 else None
            X_in = self._compose_inputs(
                current_feature=None,
                prev_features=prev,
                X_condition=X_condition,
            )

            # XGBoost classifier expects contiguous class IDs [0, ..., K-1].
            # Subsampling can make observed label codes sparse (e.g., {0,2,5,7}).
            orig_to_local = {int(v): i for i, v in enumerate(unique_original.tolist())}
            local_to_orig = np.array(unique_original, dtype=np.int32)
            y_local = np.array(
                [orig_to_local[int(v)] for v in y_original], dtype=np.int32
            )

            clf = self._base_classifier(n_classes=len(unique_original))
            clf.fit(X_in, y_local)
            categorical_models[j] = {
                "model": clf,
                "local_to_orig": local_to_orig,
            }

        # Train flow regressors for continuous targets at each t level.
        # Parallelize over time levels when n_jobs != 1 to match ForestFlow.
        self.models_ = {}
        if self.n_jobs != 1 and len(self.t_levels_) > 1:
            print(
                f"HS3F level-parallel training enabled (n_jobs={self.n_jobs}); "
                "per-regressor XGBoost uses n_jobs=1 to avoid oversubscription."
            )
            with self._tqdm_joblib(
                tqdm(
                    total=len(self.t_levels_), desc="HS3F training levels", unit="level"
                )
            ):
                level_results = Parallel(
                    n_jobs=self.n_jobs,
                    verbose=0,
                    batch_size=1,  # Make progress updates respond per completed level.
                )(
                    delayed(self._train_continuous_level)(
                        X=X,
                        X_condition=X_condition,
                        t=float(t),
                        t_index=t_index,
                        reg_n_jobs=1,
                        show_dim_progress=False,
                    )
                    for t_index, t in enumerate(self.t_levels_)
                )
        else:
            level_results = []
            for t_index, t in enumerate(
                tqdm(self.t_levels_, desc="HS3F training levels", unit="level")
            ):
                level_results.append(
                    self._train_continuous_level(
                        X=X,
                        X_condition=X_condition,
                        t=float(t),
                        t_index=t_index,
                        reg_n_jobs=self.n_jobs,
                        show_dim_progress=True,
                    )
                )

        for t_index, level_models in level_results:
            # Keep categorical models available at every level for uniform access.
            for j, clf in categorical_models.items():
                level_models[j] = clf
            self.models_[t_index] = level_models

        return self

    def _predict_continuous_drift(
        self,
        model: object,
        x_state: np.ndarray,
        prev_generated: np.ndarray | None,
        X_condition: np.ndarray | None,
    ) -> np.ndarray:
        X_in = self._compose_inputs(
            current_feature=x_state,
            prev_features=prev_generated,
            X_condition=X_condition,
        )
        return model.predict(X_in).astype(np.float32)

    def _rk4_step(
        self,
        model: object,
        x_state: np.ndarray,
        h: float,
        prev_generated: np.ndarray | None,
        X_condition: np.ndarray | None,
    ) -> np.ndarray:
        k1 = self._predict_continuous_drift(model, x_state, prev_generated, X_condition)
        k2 = self._predict_continuous_drift(
            model, x_state + 0.5 * h * k1, prev_generated, X_condition
        )
        k3 = self._predict_continuous_drift(
            model, x_state + 0.5 * h * k2, prev_generated, X_condition
        )
        k4 = self._predict_continuous_drift(
            model, x_state + h * k3, prev_generated, X_condition
        )
        return x_state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def sample(
        self,
        n_samples: int,
        X_condition: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Generate synthetic target samples sequentially."""
        if not self.models_:
            raise RuntimeError("HS3F must be fit before sampling")

        if self.is_conditional_ and X_condition is None:
            raise ValueError(
                "Model was trained conditionally, must provide X_condition for sampling"
            )
        if not self.is_conditional_ and X_condition is not None:
            raise ValueError(
                "Model was trained in IID mode, should not provide X_condition"
            )

        if X_condition is not None:
            X_condition = np.asarray(X_condition, dtype=np.float32)
            if X_condition.shape != (n_samples, self.d_condition_):
                raise ValueError(
                    f"X_condition shape mismatch: expected ({n_samples}, {self.d_condition_}), got {X_condition.shape}"
                )

        rng = np.random.default_rng(
            random_state if random_state is not None else self.random_state
        )
        h = 1.0 / self.nt
        generated = np.zeros((n_samples, self.d_target_), dtype=np.float32)

        for j in range(self.d_target_):
            prev = generated[:, :j] if j > 0 else None

            if j in self.categorical_target_indices_:
                cat_model = self.models_[0][j]
                if isinstance(cat_model, dict) and "constant" in cat_model:
                    generated[:, j] = float(cat_model["constant"])
                    continue

                X_in = self._compose_inputs(
                    current_feature=None,
                    prev_features=prev,
                    X_condition=X_condition,
                )
                clf = cat_model["model"]
                local_to_orig = cat_model["local_to_orig"]
                probs = clf.predict_proba(X_in)

                sampled = np.empty(n_samples, dtype=np.int32)
                local_classes = clf.classes_.astype(np.int32)
                for i in range(n_samples):
                    sampled_local = int(rng.choice(local_classes, p=probs[i]))
                    sampled[i] = int(local_to_orig[sampled_local])
                generated[:, j] = sampled.astype(np.float32)
            else:
                x_state = rng.standard_normal(n_samples).astype(np.float32)

                for t_index in reversed(range(self.nt)):
                    reg = self.models_[t_index][j]
                    if self.solver == "euler":
                        drift = self._predict_continuous_drift(
                            reg, x_state, prev, X_condition
                        )
                        x_state = x_state + h * drift
                    elif self.solver == "rk4":
                        x_state = self._rk4_step(reg, x_state, h, prev, X_condition)
                    else:
                        raise ValueError(f"Unknown solver: {self.solver}")

                generated[:, j] = x_state

        return generated.astype(np.float32)
