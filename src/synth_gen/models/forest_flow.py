"""Forest-Flow generative model using XGBoost and Flow Matching."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from tqdm import tqdm

from .iterator import FlowMatchingDataIterator


class ForestFlow:
    """Flow-matching generative model with XGBoost vector fields.

    Implements Independent Conditional Flow Matching (I-CFM) where the
    vector field is approximated using Gradient-Boosted Trees at each
    discrete time level.

    Supports autoregressive conditional generation: P(X_t | X_{t-1}, Static).
    """

    def __init__(
        self,
        nt: int = 50,
        n_noise: int = 100,
        n_jobs: int = -1,
        xgb_params: dict | None = None,
        random_state: int = 42,
        use_data_iterator: bool = True,
        batch_size: int = 1000,
        n_target_dims: int | None = None,
        n_condition_dims: int | None = None,
    ):
        """Initialize ForestFlow.

        Args:
            nt: Number of discrete time levels (flow time 0 to 1).
            n_noise: Number of Gaussian noise samples per data point.
            n_jobs: Number of parallel jobs for training (-1 = all CPUs).
            xgb_params: Optional dict of XGBRegressor parameters.
            random_state: Random seed for reproducibility.
            use_data_iterator: If True, use XGBoost 2.0+ data iterator (avoids duplication).
            batch_size: Batch size for data iterator (only used if use_data_iterator=True).
            n_target_dims: Number of target dimensions (variables to generate).
                          If None, assumes IID mode (all features are targets).
            n_condition_dims: Number of condition dimensions (history/static features).
                             If None, assumes IID mode (no conditioning).
        """
        self.nt = nt
        self.n_noise = n_noise
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_data_iterator = use_data_iterator
        self.batch_size = batch_size
        self.n_target_dims = n_target_dims
        self.n_condition_dims = n_condition_dims

        # Default XGBoost parameters (as per paper)
        self.xgb_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "tree_method": "hist",
            "n_jobs": 1,  # Parallelism at time-level, not inside XGB
            "random_state": random_state,
            "enable_categorical": True,  # XGBoost 2.0+ native categorical support
            "max_cat_to_onehot": 4,  # Only one-hot if ≤4 categories (use partition-based otherwise)
        }
        if xgb_params:
            self.xgb_params.update(xgb_params)

        # models_ can be either MultiOutputRegressor or list of XGBRegressor (per dimension)
        self.models_: dict[int, MultiOutputRegressor | list] = {}
        self.t_levels_: np.ndarray | None = None
        self.d_target_: int | None = None  # Target dimensions
        self.d_condition_: int | None = None  # Condition dimensions
        self.is_conditional_: bool = False
        self.feature_types_: list[str] | None = (
            None  # XGBoost feature types ('q' or 'c')
        )

    def _xgb_train_params(self) -> dict:
        """Build stable params for xgb.train.

        `enable_categorical` is intentionally omitted here: in XGBoost 3.1.x
        the learner does not consume it in the native training path and emits
        repeated "not used" warnings. Categorical handling is still driven by
        `feature_types` passed to DMatrix/QuantileDMatrix.
        """
        return {
            "objective": "reg:squarederror",
            "max_depth": self.xgb_params.get("max_depth", 6),
            "learning_rate": self.xgb_params.get("learning_rate", 0.1),
            "subsample": self.xgb_params.get("subsample", 1.0),
            "colsample_bytree": self.xgb_params.get("colsample_bytree", 1.0),
            "reg_alpha": self.xgb_params.get("reg_alpha", 0.0),
            "reg_lambda": self.xgb_params.get("reg_lambda", 0.0),
            "tree_method": self.xgb_params.get("tree_method", "hist"),
            "max_cat_to_onehot": self.xgb_params.get("max_cat_to_onehot", 4),
        }

    def _duplicate_and_noise(
        self, X_target: np.ndarray, X_condition: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Duplicate data and sample Gaussian noise.

        Args:
            X_target: Target data array of shape (n, d_target).
            X_condition: Optional condition data of shape (n, d_condition).

        Returns:
            X_target_dup: Duplicated target data of shape (n * n_noise, d_target).
            X_condition_dup: Duplicated condition data or None.
            Z: Gaussian noise of shape (n * n_noise, d_target).
        """
        rng = np.random.default_rng(self.random_state)
        X_target_dup = np.repeat(X_target, self.n_noise, axis=0)
        Z = rng.standard_normal(X_target_dup.shape)

        if X_condition is not None:
            X_condition_dup = np.repeat(X_condition, self.n_noise, axis=0)
        else:
            X_condition_dup = None

        return X_target_dup, X_condition_dup, Z

    def _train_level_with_iterator(
        self,
        X_target: np.ndarray,
        X_condition: np.ndarray | None,
        t: float,
        t_index: int,
    ) -> tuple[int, list]:
        """Train model for a single time level using data iterator (XGBoost 2.0+).

        This avoids duplicating data n_noise times by generating noise on-the-fly.
        Trains one XGBRegressor per output dimension.

        Args:
            X_target: Target data to add noise to, shape (n, d_target).
            X_condition: Optional condition data (no noise), shape (n, d_condition).
            t: Time level value (flow time 0 to 1).
            t_index: Index of this time level.

        Returns:
            Tuple of (t_index, list of fitted models per dimension).
        """
        n_samples, n_dims = X_target.shape
        models_per_dim = []

        for dim in tqdm(range(n_dims), desc=f"  t={t:.3f}", leave=False, unit="dim"):
            # Create iterator for this dimension
            iterator = FlowMatchingDataIterator(
                X_target,
                t,
                self.n_noise,
                self.batch_size,
                random_state=self.random_state + t_index * n_dims + dim,
                X_condition=X_condition,
                target_dim=dim,  # Only yield targets for this dimension
            )

            # Try streaming training with QuantileDMatrix (XGBoost 2.0+)
            # This avoids loading all data into memory at once
            try:
                # XGBoost 2.0+ supports QuantileDMatrix with iterators
                dtrain = xgb.QuantileDMatrix(
                    iterator.as_dmatrix_iterator(),
                    max_bin=256,  # Histogram bins for memory efficiency
                    feature_types=self.feature_types_,  # Pass categorical feature types
                )

                # Extract XGBoost parameters and train with native API
                params = self._xgb_train_params()

                bst = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.xgb_params.get("n_estimators", 100),
                    verbose_eval=False,
                )

                # Wrap in a simple predictor class for compatibility
                class XGBoostPredictor:
                    def __init__(self, booster, feature_types):
                        self.booster = booster
                        self.feature_types = feature_types

                    def predict(self, X):
                        dtest = xgb.DMatrix(X, feature_types=self.feature_types)
                        return self.booster.predict(dtest)

                model = XGBoostPredictor(bst, self.feature_types_)

            except (AttributeError, TypeError):
                # Fallback: QuantileDMatrix not available or doesn't support iterators
                # Collect batches (uses more memory but works with older XGBoost)
                X_t_all = []
                Y_t_all = []
                for X_t_batch, Y_t_batch in iterator:
                    X_t_all.append(X_t_batch)
                    Y_t_all.append(Y_t_batch)

                X_t_full = np.vstack(X_t_all)
                Y_t_full = np.concatenate(Y_t_all)

                # Train model for this dimension with categorical support
                # XGBRegressor doesn't directly support feature_types in fit,
                # so we use the xgb.train API wrapper
                if self.feature_types_:
                    dtrain = xgb.DMatrix(
                        X_t_full, label=Y_t_full, feature_types=self.feature_types_
                    )
                    params = self._xgb_train_params()
                    bst = xgb.train(
                        params,
                        dtrain,
                        num_boost_round=self.xgb_params.get("n_estimators", 100),
                        verbose_eval=False,
                    )

                    class XGBoostPredictor:
                        def __init__(self, booster, feature_types):
                            self.booster = booster
                            self.feature_types = feature_types

                        def predict(self, X):
                            dtest = xgb.DMatrix(X, feature_types=self.feature_types)
                            return self.booster.predict(dtest)

                    model = XGBoostPredictor(bst, self.feature_types_)
                else:
                    # No categorical features - use sklearn interface
                    model = XGBRegressor(**self.xgb_params)
                    model.fit(X_t_full, Y_t_full)

            models_per_dim.append(model)

        return t_index, models_per_dim

    def _train_level(
        self,
        X_target_dup: np.ndarray,
        X_condition_dup: np.ndarray | None,
        Z: np.ndarray,
        t: float,
        t_index: int,
    ) -> tuple[int, MultiOutputRegressor]:
        """Train model for a single time level (legacy approach with duplication).

        Args:
            X_target_dup: Duplicated target data.
            X_condition_dup: Duplicated condition data (no noise).
            Z: Gaussian noise.
            t: Time level value (flow time 0 to 1).
            t_index: Index of this time level.

        Returns:
            Tuple of (t_index, fitted model).
        """
        # Noisy target
        X_t_noisy = (1 - t) * Z + t * X_target_dup

        # Target vector field
        Y_t = X_target_dup - Z

        # If conditional, concatenate clean condition features
        if X_condition_dup is not None:
            X_t = np.concatenate([X_t_noisy, X_condition_dup], axis=1)
        else:
            X_t = X_t_noisy

        # Check for NaN in targets
        if np.isnan(Y_t).any():
            n_nan = np.isnan(Y_t).any(axis=1).sum()
            print(f"  Warning: At t={t:.3f}, {n_nan} rows have NaN in Y_t")

        # Train multi-output model with categorical support if needed
        if self.feature_types_:
            # Train separate model per dimension using xgb.train API for categorical support
            models_per_dim = []
            for dim in range(Y_t.shape[1]):
                dtrain = xgb.DMatrix(
                    X_t, label=Y_t[:, dim], feature_types=self.feature_types_
                )
                params = self._xgb_train_params()
                bst = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.xgb_params.get("n_estimators", 100),
                    verbose_eval=False,
                )

                class XGBoostPredictor:
                    def __init__(self, booster, feature_types):
                        self.booster = booster
                        self.feature_types = feature_types

                    def predict(self, X):
                        dtest = xgb.DMatrix(X, feature_types=self.feature_types)
                        return self.booster.predict(dtest)

                models_per_dim.append(XGBoostPredictor(bst, self.feature_types_))

            # Return list of models (compatible with sample() method)
            return t_index, models_per_dim
        else:
            # No categorical features - use sklearn interface
            model = MultiOutputRegressor(XGBRegressor(**self.xgb_params))
            model.fit(X_t, Y_t)
            return t_index, model

    def fit(
        self,
        X: np.ndarray,
        X_condition: np.ndarray | None = None,
        feature_types: list[str] | None = None,
    ) -> "ForestFlow":
        """Fit the Forest-Flow model on preprocessed data.

        Args:
            X: Preprocessed target data array of shape (n, d_target).
               In IID mode, this is all features.
               In conditional mode, this is only the dynamic/target features.
               XGBoost handles NaNs in inputs.
            X_condition: Optional condition data of shape (n, d_condition).
                        These features are kept clean (no noise) during training.
                        Use for autoregressive conditioning or static features.
            feature_types: Optional list of feature types for XGBoost categorical support.
                          Each element is 'q' (quantitative) or 'c' (categorical).
                          Length must match total features (target + condition).
                          Format: ['q', 'q', ..., 'c', 'c'] for [target_features..., condition_features...]

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        # Determine mode
        self.is_conditional_ = X_condition is not None
        self.d_target_ = X.shape[1]

        if self.is_conditional_:
            X_condition = np.asarray(X_condition, dtype=np.float32)
            if X_condition.ndim != 2:
                raise ValueError("X_condition must be a 2D array")
            if len(X_condition) != len(X):
                raise ValueError("X and X_condition must have same number of rows")
            self.d_condition_ = X_condition.shape[1]
            print("Training conditional ForestFlow:")
            print(f"  Target dims: {self.d_target_} (noise applied)")
            print(f"  Condition dims: {self.d_condition_} (kept clean)")

            # CRITICAL: Split feature_types correctly for flow matching
            # Target features get noise → must be 'q' (can't use 'c' with fractional/negative values)
            # Condition features stay clean → can be 'c' if originally categorical
            if feature_types is not None:
                if len(feature_types) != self.d_target_ + self.d_condition_:
                    raise ValueError(
                        f"feature_types length {len(feature_types)} doesn't match total features {self.d_target_ + self.d_condition_}"
                    )
                # Override target features to 'q' (they get noise)
                self.feature_types_ = ["q"] * self.d_target_ + feature_types[
                    self.d_target_ :
                ]
                n_categorical_in_condition = sum(
                    1 for ft in feature_types[self.d_target_ :] if ft == "c"
                )
                if n_categorical_in_condition > 0:
                    print(
                        f"  Categorical features: {n_categorical_in_condition} (in condition only)"
                    )
            else:
                self.feature_types_ = None
        else:
            self.d_condition_ = 0
            print("Training IID ForestFlow:")
            print(f"  Feature dims: {self.d_target_}")

            # In IID mode, ALL features get noise → must all be 'q'
            # Cannot use 'c' type because noise makes them fractional/negative
            if feature_types is not None:
                self.feature_types_ = ["q"] * self.d_target_
                n_was_categorical = sum(1 for ft in feature_types if ft == "c")
                if n_was_categorical > 0:
                    print(
                        f"  Note: {n_was_categorical} categorical features treated as 'q' (all features get noise in IID mode)"
                    )
            else:
                self.feature_types_ = None

        # Time levels: [1/nt, 2/nt, ..., 1.0] (flow time, not patient time)
        self.t_levels_ = np.array([(i + 1) / self.nt for i in range(self.nt)])

        if self.use_data_iterator:
            print(
                f"  Using XGBoost 2.0+ data iterator (no {self.n_noise}x duplication)"
            )
            # Use data iterator approach - avoids pre-duplicating data
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self._train_level_with_iterator)(X, X_condition, t, t_index)
                for t_index, t in tqdm(
                    enumerate(self.t_levels_),
                    desc="  Training time levels",
                    total=len(self.t_levels_),
                    unit="level",
                )
            )
        else:
            print(f"  Using legacy duplication approach ({self.n_noise}x duplication)")
            # Legacy approach - duplicate data once
            X_target_dup, X_condition_dup, Z = self._duplicate_and_noise(X, X_condition)
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(self._train_level)(X_target_dup, X_condition_dup, Z, t, t_index)
                for t_index, t in tqdm(
                    enumerate(self.t_levels_),
                    desc="  Training time levels",
                    total=len(self.t_levels_),
                    unit="level",
                )
            )

        # Collect results
        self.models_ = {t_index: model for t_index, model in results}

        return self

    def sample(
        self,
        n_samples: int,
        X_condition: np.ndarray | None = None,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Generate synthetic samples via backward ODE integration.

        Args:
            n_samples: Number of samples to generate.
            X_condition: Optional condition data of shape (n_samples, d_condition).
                        Required if model was trained conditionally.
            random_state: Optional random seed (uses self.random_state if None).

        Returns:
            Synthetic target samples of shape (n_samples, d_target) in preprocessed space.
        """
        if not self.models_:
            raise RuntimeError("ForestFlow must be fit before sampling")

        # Validate conditional mode
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
                    f"X_condition shape mismatch: expected ({n_samples}, {self.d_condition_}), "
                    f"got {X_condition.shape}"
                )

        seed = random_state if random_state is not None else self.random_state
        rng = np.random.default_rng(seed)

        # Initialize target from noise
        X_t = rng.standard_normal((n_samples, self.d_target_)).astype(np.float32)

        # Step size (flow time)
        h = 1.0 / self.nt

        # Backward integration: flow time from 1 to 0
        for t_index in tqdm(
            reversed(range(self.nt)),
            desc="  Sampling steps",
            total=self.nt,
            unit="step",
        ):
            model = self.models_[t_index]

            # Prepare input: [Noisy_Target, Clean_Condition] if conditional
            if X_condition is not None:
                X_input = np.concatenate([X_t, X_condition], axis=1)
            else:
                X_input = X_t

            if isinstance(model, list):
                # Data iterator approach: list of models per dimension
                Y_hat = np.zeros((n_samples, self.d_target_), dtype=np.float32)
                for dim, dim_model in enumerate(model):
                    Y_hat[:, dim] = dim_model.predict(X_input)
            else:
                # Legacy approach: MultiOutputRegressor
                Y_hat = model.predict(X_input)

            X_t = X_t + h * Y_hat

        return X_t


def fit_label_conditional(
    X: np.ndarray,
    y: np.ndarray,
    nt: int = 50,
    n_noise: int = 100,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict:
    """Fit separate ForestFlow models per label (for conditional generation).

    Args:
        X: Preprocessed data of shape (n, d).
        y: Label array of shape (n,).
        nt, n_noise, n_jobs, random_state: ForestFlow parameters.

    Returns:
        Dict with 'models' (label -> ForestFlow) and 'label_probs' (label -> prob).
    """
    unique_labels, counts = np.unique(y, return_counts=True)
    label_probs = dict(zip(unique_labels, counts / len(y)))

    models = {}
    for label in unique_labels:
        mask = y == label
        X_label = X[mask]
        ff = ForestFlow(
            nt=nt, n_noise=n_noise, n_jobs=n_jobs, random_state=random_state
        )
        ff.fit(X_label)
        models[label] = ff

    return {"models": models, "label_probs": label_probs}


def sample_label_conditional(
    conditional_result: dict,
    n_samples: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from label-conditional ForestFlow models.

    Args:
        conditional_result: Output from fit_label_conditional.
        n_samples: Total number of samples to generate.
        random_state: Optional random seed.

    Returns:
        X_synth: Synthetic samples of shape (n_samples, d).
        y_synth: Synthetic labels of shape (n_samples,).
    """
    rng = np.random.default_rng(random_state)
    models = conditional_result["models"]
    label_probs = conditional_result["label_probs"]

    labels = list(label_probs.keys())
    probs = np.array([label_probs[label] for label in labels])

    # Sample labels according to empirical distribution
    sampled_labels = rng.choice(labels, size=n_samples, p=probs)

    # Count samples per label
    label_counts = {label: np.sum(sampled_labels == label) for label in labels}

    # Generate samples per label
    X_parts = []
    y_parts = []
    for label, count in label_counts.items():
        if count > 0:
            X_label = models[label].sample(int(count), random_state=random_state)
            X_parts.append(X_label)
            y_parts.append(np.full(count, label))

    X_synth = np.vstack(X_parts)
    y_synth = np.concatenate(y_parts)

    # Shuffle
    idx = rng.permutation(len(X_synth))
    return X_synth[idx], y_synth[idx]
