"""Flow matching data iterator for memory-efficient training."""

from __future__ import annotations

import numpy as np
from typing import Iterator, Tuple


class FlowMatchingDataIterator:
    """Data iterator for XGBoost that generates training pairs on-the-fly.

    This implements the approach from XGBoost 2.0+ mentioned in the paper
    (footnote 2, Section 3.1) to avoid duplicating the dataset n_noise times.
    Instead, we generate noise samples on-the-fly in minibatches.

    Supports conditional generation where conditions are kept clean (no noise).
    """

    def __init__(
        self,
        X_target: np.ndarray,
        t: float,
        n_noise: int,
        batch_size: int = 1000,
        random_state: int = 42,
        X_condition: np.ndarray | None = None,
        target_dim: int | None = None,
    ):
        """Initialize the data iterator.

        Args:
            X_target: Target data to add noise to, shape (n, d_target).
            t: Time level for this iterator (flow time 0 to 1).
            n_noise: Number of noise samples per data point.
            batch_size: Size of each batch.
            random_state: Random seed.
            X_condition: Optional condition data (no noise), shape (n, d_condition).
                        If provided, will be concatenated to noisy target.
            target_dim: If specified, only yield targets for this dimension (for per-dim training).
        """
        self.X_target = X_target
        self.X_condition = X_condition
        self.t = t
        self.n_noise = n_noise
        self.batch_size = batch_size
        self.rng = np.random.default_rng(random_state)
        self.target_dim = target_dim

        self.n_samples = len(X_target) * n_noise
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        self._iter_count = 0

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate through batches of (X_t, Y_t) pairs.

        Yields:
            X_t: Input features [Noisy_Target, Clean_Condition] if conditional,
                 else just [Noisy_Target].
            Y_t: Target vector field (Target - Noise) or single dimension if target_dim is set.
        """
        self._iter_count += 1
        # Re-seed for each iteration to get different noise
        rng = np.random.default_rng(self.rng.integers(0, 2**31) + self._iter_count)

        for batch_idx in range(self.n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.n_samples)

            # Always yield full batch_size for QuantileDMatrix compatibility
            # Wrap around if needed for last batch
            batch_actual_size = end_idx - start_idx
            if batch_actual_size < self.batch_size:
                # Last batch is smaller - wrap around to get full batch
                indices_1 = np.arange(start_idx, end_idx)
                indices_2 = np.arange(0, self.batch_size - batch_actual_size)
                full_indices = np.concatenate([indices_1, indices_2])
            else:
                full_indices = np.arange(start_idx, end_idx)

            # Determine which original samples this batch corresponds to
            data_indices = (full_indices % len(self.X_target)).astype(int)

            # Get the corresponding target samples
            X_target_batch = self.X_target[data_indices]

            # Generate fresh noise for this batch
            Z_batch = rng.standard_normal(X_target_batch.shape, dtype=np.float32)

            # Compute noisy target: X_t = (1-t)*Z + t*X
            X_t_noisy = (1 - self.t) * Z_batch + self.t * X_target_batch

            # Target vector field: Y_t = X - Z
            Y_t = X_target_batch - Z_batch

            # Extract specific dimension if requested
            if self.target_dim is not None:
                Y_t = Y_t[:, self.target_dim]

            # If conditional, concatenate clean condition features
            if self.X_condition is not None:
                X_condition_batch = self.X_condition[data_indices]
                X_t = np.concatenate([X_t_noisy, X_condition_batch], axis=1)
            else:
                X_t = X_t_noisy

            yield X_t, Y_t

    def __len__(self) -> int:
        """Return number of batches."""
        return self.n_batches

    def as_dmatrix_iterator(self):
        """Return an iterator compatible with xgboost.QuantileDMatrix.

        Yields:
            Tuple of (data, label) for each batch.
        """
        for X_batch, Y_batch in self:
            yield X_batch, Y_batch
