"""Privacy metrics for synthetic tabular data.

This module implements a practical privacy-evaluation stack for synthetic data:

1) Distance to Closest Record (DCR) summary statistics.
2) DCR baseline protection score (relative to random baseline), following
   the formulation used in SDMetrics docs.
3) DCR overfitting protection score (synthetic proximity to training
   versus holdout records), also aligned with SDMetrics definitions.
4) A DOMIAS-inspired membership inference attack score based on local
   nearest-neighbor density ratios.

References:
- SDMetrics privacy docs (DCRBaselineProtection/DCROverfittingProtection)
- van Breugel et al. (AISTATS 2023), DOMIAS
- Shokri et al. (IEEE S&P 2017), membership inference
- Yao et al. (arXiv 2025), caution on DCR-only privacy claims
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

DEFAULT_ID_COLUMNS = (
    "subject_id",
    "hadm_id",
    "icustay_id",
    "hours_in",
)
MISSING_TOKEN = "__MISSING__"


def _empty_domias_metrics() -> dict[str, float]:
    """Return NaN-valued DOMIAS-like metrics when attack is not computable."""
    return {
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "attacker_advantage": float("nan"),
        "member_score_mean": float("nan"),
        "nonmember_score_mean": float("nan"),
    }


@dataclass
class MixedTabularDistanceTransformer:
    """Encode mixed tabular data for nearest-neighbor privacy metrics."""

    columns: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    numeric_min: dict[str, float]
    numeric_max: dict[str, float]
    numeric_fill: dict[str, float]
    category_values: dict[str, list[str]]

    @classmethod
    def fit(
        cls,
        real_train_df: pd.DataFrame,
        other_dfs: list[pd.DataFrame] | None = None,
        drop_columns: list[str] | None = None,
    ) -> "MixedTabularDistanceTransformer":
        """Fit a deterministic mixed-type encoder from real training data."""
        if other_dfs is None:
            other_dfs = []
        if drop_columns is None:
            drop_columns = []

        common_cols = set(real_train_df.columns)
        for df in other_dfs:
            common_cols &= set(df.columns)
        common_cols -= set(drop_columns)
        columns = sorted(common_cols)

        if not columns:
            raise ValueError(
                "No common columns remain after alignment. "
                "Check input tables and dropped ID columns."
            )

        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        for col in columns:
            if pd.api.types.is_numeric_dtype(real_train_df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        numeric_min: dict[str, float] = {}
        numeric_max: dict[str, float] = {}
        numeric_fill: dict[str, float] = {}
        for col in numeric_cols:
            vals = pd.to_numeric(real_train_df[col], errors="coerce")
            vmin = float(np.nanmin(vals.values)) if vals.notna().any() else 0.0
            vmax = float(np.nanmax(vals.values)) if vals.notna().any() else 1.0
            if vmin == vmax:
                vmax = vmin + 1.0
            fill = float(np.nanmedian(vals.values)) if vals.notna().any() else 0.0
            numeric_min[col] = vmin
            numeric_max[col] = vmax
            numeric_fill[col] = fill

        category_values: dict[str, list[str]] = {}
        stacked_dfs = [real_train_df] + other_dfs
        for col in categorical_cols:
            cats: list[str] = []
            seen = set()
            for df in stacked_dfs:
                if col not in df.columns:
                    continue
                series = df[col].astype("string").fillna(MISSING_TOKEN)
                for raw_value in series.values:
                    value = str(raw_value)
                    if value not in seen:
                        seen.add(value)
                        cats.append(value)
            if not cats:
                cats = [MISSING_TOKEN]
            category_values[col] = sorted(cats)

        return cls(
            columns=columns,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            numeric_min=numeric_min,
            numeric_max=numeric_max,
            numeric_fill=numeric_fill,
            category_values=category_values,
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame into a dense numeric matrix."""
        aligned = df.reindex(columns=self.columns)
        blocks: list[np.ndarray] = []

        if self.numeric_cols:
            num_df = aligned[self.numeric_cols].apply(pd.to_numeric, errors="coerce")
            for col in self.numeric_cols:
                num_df[col] = num_df[col].fillna(self.numeric_fill[col])
                denom = self.numeric_max[col] - self.numeric_min[col]
                num_df[col] = (num_df[col] - self.numeric_min[col]) / denom
            num_block = num_df[self.numeric_cols].to_numpy(dtype=np.float64, copy=False)
            blocks.append(num_block)

        if self.categorical_cols:
            cat_blocks: list[np.ndarray] = []
            for col in self.categorical_cols:
                categories = self.category_values[col]
                vals = aligned[col].astype("string").fillna(MISSING_TOKEN).astype(str)
                cat = pd.Categorical(vals, categories=categories)
                dummies = pd.get_dummies(cat, prefix=None)
                dummies = dummies.reindex(columns=categories, fill_value=False)
                cat_blocks.append(dummies.to_numpy(dtype=np.float64, copy=False))
            cat_block = np.hstack(cat_blocks) if cat_blocks else np.zeros((len(df), 0))
            blocks.append(cat_block)

        if not blocks:
            return np.zeros((len(df), 0), dtype=np.float64)
        return np.hstack(blocks)

    def sample_random(self, n_rows: int, random_state: int = 42) -> pd.DataFrame:
        """Generate random baseline rows following SDMetrics-style logic."""
        rng = np.random.default_rng(random_state)
        data: dict[str, np.ndarray] = {}

        for col in self.numeric_cols:
            low = self.numeric_min[col]
            high = self.numeric_max[col]
            data[col] = rng.uniform(low=low, high=high, size=n_rows)

        for col in self.categorical_cols:
            categories = self.category_values[col]
            data[col] = rng.choice(categories, size=n_rows, replace=True)

        return pd.DataFrame(data, columns=self.numeric_cols + self.categorical_cols)


def _subsample_rows(
    df: pd.DataFrame,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Optionally subsample rows for computationally expensive NN metrics."""
    if max_rows is None or len(df) <= max_rows:
        return df.copy()
    return (
        df.sample(n=max_rows, random_state=random_state, replace=False)
        .reset_index(drop=True)
        .copy()
    )


def _knn_distances(
    query: np.ndarray,
    reference: np.ndarray,
    n_neighbors: int = 1,
) -> np.ndarray:
    """Compute nearest-neighbor distances from query to reference."""
    if len(reference) == 0:
        raise ValueError("Reference matrix is empty; cannot compute nearest neighbors.")
    if len(query) == 0:
        return np.array([], dtype=np.float64)

    k = min(n_neighbors, len(reference))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(reference)
    distances, _ = nn.kneighbors(query, return_distance=True)
    return distances


def compute_distance_to_closest_record(
    real_train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    id_columns: list[str] | tuple[str, ...] = DEFAULT_ID_COLUMNS,
    max_rows: int | None = 5000,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute DCR and nearest-neighbor ratio summary statistics."""
    real = _subsample_rows(real_train_df, max_rows=max_rows, random_state=random_state)
    synth = _subsample_rows(synthetic_df, max_rows=max_rows, random_state=random_state)

    transformer = MixedTabularDistanceTransformer.fit(
        real_train_df=real,
        other_dfs=[synth],
        drop_columns=list(id_columns),
    )
    real_encoded = transformer.transform(real)
    synth_encoded = transformer.transform(synth)

    d1 = _knn_distances(synth_encoded, real_encoded, n_neighbors=1)[:, 0]
    if len(real_encoded) >= 2:
        d2 = _knn_distances(synth_encoded, real_encoded, n_neighbors=2)[:, 1]
        nndr = np.divide(
            d1,
            np.maximum(d2, 1e-12),
            out=np.zeros_like(d1),
            where=np.isfinite(d2),
        )
        nndr_median = float(np.median(nndr))
    else:
        nndr_median = float("nan")

    return {
        "dcr_mean": float(np.mean(d1)) if len(d1) > 0 else float("nan"),
        "dcr_median": float(np.median(d1)) if len(d1) > 0 else float("nan"),
        "dcr_p05": float(np.percentile(d1, 5)) if len(d1) > 0 else float("nan"),
        "dcr_p95": float(np.percentile(d1, 95)) if len(d1) > 0 else float("nan"),
        "exact_match_rate": float(np.mean(d1 <= 1e-12))
        if len(d1) > 0
        else float("nan"),
        "nndr_median": nndr_median,
    }


def compute_dcr_baseline_protection(
    real_train_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    id_columns: list[str] | tuple[str, ...] = DEFAULT_ID_COLUMNS,
    max_rows: int | None = 5000,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute SDMetrics-style DCR baseline protection score."""
    real = _subsample_rows(real_train_df, max_rows=max_rows, random_state=random_state)
    synth = _subsample_rows(synthetic_df, max_rows=max_rows, random_state=random_state)

    transformer = MixedTabularDistanceTransformer.fit(
        real_train_df=real,
        other_dfs=[synth],
        drop_columns=list(id_columns),
    )
    real_encoded = transformer.transform(real)
    synth_encoded = transformer.transform(synth)
    random_df = transformer.sample_random(len(synth), random_state=random_state)
    random_encoded = transformer.transform(random_df)

    synth_dcr = _knn_distances(synth_encoded, real_encoded, n_neighbors=1)[:, 0]
    random_dcr = _knn_distances(random_encoded, real_encoded, n_neighbors=1)[:, 0]

    synth_median = float(np.median(synth_dcr)) if len(synth_dcr) > 0 else float("nan")
    random_median = (
        float(np.median(random_dcr)) if len(random_dcr) > 0 else float("nan")
    )

    if np.isnan(synth_median) or np.isnan(random_median) or random_median <= 0:
        score = float("nan")
    else:
        score = float(min(1.0, synth_median / random_median))

    return {
        "score": score,
        "median_dcr_to_real_synthetic": synth_median,
        "median_dcr_to_real_random": random_median,
    }


def compute_dcr_overfitting_protection(
    real_train_df: pd.DataFrame,
    real_holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    id_columns: list[str] | tuple[str, ...] = DEFAULT_ID_COLUMNS,
    max_rows: int | None = 5000,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute SDMetrics-style DCR overfitting protection score."""
    train = _subsample_rows(real_train_df, max_rows=max_rows, random_state=random_state)
    holdout = _subsample_rows(
        real_holdout_df,
        max_rows=max_rows,
        random_state=random_state,
    )
    synth = _subsample_rows(synthetic_df, max_rows=max_rows, random_state=random_state)

    transformer = MixedTabularDistanceTransformer.fit(
        real_train_df=train,
        other_dfs=[holdout, synth],
        drop_columns=list(id_columns),
    )
    train_encoded = transformer.transform(train)
    holdout_encoded = transformer.transform(holdout)
    synth_encoded = transformer.transform(synth)

    d_train = _knn_distances(synth_encoded, train_encoded, n_neighbors=1)[:, 0]
    d_holdout = _knn_distances(synth_encoded, holdout_encoded, n_neighbors=1)[:, 0]

    closer_to_train = float(np.mean(d_train < d_holdout))
    closer_to_holdout = float(np.mean(d_train >= d_holdout))
    score = float(min(1.0, max(0.0, (1.0 - closer_to_train) * 2.0)))

    return {
        "score": score,
        "closer_to_training_pct": closer_to_train,
        "closer_to_holdout_pct": closer_to_holdout,
        "median_dcr_to_training": float(np.median(d_train)),
        "median_dcr_to_holdout": float(np.median(d_holdout)),
    }


def compute_domias_like_membership_inference(
    real_train_df: pd.DataFrame,
    real_holdout_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    id_columns: list[str] | tuple[str, ...] = DEFAULT_ID_COLUMNS,
    max_rows: int | None = 3000,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute a DOMIAS-inspired membership inference attack score.

    The attack uses nearest-neighbor density ratio surrogates:
    score(x) = log((d_ref(x) + eps) / (d_syn(x) + eps)).

    Larger scores indicate records closer to synthetic data than to an
    attacker reference set (holdout split), and are therefore more likely
    to be inferred as members.
    """
    if len(real_holdout_df) < 4:
        return _empty_domias_metrics()

    train = _subsample_rows(real_train_df, max_rows=max_rows, random_state=random_state)
    holdout = _subsample_rows(
        real_holdout_df,
        max_rows=max_rows,
        random_state=random_state,
    )
    synth = _subsample_rows(synthetic_df, max_rows=max_rows, random_state=random_state)

    holdout_reference, holdout_candidates = train_test_split(
        holdout,
        test_size=0.5,
        random_state=random_state,
        shuffle=True,
    )
    n_candidates = min(len(train), len(holdout_candidates))
    if n_candidates < 4:
        return _empty_domias_metrics()

    members = train.sample(n=n_candidates, random_state=random_state).reset_index(
        drop=True
    )
    non_members = holdout_candidates.sample(
        n=n_candidates,
        random_state=random_state,
    ).reset_index(drop=True)

    transformer = MixedTabularDistanceTransformer.fit(
        real_train_df=train,
        other_dfs=[holdout_reference, holdout_candidates, synth],
        drop_columns=list(id_columns),
    )
    synth_encoded = transformer.transform(synth)
    ref_encoded = transformer.transform(holdout_reference)
    member_encoded = transformer.transform(members)
    non_member_encoded = transformer.transform(non_members)

    eps = 1e-12
    d_syn_member = _knn_distances(member_encoded, synth_encoded, n_neighbors=1)[:, 0]
    d_ref_member = _knn_distances(member_encoded, ref_encoded, n_neighbors=1)[:, 0]
    d_syn_non = _knn_distances(non_member_encoded, synth_encoded, n_neighbors=1)[:, 0]
    d_ref_non = _knn_distances(non_member_encoded, ref_encoded, n_neighbors=1)[:, 0]

    member_scores = np.log((d_ref_member + eps) / (d_syn_member + eps))
    non_member_scores = np.log((d_ref_non + eps) / (d_syn_non + eps))

    y_true = np.concatenate([np.ones(n_candidates), np.zeros(n_candidates)])
    y_score = np.concatenate([member_scores, non_member_scores])

    roc_auc = float(roc_auc_score(y_true, y_score))
    avg_precision = float(average_precision_score(y_true, y_score))

    threshold = 0.0
    member_hit = float(np.mean(member_scores > threshold))
    non_member_false_hit = float(np.mean(non_member_scores > threshold))
    attacker_advantage = float(member_hit - non_member_false_hit)

    return {
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "attacker_advantage": attacker_advantage,
        "member_score_mean": float(np.mean(member_scores)),
        "nonmember_score_mean": float(np.mean(non_member_scores)),
    }


def compute_privacy_metrics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    holdout_df: pd.DataFrame | None = None,
    id_columns: list[str] | tuple[str, ...] = DEFAULT_ID_COLUMNS,
    holdout_fraction: float = 0.3,
    max_rows: int | None = 5000,
    random_state: int = 42,
) -> dict[str, dict[str, float] | int]:
    """Compute a complete privacy report for synthetic tabular data."""
    if holdout_df is None:
        real_train, real_holdout = train_test_split(
            real_df,
            test_size=holdout_fraction,
            random_state=random_state,
            shuffle=True,
        )
        real_train = real_train.reset_index(drop=True)
        real_holdout = real_holdout.reset_index(drop=True)
    else:
        real_train = real_df.reset_index(drop=True)
        real_holdout = holdout_df.reset_index(drop=True)

    dcr_stats = compute_distance_to_closest_record(
        real_train_df=real_train,
        synthetic_df=synthetic_df,
        id_columns=id_columns,
        max_rows=max_rows,
        random_state=random_state,
    )
    dcr_baseline = compute_dcr_baseline_protection(
        real_train_df=real_train,
        synthetic_df=synthetic_df,
        id_columns=id_columns,
        max_rows=max_rows,
        random_state=random_state,
    )
    dcr_overfit = compute_dcr_overfitting_protection(
        real_train_df=real_train,
        real_holdout_df=real_holdout,
        synthetic_df=synthetic_df,
        id_columns=id_columns,
        max_rows=max_rows,
        random_state=random_state,
    )
    domias_like = compute_domias_like_membership_inference(
        real_train_df=real_train,
        real_holdout_df=real_holdout,
        synthetic_df=synthetic_df,
        id_columns=id_columns,
        max_rows=max_rows,
        random_state=random_state,
    )

    return {
        "n_real_train": int(len(real_train)),
        "n_real_holdout": int(len(real_holdout)),
        "n_synthetic": int(len(synthetic_df)),
        "dcr_stats": dcr_stats,
        "dcr_baseline_protection": dcr_baseline,
        "dcr_overfitting_protection": dcr_overfit,
        "membership_inference_domias_like": domias_like,
    }


def _format_metric(v: float) -> str:
    if np.isnan(v):
        return "nan"
    return f"{v:.4f}"


def main() -> None:
    """CLI entrypoint for privacy-metric evaluation."""
    parser = argparse.ArgumentParser(
        description="Compute synthetic-data privacy metrics (DCR + membership inference)."
    )
    parser.add_argument(
        "--real-data",
        type=str,
        required=True,
        help="Path to real data CSV (training real data, or full real data if no holdout).",
    )
    parser.add_argument(
        "--synthetic-data",
        type=str,
        required=True,
        help="Path to synthetic data CSV.",
    )
    parser.add_argument(
        "--holdout-data",
        type=str,
        default=None,
        help="Optional holdout real CSV for overfitting and membership inference.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/privacy_metrics.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.3,
        help="If --holdout-data is not provided, fraction split from --real-data.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Optional subsampling cap per dataset for faster NN computations.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--id-cols",
        nargs="*",
        default=list(DEFAULT_ID_COLUMNS),
        help="Columns to exclude from distance and attack metrics.",
    )
    args = parser.parse_args()

    real_df = pd.read_csv(args.real_data, low_memory=False)
    synthetic_df = pd.read_csv(args.synthetic_data, low_memory=False)
    holdout_df = (
        pd.read_csv(args.holdout_data, low_memory=False) if args.holdout_data else None
    )

    report = compute_privacy_metrics(
        real_df=real_df,
        synthetic_df=synthetic_df,
        holdout_df=holdout_df,
        id_columns=args.id_cols,
        holdout_fraction=args.holdout_fraction,
        max_rows=args.max_rows,
        random_state=args.random_state,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("PRIVACY METRICS SUMMARY")
    print("=" * 72)
    print(f"Real train rows:   {report['n_real_train']:,}")
    print(f"Real holdout rows: {report['n_real_holdout']:,}")
    print(f"Synthetic rows:    {report['n_synthetic']:,}")
    print("")
    print("[DCR Stats]")
    dcr = report["dcr_stats"]
    print(f"  DCR median:      {_format_metric(float(dcr['dcr_median']))}")
    print(f"  DCR p05:         {_format_metric(float(dcr['dcr_p05']))}")
    print(f"  Exact match rate:{_format_metric(float(dcr['exact_match_rate']))}")
    print("")
    print("[DCR Baseline Protection]")
    dcr_base = report["dcr_baseline_protection"]
    print(f"  Score:           {_format_metric(float(dcr_base['score']))}")
    print("")
    print("[DCR Overfitting Protection]")
    dcr_overfit = report["dcr_overfitting_protection"]
    print(f"  Score:           {_format_metric(float(dcr_overfit['score']))}")
    print(
        "  Closer to train: "
        f"{_format_metric(float(dcr_overfit['closer_to_training_pct']))}"
    )
    print("")
    print("[Membership Inference (DOMIAS-like)]")
    mia = report["membership_inference_domias_like"]
    print(f"  ROC-AUC:         {_format_metric(float(mia['roc_auc']))}")
    print("  Attacker adv.:   " f"{_format_metric(float(mia['attacker_advantage']))}")
    print("")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
