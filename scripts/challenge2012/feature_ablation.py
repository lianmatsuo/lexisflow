#!/usr/bin/env python3
"""Minimal focused feature ablation for Challenge2012 TSTR.

Compares:
1) Full feature set
2) Top-K clinically predictive subset (ranked on real mortality signal)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from lexisflow.config import get_dataset_config
from lexisflow.evaluation.tstr_framework import LOSTask, MortalityTask
from lexisflow.sweep import generate_synthetic_data, load_autoregressive_inputs
from lexisflow.sweep.generation import create_flat_dataframe


def _parse_int_list(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return vals


def _pick_model_path(cfg, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"Model not found: {explicit}")
        return explicit

    if not cfg.results_csv.exists():
        raise SystemExit(f"Results CSV not found: {cfg.results_csv}")

    df = pd.read_csv(cfg.results_csv)
    ok = df[(df.get("error").isna()) & (df.get("degenerate_flag", 0).fillna(0) == 0)]
    if ok.empty:
        raise SystemExit("No successful non-degenerate rows found in results CSV.")

    best = ok.sort_values("synth_roc_auc", ascending=False).iloc[0]
    nt = int(best["nt"])
    n_noise = int(best["n_noise"])
    path = cfg.sweep_models_dir / f"autoregressive_nt{nt}_noise{n_noise}.pkl"
    if not path.exists():
        raise SystemExit(f"Best model artifact not found: {path}")
    return path


def _patient_level_first_rows(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["subject_id", "hours_in"])
        .groupby("subject_id", as_index=False)
        .first()
    )


def _rank_topk_features(real_df: pd.DataFrame, top_k: int) -> list[str]:
    base_exclude = {
        "subject_id",
        "hours_in",
        "hadm_id",
        "icustay_id",
        "trajectory_id",
        "hospital_expire_flag",
        "mort_icu",
        "mort_hosp",
        "los_icu",
        "los_hospital",
    }
    first = _patient_level_first_rows(real_df)
    y = pd.to_numeric(first["hospital_expire_flag"], errors="coerce")
    feature_cols = [c for c in first.columns if c not in base_exclude]
    X = first[feature_cols].apply(pd.to_numeric, errors="coerce")

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)
    if y.nunique() < 2:
        raise SystemExit("Mortality labels have <2 classes in real data.")

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med).fillna(0.0)

    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
        C=0.5,
    )
    clf.fit(X_train, y_train)

    coef = np.abs(clf.coef_).ravel()
    ranking = pd.Series(coef, index=X_train.columns).sort_values(ascending=False)
    top = ranking.head(top_k).index.tolist()
    return top


def _evaluate_pair(
    synth_df: pd.DataFrame, real_df: pd.DataFrame, eval_seed: int
) -> dict[str, float]:
    mort = MortalityTask(
        random_state=eval_seed, use_sequence_model=True, sequence_max_patients=2500
    )
    los = LOSTask(
        random_state=eval_seed, use_sequence_model=True, sequence_max_patients=2500
    )
    m = mort.evaluate(synth_df, real_df, test_size=0.3, verbose=False)
    los_result = los.evaluate(synth_df, real_df, test_size=0.3, verbose=False)
    return {
        "mortality_synth_roc_auc": float(m.get("synth_roc_auc", np.nan)),
        "mortality_real_roc_auc": float(m.get("real_roc_auc", np.nan)),
        "los_synth_macro_f1": float(los_result.get("synth_macro_f1", np.nan)),
        "los_real_macro_f1": float(los_result.get("real_macro_f1", np.nan)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused feature ablation for Challenge2012."
    )
    parser.add_argument("--dataset", default="challenge2012", choices=["challenge2012"])
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--top-k-values", type=_parse_int_list, default=[8, 12, 16, 24])
    parser.add_argument("--synth-samples", type=int, default=24000)
    parser.add_argument("--seeds", type=_parse_int_list, default=[7, 42, 99])
    args = parser.parse_args()

    cfg = get_dataset_config(args.dataset)
    model_path = _pick_model_path(cfg, args.model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)["model"]

    inputs = load_autoregressive_inputs(
        train_rows=20000, dataset=args.dataset, refresh_cache=False, show_progress=False
    )
    real_df = pd.read_csv(inputs.real_test_path, low_memory=False)
    if len(real_df) > 50000:
        real_df = real_df.sample(50000, random_state=42).reset_index(drop=True)

    max_k = max(args.top_k_values)
    ranked_features = _rank_topk_features(real_df, max_k)
    top_features_by_k = {k: ranked_features[:k] for k in args.top_k_values}

    rows: list[dict] = []
    for seed in args.seeds:
        X_synth, X_cond, sid, hrs = generate_synthetic_data(
            model, inputs.X_cond_train, args.synth_samples, trajectory_seed=seed
        )
        synth_df = create_flat_dataframe(
            X_synth,
            X_cond,
            inputs.preprocessor,
            inputs.target_indices,
            inputs.condition_indices,
            subject_id=sid,
            hours_in=hrs,
        )

        full_metrics = _evaluate_pair(synth_df, real_df, eval_seed=seed)
        rows.append(
            {
                "seed": seed,
                "setting": "full",
                "n_features": int(synth_df.shape[1] - 2),
                **full_metrics,
            }
        )

        for k in args.top_k_values:
            top_features = top_features_by_k[k]
            keep = [
                "subject_id",
                "hours_in",
                "hospital_expire_flag",
                "los_icu",
            ] + top_features
            keep = [c for c in keep if c in real_df.columns and c in synth_df.columns]
            subset_metrics = _evaluate_pair(
                synth_df[keep].copy(),
                real_df[keep].copy(),
                eval_seed=seed,
            )
            rows.append(
                {
                    "seed": seed,
                    "setting": f"top_{k}",
                    "n_features": len(top_features),
                    **subset_metrics,
                }
            )

    out_df = pd.DataFrame(rows)
    out_path = cfg.results_csv.parent / "challenge2012_feature_ablation.csv"
    out_df.to_csv(out_path, index=False)
    summary = (
        out_df.groupby(["setting", "n_features"], as_index=False)
        .agg(
            mortality_auc_mean=("mortality_synth_roc_auc", "mean"),
            mortality_auc_std=("mortality_synth_roc_auc", "std"),
            los_f1_mean=("los_synth_macro_f1", "mean"),
            los_f1_std=("los_synth_macro_f1", "std"),
        )
        .sort_values("mortality_auc_mean", ascending=False)
    )
    summary_path = cfg.results_csv.parent / "challenge2012_feature_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"model: {model_path}")
    print(f"top_features_max_k({max_k}): {ranked_features[:max_k]}")
    print("\nPer-seed results:")
    print(out_df.to_string(index=False))
    print("\nSummary (mean/std across seeds):")
    print(summary.to_string(index=False))
    print(f"saved: {out_path}")
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
