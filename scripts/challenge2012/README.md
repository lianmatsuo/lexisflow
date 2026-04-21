# PhysioNet Challenge 2012 — Reproducibility Example

A publicly available (ODC-BY, no credentialing) ICU benchmark used to
demonstrate the `lexisflow` pipeline end-to-end without MIMIC-III access.
4,000 ICU patients, 48-hour hourly trajectories, mortality + LOS labels —
semantically aligned with the MIMIC-III flow.

## Dataset

- **Source:** [PhysioNet Challenge 2012 set-A](https://physionet.org/content/challenge-2012/1.0.0/)
- **Size:** 4,000 patients × 48 hourly bins (189,120 rows after gridding)
- **Features retained:** 39 (static demographics + vital medians;
  `TroponinI` and `Cholesterol` dropped for <10% coverage; `SAPS-I` / `SOFA`
  not used).
- **Labels:** `hospital_expire_flag` (binary), `los_icu` (days).
- **License:** Open Data Commons Attribution (ODC-BY) — no PhysioNet
  credentialing required.

## Download

```bash
mkdir -p data/challenge2012/raw
cd data/challenge2012/raw
wget https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz
tar -xzf set-a.tar.gz
wget https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt
```

## Pipeline

All commands run from the repo root.

```bash
# Single command: prepare data + fit preprocessors + run sweep
uv run python scripts/run_sweep.py --dataset challenge2012

# Optional: clear generated challenge artifacts and rerun from scratch
uv run python scripts/run_sweep.py --dataset challenge2012 --reset
```

Results are written to `results/challenge2012_sweep_results.csv` and include:

- **TSTR utility:** Mortality AUROC, LOS macro-F1 (synth vs real baselines).
- **Distribution:** avg KS statistic, correlation Frobenius, range violations.
- **Temporal:** stay-length KS, transition smoothness, autocorrelation,
  temporal-correlation drift.
- **Privacy:** DCR median / p05, DCR baseline + overfitting protection,
  DOMIAS-style MIA ROC-AUC.

## Design notes

- Column names (`subject_id`, `hours_in`, `hospital_expire_flag`, `los_icu`)
  match MIMIC-III so `MortalityTask`, `LOSTask`, trajectory metrics, and
  privacy metrics are reused **without modification**.
- Labels (`hospital_expire_flag`, `los_icu`) are encoded as static condition
  columns so they survive the inverse transform in the synthetic DataFrame
  (same trick as the MIMIC preprocessor — see
  `artifacts/preprocessor_full.pkl`).
- Patient-disjoint 60/20/20 train/test/holdout split inside `prepare_autoregressive.py`
  guards the privacy overfitting-protection metric from train/holdout overlap.
- Imputation (ffill→bfill within patient → global median) is required for the
  flow-matching target matrix and the downstream `MinMaxScaler` /
  `LogisticRegression`; XGBoost would tolerate NaN inputs but they cannot be
  flowed-through as generator targets.
