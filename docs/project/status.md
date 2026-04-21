# Project Status Tracker

**Last Updated:** April 13, 2026
**Current Phase:** Thesis Results Consolidation

---

## Completed

### Core pipeline and architecture
1. ✅ **CSV path and format standardization**
   - Unified under `src/lexisflow/data/processed/`
   - Legacy parquet/memmap paths removed

2. ✅ **Feature engineering and preprocessing workflow**
   - Autoregressive transformation with lag features
   - Full-data preprocessor fitting (`fit_autoregressive_preprocessor.py`)
   - Canonical preprocessor artifacts and transformed index metadata

3. ✅ **Model implementations**
   - ForestFlow backbone implemented
   - HS3F backbone implemented
   - Unified training interface (`train_autoregressive.py --model-type {forest-flow,hs3f}`)

4. ✅ **Hour-0 model workflow**
   - `prepare_hour0.py`
   - `fit_hour0_preprocessor.py`
   - `train_hour0.py`
   - `generate.py --use-hour0` integration completed

5. ✅ **Evaluation stack**
   - Quality metrics: KS, correlation Frobenius, clinical range violations
   - Multi-task TSTR: mortality, LOS (vasopressor task removed after framework revision to patient-level — see `report_latex/Chapters/Evaluation.tex`, "Revised design: patient-level sequence TSTR")
   - Privacy metrics: DCR diagnostics + DOMIAS-style membership inference

6. ✅ **Sweep infrastructure**
   - Unified sweep orchestration (`run_sweep.py`)
   - Caching, resumability, schema-stable logging
   - Visualization suite (`analyze_sweep.py`)

7. ✅ **Temporal coherence metrics implementation**
   - Added trajectory-level metrics in `src/lexisflow/evaluation/trajectory_metrics.py`
   - Metrics include autocorrelation distance, stay-length KS, transition smoothness ratio, temporal correlation drift
   - Integrated into sweep evaluation path (`run_sweep.py`)
   - Covered by dedicated unit tests (`src/lexisflow/evaluation/tests/test_trajectory_metrics.py`)

8. ✅ **CTGAN baseline integration (implementation stage)**
   - Added `CTGANAdapter` (`src/lexisflow/models/ctgan_adapter.py`)
   - Integrated into model registry and sweep training paths
   - Added matched-comparison runner (`scripts/run_backbone_comparison.py`) for HS3F vs ForestFlow vs CTGAN at a matched cell

9. ✅ **Testing**
   - 54 unit tests in `src/lexisflow/*/tests`
   - Core data/model/evaluation modules covered

10. ✅ **End-to-end pipeline verification**
   - Full workflow validated (`01 -> 01b -> 02 -> 03 -> 04`)
   - Expected outputs confirmed:
     - `results/synthetic_patients.csv`
     - `results/quality_metrics.txt`

11. ✅ **Report integration**
   - Sweep figures and expanded analysis integrated into `report_latex/`
   - Literature/citation coverage expanded (TabDDPM, TabSyn, STaSy, GReaT, REaLTabFormer, PromptEHR)
   - Novelty framing corrected to application-level contribution

---

## Current Capabilities

### What works now
1. ✅ End-to-end autoregressive generation pipeline
2. ✅ Fully synthetic generation via hour-0 + autoregressive rollout
3. ✅ Multi-task utility evaluation (mortality/LOS; patient-level)
4. ✅ Privacy metric computation in sweep outputs
5. ✅ Hyperparameter sweep plotting and analysis
6. ✅ CPU-first training with iterator-backed ForestFlow path
7. ✅ Trajectory-level temporal metrics wired into sweep evaluation
8. ✅ Single-cell matched backbone comparison runner (HS3F/ForestFlow/CTGAN)

### Current known constraints
1. 🟡 Single-dataset evidence (MIMIC-derived pipeline only)
2. 🟡 No full-grid external baseline comparison in main thesis tables (CTGAN currently smoke-level evidence)
3. 🟡 Main committed `results/sweep_results.csv` still reflects legacy single-seed runs
4. 🟡 Matched backbone evidence is currently single-cell/smoke, not full sweep grid
5. 🟡 No subgroup fairness slices in standard reports

---

## In Progress / Next Priority

### Thesis-critical empirical gaps
1. 🔜 **Matched backbone comparison**
   - Extend from current single-cell runner output to full matched grid evidence (HS3F vs ForestFlow)

2. 🔜 **Baseline comparison**
   - Promote CTGAN from smoke run to reproducible multi-cell baseline evidence in thesis tables

3. 🔜 **Statistical reliability**
   - Regenerate canonical sweep artifacts with multi-seed mean/std/CI columns
   - Ensure report tables/figures use uncertainty-aware outputs

4. 🔜 **Temporal realism evaluation**
   - Metrics implemented; remaining work is to report and interpret them in final thesis result sections

---

## Deferred (Intentional)

1. 🔜 **Advanced scalability engineering (post-thesis)**
   - Distributed training (Ray/Dask)
   - GPU acceleration experiments
   - Database-backed training for very large corpora
   - Large-scale parallel generation services

2. 🔜 **CI/CD and release engineering**
   - Automated pipeline/integration checks in CI
   - Packaging/publishing hardening

---

## Quick Action Checklist

### Immediate
- [x] End-to-end pipeline verification
- [x] Hyperparameter sweep visualization integrated into report
- [x] Report evaluation/results section expanded using current plots
- [x] Implement trajectory-level temporal metrics + tests
- [x] Add CTGAN baseline integration and matched-cell comparison runner
- [ ] Run full matched backbone comparison grid (not just single-cell)
- [ ] Produce baseline comparison tables from multi-cell runs

### Short term
- [ ] Regenerate canonical sweep outputs with multi-seed uncertainty columns
- [ ] Refresh thesis tables/figures from uncertainty-aware outputs
- [ ] Subgroup fairness slices

### Post-thesis
- [ ] Advanced scalability improvements
- [ ] CI/CD hardening

---

## Key Files

- **Workflow scripts:** `scripts/WORKFLOW.md`
- **Sweep design:** `docs/SWEEP_ARCHITECTURE.md`
- **Hour-0 design:** `docs/HOUR_0_MODEL_PLAN.md`
- **Architecture overview:** `docs/architecture/PROJECT_ARCHITECTURE.md`
- **Main report:** `report_latex/lexisflow_report.tex`

---

**Project Status:** 🟢 Functional and thesis-ready with clear next empirical milestones
**Code Quality:** 🟢 Strong modular structure + unit tests
**Primary Remaining Work:** Comparative evaluation breadth (baselines + matched backbone + uncertainty)
