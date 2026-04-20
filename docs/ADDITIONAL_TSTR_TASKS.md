# Additional TSTR Tasks Planning

## Overview

TSTR (Train on Synthetic, Test on Real) is a powerful utility-based evaluation that measures whether synthetic data is useful for downstream ML tasks. Currently, we only evaluate **mortality prediction**. This document outlines additional clinically relevant prediction tasks.

## Why Multiple TSTR Tasks?

1. **Robustness:** Single task may be too easy/hard or sensitive to specific features
2. **Coverage:** Different tasks test different aspects of the data
3. **Clinical relevance:** Real-world ICU predictions are multifaceted
4. **Publication quality:** Multiple TSTR tasks strengthen research claims

## Proposed TSTR Tasks

### 1. ✅ Mortality Prediction (Current)
**Status:** ✅ Already implemented

**Task:** Predict in-hospital mortality from ICU features
**Model:** Logistic Regression
**Features:** All dynamic features (aggregated) + static features
**Target:** `hospital_expire_flag` or `expire_flag`
**Metric:** AUROC, F1 score

---

### 2. 🔴 Length-of-Stay (LOS) Prediction
**Status:** 🔴 Not implemented

**Task:** Predict ICU length-of-stay category (short/medium/long)
**Model:** Multi-class Logistic Regression or Random Forest
**Features:** Hour-0 to Hour-24 features + demographics
**Target:** Binned `los_icu`:
- Short: 0-2 days
- Medium: 3-7 days
- Long: 8+ days

**Rationale:** LOS is critical for resource planning and reflects overall patient trajectory complexity.

**Implementation:**
```python
# src/synth_gen/mortality/los_classifier.py
class LOSClassifier:
    def __init__(self):
        self.model = LogisticRegression(multi_class='multinomial')

    def fit(self, X, los_hours):
        # Bin LOS into categories
        y = self._bin_los(los_hours)
        self.model.fit(X, y)

    def _bin_los(self, los_hours):
        # 0-48h=0, 48-168h=1, 168+h=2
        ...
```

---

### 3. 🔴 30-Day Readmission Prediction
**Status:** 🔴 Not implemented (requires linkage)

**Task:** Predict 30-day hospital readmission
**Model:** Logistic Regression
**Features:** Full ICU stay features + discharge characteristics
**Target:** Binary (readmitted within 30 days or not)

**Rationale:** Readmission is a key quality-of-care metric and tests long-term outcome prediction.

**Data Requirements:**
- Requires joining `ADMISSIONS` table on `SUBJECT_ID`
- Track subsequent admissions within 30 days of discharge
- This is **more complex** and requires additional preprocessing

**Implementation Complexity:** 🟡 Medium (need to link ADMISSIONS table)

---

### 4. 🔴 Vasopressor Requirement Prediction
**Status:** 🔴 Not implemented

**Task:** Predict if patient will require vasopressors in next 6 hours
**Model:** Logistic Regression or XGBoost
**Features:** Current + past 6 hours of vital signs
**Target:** Binary (`vaso` or `vasopressor` feature at t+6)

**Rationale:** Vasopressor need is a critical clinical decision and tests model's ability to capture hemodynamic instability patterns.

**Implementation:**
```python
# Create look-ahead target
df['vaso_next_6h'] = df.groupby('subject_id')['vaso'].shift(-6)

# Train classifier
X = features_at_current_hour
y = vaso_next_6h
```

**Implementation Complexity:** 🟢 Easy (data already available)

---

### 5. 🔴 Mechanical Ventilation Duration
**Status:** 🔴 Not implemented

**Task:** Predict total ventilation duration (regression)
**Model:** Linear Regression or XGBoost Regressor
**Features:** Hour-0 to Hour-24 features
**Target:** Total hours on ventilator (`vent` feature)

**Rationale:** Ventilation duration is clinically important and tests regression performance (not just classification).

**Implementation:**
```python
# Aggregate ventilation hours per patient
vent_duration = df.groupby('subject_id')['vent'].sum()

# Regression task
model = XGBRegressor()
model.fit(X_features, vent_duration)
```

**Implementation Complexity:** 🟢 Easy

---

### 6. 🔴 Sepsis Onset Prediction
**Status:** 🔴 Not implemented (requires Sepsis-3 criteria)

**Task:** Predict sepsis onset in next 12 hours
**Model:** Logistic Regression or XGBoost
**Features:** Current + past vital signs, labs
**Target:** Binary (sepsis onset within 12 hours)

**Rationale:** Sepsis is a major cause of ICU mortality and a clinically critical prediction task.

**Data Requirements:**
- Need to implement Sepsis-3 criteria (SOFA score changes)
- Requires lab results (lactate, WBC, etc.) from LABEVENTS
- **Complex** - may require additional preprocessing

**Implementation Complexity:** 🔴 Hard (need Sepsis-3 definition + lab data)

---

## Priority Ranking

| Task | Priority | Complexity | Clinical Importance | Implementation Effort |
|------|----------|------------|---------------------|----------------------|
| 1. Mortality | ✅ Done | 🟢 Easy | ⭐⭐⭐ High | Done |
| 2. Vasopressor | 🟢 High | 🟢 Easy | ⭐⭐⭐ High | ~2-3 hours |
| 3. LOS | 🟢 High | 🟢 Easy | ⭐⭐⭐ High | ~3-4 hours |
| 4. Vent Duration | 🟡 Medium | 🟢 Easy | ⭐⭐ Medium | ~2-3 hours |
| 5. Readmission | 🟡 Medium | 🟡 Medium | ⭐⭐ Medium | ~6-8 hours |
| 6. Sepsis Onset | 🔴 Low | 🔴 Hard | ⭐⭐⭐ High | ~12+ hours |

**Recommendation:** Implement tasks **2 (Vasopressor)** and **3 (LOS)** next for:
- Easy implementation
- High clinical relevance
- Good diversity (classification + multi-class)

## Unified TSTR Framework

### Proposed Architecture
**File:** `src/synth_gen/evaluation/tstr_framework.py`

```python
class TSTRTask(ABC):
    """Abstract base class for TSTR tasks."""

    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prepare X, y from raw dataframe."""
        pass

    @abstractmethod
    def get_metrics(self, y_true, y_pred) -> dict:
        """Compute task-specific metrics."""
        pass


class MortalityTask(TSTRTask):
    def prepare_features(self, df):
        # Existing mortality preparation logic
        ...

    def get_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auroc': roc_auc_score(y_true, y_pred),
        }


class LOSTask(TSTRTask):
    def prepare_features(self, df):
        # Bin LOS into categories
        ...

    def get_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
        }


# Unified evaluation
def evaluate_tstr_multi_task(
    synth_df, real_df, tasks: list[TSTRTask]
) -> dict:
    """Evaluate multiple TSTR tasks."""
    results = {}
    for task in tasks:
        results[task.name] = task.evaluate(synth_df, real_df)
    return results
```

### Benefits of Unified Framework
1. **Consistency:** All tasks use same data preparation
2. **Extensibility:** Easy to add new tasks
3. **Reporting:** Unified metrics reporting
4. **Comparison:** Easy to compare synthetic vs real performance across tasks

## Modified Sweep Integration

Update `run_sweep.py` to evaluate all tasks:

```python
# After generating synthetic data
print("\n  [3/4] TSTR Evaluation (Multi-Task)...")

tasks = [
    MortalityTask(),
    VasopressorTask(),
    LOSTask(),
]

multi_task_metrics = evaluate_tstr_multi_task(
    synth_df, real_test_df, tasks
)

# Record all metrics
result = {
    'nt': nt,
    'n_noise': n_noise,
    'train_time_sec': train_time,

    # Mortality
    'mortality_synth_auroc': multi_task_metrics['mortality']['synth_auroc'],
    'mortality_real_auroc': multi_task_metrics['mortality']['real_auroc'],

    # Vasopressor
    'vaso_synth_auroc': multi_task_metrics['vasopressor']['synth_auroc'],
    'vaso_real_auroc': multi_task_metrics['vasopressor']['real_auroc'],

    # LOS
    'los_synth_f1': multi_task_metrics['los']['synth_macro_f1'],
    'los_real_f1': multi_task_metrics['los']['real_macro_f1'],
}
```

## Implementation Order

### Short-Term (Thesis Submission)
1. ✅ Keep current mortality TSTR (proven working)
2. 🟢 Add vasopressor prediction TSTR (~2-3 hours)
3. 🟢 Add LOS prediction TSTR (~3-4 hours)
4. Document in thesis: "Evaluated across 3 distinct clinical tasks"

### Medium-Term (Post-Thesis)
5. 🟡 Add ventilation duration regression
6. 🟡 Implement unified TSTR framework
7. 🟡 Add readmission prediction (requires data linkage)

### Long-Term (If time permits)
8. 🔴 Implement Sepsis-3 criteria
9. 🔴 Add sepsis onset prediction
10. 🔴 Consider multi-task learning evaluation

## Expected Impact on Results

### Current (1 Task)
```
TSTR Performance:
✓ Mortality AUROC: 0.75 (synthetic) vs 0.78 (real)
```

### Proposed (3 Tasks)
```
TSTR Performance:
✓ Mortality AUROC: 0.75 (synth) vs 0.78 (real)
✓ Vasopressor AUROC: 0.72 (synth) vs 0.74 (real)
✓ LOS Macro-F1: 0.58 (synth) vs 0.61 (real)

Average utility gap: 3.2% (synth within 97% of real)
```

**Publication strength:** Multiple tasks demonstrate robust utility preservation.

## Risks & Mitigation

### Risk 1: Some tasks may fail completely
**Mitigation:** Start with easiest tasks (vasopressor, LOS) first

### Risk 2: Synthetic data may fail on harder tasks
**Mitigation:** This is actually valuable information (shows limitations)

### Risk 3: Time investment for thesis deadline
**Mitigation:** Implement 2-3 tasks maximum before thesis submission

## Conclusion

Adding 2-3 additional TSTR tasks will significantly strengthen the evaluation without requiring major architectural changes. The unified TSTR framework ensures maintainability and extensibility for future work.

**Recommended next steps:**
1. Implement vasopressor prediction (easiest)
2. Implement LOS prediction (high impact)
3. Update sweep to evaluate both tasks
4. Document in thesis
