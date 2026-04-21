# Static vs Dynamic Features

Understanding the difference between static and dynamic features is crucial for autoregressive modeling.

## Static Features (28 columns)

**Definition:** Patient characteristics that don't change over time during the ICU stay.

### Categories

#### Demographics (3)
```
gender, ethnicity, age
```
- Fixed at admission
- Used for conditioning in autoregressive models

#### Admission Information (4)
```
insurance, diagnosis_at_admission, admission_type, first_careunit
```
- Determined at hospital entry
- Remain constant throughout stay

#### Outcomes (4)
```
mort_icu, mort_hosp, hospital_expire_flag, readmission_30
```
- ⚠️ **Caution:** These are future information
- Consider excluding from conditioning or treating as separate targets

#### Other Static (17)
- Code status flags
- Admission/discharge timestamps
- Hospital stay metadata

### Usage in Models

```python
# Static features are used for conditioning
P(X_t | X_{t-1}, Static_Features)
           ↑           ↑
      dynamic      static (constant)
```

## Dynamic Features (326 columns)

**Definition:** Measurements that change over time (hourly observations).

### Categories

#### Interventions
- **Ventilation** (2 cols): vent, nivdurations
- **Vasopressors** (8+ cols): vaso, norepinephrine, dopamine, etc.
- **Medications** (many cols): Various drug administrations

#### Vital Signs
- Heart rate, blood pressure, temperature
- Respiratory rate, SpO2
- Glasgow Coma Scale components

#### Lab Values (100+ cols)
- Chemistry: sodium, potassium, glucose, lactate, etc.
- Hematology: WBC, hemoglobin, platelets
- Coagulation: PT, PTT, INR
- Liver: bilirubin, ALT, AST
- Renal: creatinine, BUN

#### Fluid Balance
- **Inputs**: crystalloid_bolus, colloid_bolus
- **Outputs**: urine output, drainage volumes

### Usage in Models

```python
# Dynamic features are predicted
P(Dynamic_Features_t | Dynamic_Features_{t-1}, Static)
        ↑                        ↑
    targets              lagged history
```

## Autoregressive Data Structure

### Before Transformation
```csv
subject_id, hours_in, hr, bp, age, gender
1,          0,        80, 120, 65, M
1,          1,        82, 118, 65, M
1,          2,        79, 115, 65, M
```

### After `prepare_autoregressive_data()`
```csv
hours_in, hr, bp, age, gender, hr_lag1, bp_lag1
0,        80, 120, 65, M,      -1,      -1       # no history
1,        82, 118, 65, M,      80,      120      # lag from t=0
2,        79, 115, 65, M,      82,      118      # lag from t=1
```

**Target columns:** `['hr', 'bp']` (dynamic)
**Condition columns:** `['age', 'gender', 'hr_lag1', 'bp_lag1']` (static + lags)

## Loading Features

### Automatic Detection

```python
from lexisflow.data.autoregressive import split_static_dynamic

df = pd.read_csv("data/processed/flat_table.csv")
static_cols, dynamic_cols = split_static_dynamic(df)

print(f"Static: {len(static_cols)}")    # 28
print(f"Dynamic: {len(dynamic_cols)}")  # 326
```

### Manual Specification

```python
# Define static columns explicitly
static_cols = [
    'gender', 'ethnicity', 'age',
    'insurance', 'admission_type', 'first_careunit'
]

# All others are dynamic
exclude = ['subject_id', 'hadm_id', 'icustay_id', 'hours_in']
dynamic_cols = [c for c in df.columns
                if c not in static_cols + exclude]
```

## Feature Engineering Tips

### 1. Exclude Outcome Variables

Outcome variables are "future information":

```python
# Remove outcomes from static features
outcomes = ['mort_icu', 'mort_hosp', 'hospital_expire_flag']
static_cols = [c for c in static_cols if c not in outcomes]
```

### 2. Handle Timestamps

Timestamps (admittime, dischtime) may need special handling:

```python
# Option 1: Exclude timestamps
static_cols = [c for c in static_cols if 'time' not in c.lower()]

# Option 2: Convert to numeric (hours since admission)
df['hours_since_admit'] = (df['current_time'] - df['admittime']).dt.total_seconds() / 3600
```

### 3. Categorize Lab Values

Group related labs:

```python
chemistry_labs = ['sodium', 'potassium', 'chloride', 'bicarbonate']
renal_labs = ['creatinine', 'bun', 'gfr']
liver_labs = ['bilirubin', 'alt', 'ast', 'alp']
```

### 4. Handle Missing Values

Different strategies:

```python
# Preprocessing handles missing values
preprocessor = TabularPreprocessor(
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    max_missing_ratio=None  # XGBoost handles NaNs natively
)
```

## Common Pitfalls

### ❌ Wrong: Treating Dynamic as Static

```python
# BAD: heart_rate doesn't stay constant!
static_cols = ['age', 'gender', 'heart_rate']
```

### ✅ Correct: Proper Categorization

```python
# GOOD: Only truly static features
static_cols = ['age', 'gender', 'admission_type']
dynamic_cols = ['heart_rate', 'blood_pressure', ...]
```

### ❌ Wrong: Including IDs as Features

```python
# BAD: IDs are not features
feature_cols = ['subject_id', 'age', 'heart_rate']
```

### ✅ Correct: Exclude IDs

```python
# GOOD: IDs used only for grouping
id_cols = ['subject_id', 'hadm_id', 'icustay_id']
feature_cols = [c for c in df.columns if c not in id_cols + ['hours_in']]
```

## Validation Checklist

- [ ] Static columns don't change over time for same patient
- [ ] Dynamic columns vary across timesteps
- [ ] No IDs in feature columns
- [ ] Outcome variables handled appropriately
- [ ] Timestamps converted or excluded
- [ ] Missing values strategy defined

## References

- [MIMIC-III Schema](mimic-schema.md)
- [Data Processing](processing.md)
