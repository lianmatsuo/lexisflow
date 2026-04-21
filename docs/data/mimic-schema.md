# MIMIC-III Database Schema

## Overview

MIMIC-III is a relational database consisting of 26 tables. Tables are linked by identifiers which usually have the suffix 'ID'.

## Key Identifiers

- **SUBJECT_ID**: Unique patient identifier
- **HADM_ID**: Unique hospital admission identifier
- **ICUSTAY_ID**: Unique ICU stay identifier
- **ITEMID**: Concept identifier (join with D_ITEMS for definitions)

## Table Categories

### 1. Patient Stay Tables (5 tables)
- **ADMISSIONS** - Hospital admission records
- **PATIENTS** - Patient demographics
- **ICUSTAYS** - ICU stay records
- **SERVICES** - Care services provided
- **TRANSFERS** - Patient location transfers

### 2. Dictionary Tables (5 tables)
- **D_CPT** - Current Procedural Terminology codes
- **D_ICD_DIAGNOSES** - ICD-9 diagnosis codes
- **D_ICD_PROCEDURES** - ICD-9 procedure codes
- **D_ITEMS** - Definitions for charted items
- **D_LABITEMS** - Laboratory test definitions

### 3. Event Tables (9 tables)
- **CHARTEVENTS** - Vital signs and measurements
- **LABEVENTS** - Laboratory test results
- **OUTPUTEVENTS** - Output measurements (urine, drains)
- **INPUTEVENTS_CV** - Input events (CareVue system)
- **INPUTEVENTS_MV** - Input events (MetaVision system)
- **PROCEDUREEVENTS_MV** - Procedures performed
- **DATETIMEEVENTS** - Date/time events
- **MICROBIOLOGYEVENTS** - Microbiology test results
- **NOTEEVENTS** - Clinical notes

### 4. Other Tables (7 tables)
- **DIAGNOSES_ICD** - ICD-9 diagnoses
- **PROCEDURES_ICD** - ICD-9 procedures
- **CPTEVENTS** - CPT events
- **DRGCODES** - Diagnosis Related Group codes
- **PRESCRIPTIONS** - Medication prescriptions
- **CALLOUT** - Discharge readiness
- **CAREGIVERS** - Caregiver information

## Static Features (28 columns)

From `data/processed/static_data.csv`:

### Demographics (3)
- `gender` - M/F
- `ethnicity` - Patient ethnicity
- `age` - Age at admission

### Admission Info (4)
- `insurance` - Insurance type
- `diagnosis_at_admission` - Primary diagnosis
- `admission_type` - EMERGENCY/ELECTIVE/URGENT
- `first_careunit` - Initial ICU unit

### Outcomes (4)
- `mort_icu` - ICU mortality (0/1)
- `mort_hosp` - Hospital mortality (0/1)
- `hospital_expire_flag` - Expired flag (0/1)
- `readmission_30` - 30-day readmission (0/1)

### Code Status (9)
- `fullcode_first`, `fullcode` - Full code status
- `dnr_first`, `dnr` - DNR status
- `cmo_first`, `cmo`, `cmo_last` - Comfort measures
- `dnr_first_charttime`, `timecmo_chart` - Timestamps

### Timing (8)
- `admittime`, `intime`, `outtime` - Admission times
- `dischtime`, `deathtime` - Discharge/death times
- `los_icu` - Length of stay

## Dynamic Features (~326 columns)

From `data/processed/flat_table.csv`:

### Interventions
- **Ventilation**: vent, nivdurations
- **Vasopressors**: vaso, norepinephrine, dopamine, epinephrine, etc.
- **Medications**: Various drug administrations

### Vital Signs
- Heart rate
- Blood pressure
- Temperature
- Respiratory rate
- SpO2

### Lab Values
- Chemistry panels
- Blood counts
- Coagulation studies
- Liver function tests
- Renal function tests

### Fluid Balance
- Inputs: crystalloid_bolus, colloid_bolus
- Outputs: urine output, drainage

## Data Model

```
PATIENTS (subject_id) ──┐
                        ├──> ADMISSIONS (hadm_id) ──┐
                        │                           ├──> ICUSTAYS (icustay_id)
                        │                           │         │
                        │                           │         ├──> CHARTEVENTS
                        │                           │         ├──> LABEVENTS
                        │                           │         ├──> INPUTEVENTS_*
                        │                           │         └──> OUTPUTEVENTS
                        │                           │
                        │                           ├──> DIAGNOSES_ICD
                        │                           ├──> PROCEDURES_ICD
                        │                           └──> PRESCRIPTIONS
                        │
                        └──> (other patient tables)
```

## Usage in LexisFlow

### Identifying Static vs Dynamic

```python
from lexisflow.models.utils import load_static_columns, split_static_dynamic

# Load all static columns
static_cols = load_static_columns()  # Returns 28 columns

# Split a dataframe
df = pd.read_csv("data/processed/flat_table.csv")
static_cols, dynamic_cols = split_static_dynamic(df)
```

### Column Types

**Static (28):**
- Patient characteristics that don't change
- Admission-level information
- Outcome indicators

**Dynamic (326):**
- Time-varying measurements
- Hourly observations
- Interventions and treatments

## References

- [MIMIC-III Documentation](https://mimic.mit.edu/docs/iii/)
- [PhysioNet MIMIC-III](https://physionet.org/content/mimiciii/)

---

See also:
- [Data Processing](../../data/processed/COMPLETE_DATA_PROCESSING_DOCUMENTATION.md)
- [Feature Types](features.md)
