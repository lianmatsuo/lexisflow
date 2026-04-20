# Complete Data Processing & Cleaning Documentation
## From Raw MIMIC-III to Cleaned Flat Table

**Author**: Generated from MIMICExtractEasy pipeline
**Date**: December 8, 2025
**Dataset**: MIMIC-III v1.4 → MIMIC-Extract → Cleaned Flat Table

---

## Table of Contents

1. [Overview](#overview)
2. [Original MIMIC-III Dataset](#original-mimic-iii-dataset)
3. [Database Setup & Concepts](#database-setup--concepts)
4. [MIMIC-Extract Pipeline](#mimic-extract-pipeline)
5. [Data Processing Steps](#data-processing-steps)
6. [Post-Extraction Cleaning](#post-extraction-cleaning)
7. [Final Data Structure](#final-data-structure)
8. [Feature Dictionary](#feature-dictionary)
9. [SQL Queries Reference](#sql-queries-reference)

---

## Overview

This document provides complete documentation of the data processing pipeline from the raw MIMIC-III database to the final cleaned flat table dataset. The pipeline consists of three major stages:

1. **Database Build**: Converting CSV files to DuckDB database with concept tables
2. **MIMIC-Extract**: Cohort selection, data extraction, cleaning, and hourly aggregation
3. **Post-Processing**: Flat table creation and missing data cleanup

### Pipeline Summary

```
Raw MIMIC-III CSV files (7GB, 26 tables)
    ↓ [DuckDB Build + Concepts]
DuckDB Database with derived concept tables
    ↓ [MIMIC-Extract]
Extracted hourly data (4 files: static, vitals, outcomes, subjects)
    ↓ [Flat Table Creation]
Combined flat_table.csv (2.0GB, 358 columns, 2.2M rows)
    ↓ [Missing Data Cleanup]
flat_table_cleaned.csv (2.0GB, 159 columns, 2.2M rows)
```

---

## Original MIMIC-III Dataset

### Source Data

**MIMIC-III (Medical Information Mart for Intensive Care)** is a freely-accessible critical care database developed by MIT Lab for Computational Physiology.

- **Version**: MIMIC-III v1.4
- **Size**: ~7GB (26 CSV files)
- **Patients**: 46,520 patients, 58,976 hospital admissions
- **ICU Stays**: 61,532 ICU stays
- **Time Period**: 2001-2012 at Beth Israel Deaconess Medical Center

### Core Tables

| Table | Description | Rows |
|-------|-------------|------|
| `PATIENTS` | Patient demographics | 46,520 |
| `ADMISSIONS` | Hospital admissions | 58,976 |
| `ICUSTAYS` | ICU stays | 61,532 |
| `CHARTEVENTS` | Charted vital signs and observations | 330M+ |
| `LABEVENTS` | Laboratory measurements | 27M+ |
| `INPUTEVENTS_MV` | Fluid inputs (MetaVision) | 3.6M |
| `INPUTEVENTS_CV` | Fluid inputs (CareVue) | 17.5M |
| `OUTPUTEVENTS` | Fluid outputs | 4.3M |
| `PROCEDUREEVENTS_MV` | Procedures | 258K |
| `DIAGNOSES_ICD` | ICD-9 diagnosis codes | 651K |
| `D_ITEMS` | Item dictionary | 12,487 items |
| `D_LABITEMS` | Lab item dictionary | 753 items |

---

## Database Setup & Concepts

### 1. DuckDB Database Creation

The first step converts raw CSV files into an embedded DuckDB database:

```bash
cd mimic-code/mimic-iii/buildmimic/duckdb/
python3 import_duckdb.py ${MIMIC_DATA_DIR} ${DATA_DIR}/mimic3.db --skip-indexes
```

**What This Does**:
- Loads all 26 MIMIC-III CSV tables into DuckDB
- Creates appropriate data types and schemas
- Enables fast SQL queries without PostgreSQL setup

### 2. Concept Tables Generation

MIMIC-code provides ~100 concept tables that derive clinically meaningful information:

```bash
python3 import_duckdb.py ${MIMIC_DATA_DIR} ${DATA_DIR}/mimic3.db \
    --skip-tables --make-concepts
```

#### Key Concept Tables Used by MIMIC-Extract

| Concept Table | Purpose | Source Query |
|---------------|---------|--------------|
| `icustay_detail` | Demographic & administrative details per ICU stay | `concepts/demographics/icustay_detail.sql` |
| `code_status` | DNR/DNI/CMO status over time | `concepts/code_status.sql` |
| `sofa` | SOFA severity score (Sepsis-related Organ Failure Assessment) | `concepts/severityscores/sofa.sql` |
| `sapsii` | SAPS II severity score (Simplified Acute Physiology Score) | `concepts/severityscores/sapsii.sql` |
| `oasis` | OASIS severity score | `concepts/severityscores/oasis.sql` |
| `ventilation_durations` | Mechanical ventilation periods | `concepts/durations/ventilation_durations.sql` |
| `vasopressor_durations` | Any vasopressor use periods | `concepts/durations/vasopressor_durations.sql` |
| `adenosine_durations` | Adenosine infusion periods | `concepts/durations/adenosine_durations.sql` |
| `dobutamine_durations` | Dobutamine infusion periods | `concepts/durations/dobutamine_durations.sql` |
| `dopamine_durations` | Dopamine infusion periods | `concepts/durations/dopamine_durations.sql` |
| `epinephrine_durations` | Epinephrine infusion periods | `concepts/durations/epinephrine_durations.sql` |
| `isuprel_durations` | Isuprel infusion periods | `concepts/durations/isuprel_durations.sql` |
| `milrinone_durations` | Milrinone infusion periods | `concepts/durations/milrinone_durations.sql` |
| `norepinephrine_durations` | Norepinephrine infusion periods | `concepts/durations/norepinephrine_durations.sql` |
| `phenylephrine_durations` | Phenylephrine infusion periods | `concepts/durations/phenylephrine_durations.sql` |
| `vasopressin_durations` | Vasopressin infusion periods | `concepts/durations/vasopressin_durations.sql` |
| `colloid_bolus` | Colloid bolus administration | `concepts/fluid_balance/colloid_bolus.sql` |
| `crystalloid_bolus` | Crystalloid bolus administration | `concepts/fluid_balance/crystalloid_bolus.sql` |
| `nivdurations` | Non-invasive ventilation periods | `utils/niv-durations.sql` |

---

## MIMIC-Extract Pipeline

### Execution Command

```bash
cd MIMIC_Extract/
python3 mimic_direct_extract.py \
  --duckdb_database=${DATA_DIR}/mimic3.db \
  --duckdb_schema=main \
  --resource_path=./resources \
  --plot_hist=0 \
  --out_path=${DATA_DIR}/extract
```

### Pipeline Stages

The `mimic_direct_extract.py` script performs the following stages:

1. **Cohort Selection** (Static Data)
2. **Vitals & Labs Extraction**
3. **Interventions & Outcomes Extraction**
4. **Data Cleaning & Validation**
5. **Hourly Aggregation**
6. **Output Generation**

---

## Data Processing Steps

### Stage 1: Cohort Selection & Static Data Extraction

#### SQL Query: Static Data

**File**: `MIMIC_Extract/SQL_Queries/statics.sql`

```sql
SELECT DISTINCT
    i.subject_id,
    i.hadm_id,
    i.icustay_id,
    i.gender,
    i.admission_age as age,
    i.ethnicity,
    i.hospital_expire_flag,
    i.hospstay_seq,
    i.los_icu,
    i.admittime,
    i.dischtime,
    i.intime,
    i.outtime,
    a.diagnosis AS diagnosis_at_admission,
    a.admission_type,
    a.insurance,
    a.deathtime,
    a.discharge_location,
    CASE WHEN a.deathtime BETWEEN i.intime AND i.outtime THEN 1 ELSE 0 END AS mort_icu,
    CASE WHEN a.deathtime BETWEEN i.admittime AND i.dischtime THEN 1 ELSE 0 END AS mort_hosp,
    s.first_careunit,
    c.fullcode_first,
    c.dnr_first,
    c.fullcode,
    c.dnr,
    c.dnr_first_charttime,
    c.cmo_first,
    c.cmo_last,
    c.cmo,
    c.timecmo_chart,
    sofa.sofa,
    sofa.respiration as sofa_respiration,
    sofa.coagulation as sofa_coagulation,
    sofa.liver as sofa_liver,
    sofa.cardiovascular as sofa_cardiovascular,
    sofa.cns as sofa_cns,
    sofa.renal as sofa_renal,
    sapsii.sapsii,
    sapsii.sapsii_prob,
    oasis.oasis,
    oasis.oasis_prob,
    COALESCE(f.readmission_30, 0) AS readmission_30
FROM icustay_detail i
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN icustays s ON i.icustay_id = s.icustay_id
    INNER JOIN code_status c ON i.icustay_id = c.icustay_id
    LEFT OUTER JOIN (
        SELECT d.icustay_id, 1 as readmission_30
        FROM icustays c, icustays d
        WHERE c.subject_id=d.subject_id
        AND c.icustay_id > d.icustay_id
        AND c.intime - d.outtime <= interval '30 days'
        AND c.outtime = (SELECT MIN(e.outtime) FROM icustays e
                        WHERE e.subject_id=c.subject_id
                        AND e.intime>d.outtime)
    ) f ON i.icustay_id=f.icustay_id
    LEFT OUTER JOIN (SELECT icustay_id, sofa, respiration, coagulation,
                     liver, cardiovascular, cns, renal FROM sofa) sofa
        ON i.icustay_id=sofa.icustay_id
    LEFT OUTER JOIN (SELECT icustay_id, sapsii, sapsii_prob FROM sapsii) sapsii
        ON sapsii.icustay_id=i.icustay_id
    LEFT OUTER JOIN (SELECT icustay_id, oasis, oasis_prob FROM oasis) oasis
        ON oasis.icustay_id=i.icustay_id
WHERE s.first_careunit NOT LIKE 'NICU'
    AND i.hadm_id IS NOT NULL
    AND i.icustay_id IS NOT NULL
    AND i.hospstay_seq = 1
    AND i.icustay_seq = 1
    AND i.admission_age >= {min_age}
    AND i.los_icu >= {min_day}
    AND (i.outtime >= (i.intime + interval '{min_dur} hours'))
    AND (i.outtime <= (i.intime + interval '{max_dur} hours'))
ORDER BY subject_id
{limit}
```

#### Cohort Selection Criteria (Default Values)

- **First ICU stay only**: `hospstay_seq = 1` and `icustay_seq = 1`
- **Minimum age**: 15 years (`admission_age >= 15`)
- **Minimum ICU length of stay**: 0 days (`los_icu >= 0`)
- **Minimum ICU duration**: 12 hours
- **Maximum ICU duration**: 240 hours (10 days)
- **Exclude NICU**: `first_careunit NOT LIKE 'NICU'`

#### Output

- **File**: `static_data.csv`
- **Rows**: 34,472 ICU stays
- **Columns**: ~40 demographic and outcome variables

---

### Stage 2: Vitals & Labs Extraction

#### Variable Mapping

**File**: `MIMIC_Extract/resources/itemid_to_variable_map.csv`

This file maps raw `ITEMID` values from MIMIC-III to standardized variable names:

- **Total ITEMIDs**: 13,252 rows
- **Status Filter**: Only items with `STATUS='ready'` and `COUNT>0` are extracted
- **LEVEL2 Grouping**: Multiple ITEMIDs mapped to same clinical variable

**Example Mappings**:

| LEVEL2 | LEVEL1 | ITEMID | MIMIC LABEL | LINKSTO | COUNT |
|--------|--------|--------|-------------|---------|-------|
| Heart Rate | Heart Rate | 211 | Heart Rate | chartevents | 3.4M |
| Heart Rate | Heart Rate | 220045 | Heart Rate | chartevents | 5.9M |
| Systolic blood pressure | Systolic blood pressure | 51 | Arterial BP [Systolic] | chartevents | 1.5M |
| Systolic blood pressure | Systolic blood pressure | 220050 | Arterial Blood Pressure systolic | chartevents | 4.2M |
| Temperature | Temperature | 223761 | Temperature Fahrenheit | chartevents | 2.8M |
| Temperature | Temperature | 223762 | Temperature Celsius | chartevents | 4.3M |

#### Data Sources

Vitals and labs are extracted from:

1. **CHARTEVENTS**: Bedside vital signs (MetaVision & CareVue systems)
2. **LABEVENTS**: Laboratory test results

#### Unit Standardization

**Conversions Applied** (from `mimic_direct_extract.py:178-185`):

```python
UNIT_CONVERSIONS = [
    ('weight',                   'oz',  None,             lambda x: x/16.*0.45359237),  # oz → kg
    ('weight',                   'lbs', None,             lambda x: x*0.45359237),       # lbs → kg
    ('fraction inspired oxygen', None,  lambda x: x > 1,  lambda x: x/100.),            # % → fraction
    ('oxygen saturation',        None,  lambda x: x <= 1, lambda x: x*100.),            # fraction → %
    ('temperature',              'f',   lambda x: x > 79, lambda x: (x - 32) * 5./9),  # F → C
    ('height',                   'in',  None,             lambda x: x*2.54),            # in → cm
]
```

#### Outlier Removal & Value Clamping

**File**: `MIMIC_Extract/resources/variable_ranges.csv`

Each variable has defined valid ranges:

| Variable | OUTLIER LOW | VALID LOW | IMPUTE | VALID HIGH | OUTLIER HIGH |
|----------|-------------|-----------|--------|------------|--------------|
| Heart Rate | 0 | 0 | 86 | 350 | 390 |
| Systolic blood pressure | 0 | 0 | 120 | 375 | 400 |
| Diastolic blood pressure | 0 | 0 | 59 | 375 | 375 |
| Temperature | 26 | 26 | 37 | 45 | 45 |
| Respiratory rate | 0 | 0 | 19 | 300 | 330 |
| Oxygen saturation | 0 | 0 | 98 | 100 | 150 |
| Glucose | 0 | 33 | 128 | 2000 | 2200 |
| pH | 6.3 | 6.3 | 7.4 | 8.4 | 10 |

**Processing Logic** (from `mimic_direct_extract.py:737-767`):

```python
def apply_variable_limits(df, var_ranges):
    for var_name in var_names:
        # Get ranges for this variable
        outlier_low_val, outlier_high_val = var_ranges['OUTLIER_LOW'], var_ranges['OUTLIER_HIGH']
        valid_low_val, valid_high_val = var_ranges['VALID_LOW'], var_ranges['VALID_HIGH']

        # 1. Remove outliers (set to NaN)
        df.loc[df.value < outlier_low_val, 'value'] = np.nan
        df.loc[df.value > outlier_high_val, 'value'] = np.nan

        # 2. Clamp values to valid range
        df.loc[df.value < valid_low_val, 'value'] = valid_low_val
        df.loc[df.value > valid_high_val, 'value'] = valid_high_val
```

**Example**: For Heart Rate:
- Values < 0 or > 390 → Set to NaN (removed as outliers)
- Values between 0-0 → Clamped to 0
- Values between 350-390 → Clamped to 350
- Values between 0-350 → Kept as-is

#### Hourly Aggregation

After cleaning, measurements are aggregated by hour:

```python
# Group by patient, ICU stay, hour, and variable
grouped = df.groupby(['subject_id', 'hadm_id', 'icustay_id', 'LEVEL2', 'hours_in'])

# Aggregate into 3 statistics per hour
aggregated = grouped.agg(['mean', 'std', 'count'])
```

**Result**: For each vital/lab variable per hour:
- `(Variable Name, 'mean')`: Mean value in that hour
- `(Variable Name, 'std')`: Standard deviation in that hour
- `(Variable Name, 'count')`: Number of measurements in that hour

#### Output

- **File**: `vitals_hourly_data.csv.npy` (numpy array)
- **Shape**: 2,200,954 rows × 312 columns
- **Rows**: One row per hour per ICU stay
- **Columns**: 104 variables × 3 statistics = 312 columns
- **Column Names**: `vitals_colnames.txt`

---

### Stage 3: Interventions & Outcomes Extraction

#### Intervention Duration Queries

For each intervention type, the extraction queries the corresponding duration table:

**Example: Mechanical Ventilation**

```sql
SELECT i.subject_id, i.hadm_id, v.icustay_id, v.ventnum, v.starttime, v.endtime
FROM icustay_detail i
    INNER JOIN ventilation_durations v ON i.icustay_id = v.icustay_id
WHERE v.icustay_id IN ({icuids})
    AND v.starttime BETWEEN intime AND outtime
    AND v.endtime BETWEEN intime AND outtime;
```

**Example: Vasopressor Durations**

```sql
SELECT i.subject_id, i.hadm_id, v.icustay_id, v.vasonum, v.starttime, v.endtime
FROM icustay_detail i
    INNER JOIN vasopressor_durations v ON i.icustay_id = v.icustay_id
WHERE v.icustay_id IN ({icuids})
    AND v.starttime BETWEEN intime AND outtime
    AND v.endtime BETWEEN intime AND outtime;
```

#### Hourly Binary Indicators

The duration data (start/end times) is converted to hourly binary flags:

```python
def add_outcome_indicators(out_gb):
    # Get all hours where intervention was active
    on_hrs = set()
    for index, row in out_gb.iterrows():
        on_hrs.update(range(row['starttime'], row['endtime'] + 1))

    # Create binary flags: 1 if intervention active, 0 otherwise
    max_hrs = out_gb['max_hours'].unique()[0]
    off_hrs = set(range(max_hrs + 1)) - on_hrs
    on_vals = [0]*len(off_hrs) + [1]*len(on_hrs)
```

#### Extracted Interventions

| Intervention | Source Table | Description |
|--------------|--------------|-------------|
| `vent` | `ventilation_durations` | Mechanical ventilation |
| `vaso` | `vasopressor_durations` | Any vasopressor |
| `adenosine` | `adenosine_durations` | Adenosine infusion |
| `dobutamine` | `dobutamine_durations` | Dobutamine infusion |
| `dopamine` | `dopamine_durations` | Dopamine infusion |
| `epinephrine` | `epinephrine_durations` | Epinephrine infusion |
| `isuprel` | `isuprel_durations` | Isuprel infusion |
| `milrinone` | `milrinone_durations` | Milrinone infusion |
| `norepinephrine` | `norepinephrine_durations` | Norepinephrine infusion |
| `phenylephrine` | `phenylephrine_durations` | Phenylephrine infusion |
| `vasopressin` | `vasopressin_durations` | Vasopressin infusion |
| `colloid_bolus` | `colloid_bolus` | Colloid bolus administration |
| `crystalloid_bolus` | `crystalloid_bolus` | Crystalloid bolus administration |
| `nivdurations` | `nivdurations` | Non-invasive ventilation |

#### Output

- **File**: `outcomes_hourly_data.csv`
- **Shape**: 2,200,954 rows × 15 columns
- **Columns**: `subject_id`, `hadm_id`, `icustay_id`, `hours_in`, + 11 intervention binary flags
- **Column Names**: `outcomes_colnames.txt`

---

### Stage 4: ICD-9 Diagnosis Codes (Optional)

#### SQL Query: Diagnosis Codes

**File**: `MIMIC_Extract/SQL_Queries/codes.sql`

```sql
SET SEARCH_PATH TO public,mimiciii;
SELECT
    i.icustay_id, d.subject_id, d.hadm_id,
    array_agg(d.icd9_code ORDER BY seq_num ASC) AS icd9_codes
FROM diagnoses_icd d
    LEFT OUTER JOIN (SELECT ccs_matched_id, icd9_code FROM ccs_dx) c
        ON c.icd9_code = d.icd9_code
    INNER JOIN icustays i
        ON i.hadm_id = d.hadm_id AND i.subject_id = d.subject_id
WHERE d.hadm_id IN ('{hadm_id}') AND seq_num IS NOT NULL
GROUP BY i.icustay_id, d.subject_id, d.hadm_id
```

---

## Post-Extraction Cleaning

### Stage 5: Flat Table Creation

**Script**: `data/extract/create_flat_table.py`

This script combines all extracted data into a single denormalized table:

```python
# 1. Load all data sources
static_df = pd.read_csv('static_data.csv')              # 34,472 × 40
outcomes_df = pd.read_csv('outcomes_hourly_data.csv')   # 2,200,954 × 15
vitals_array = np.load('vitals_hourly_data.csv.npy')    # 2,200,954 × 312
subjects = np.load('subjects.npy')                       # 34,472
fenceposts = np.load('fenceposts.npy')                   # 34,472

# 2. Reconstruct vitals DataFrame with proper index
vitals_df = pd.DataFrame(vitals_array, columns=vitals_colnames)
# Add subject_id, hadm_id, icustay_id, hours_in from subjects/fenceposts

# 3. Merge outcomes with vitals
merged_df = outcomes_df.merge(vitals_df,
    on=['subject_id', 'hadm_id', 'icustay_id', 'hours_in'],
    how='outer')

# 4. Merge with static data (denormalize - static repeated for each hour)
final_df = merged_df.merge(static_df,
    on=['subject_id', 'hadm_id', 'icustay_id'],
    how='left')

# 5. Save to CSV
final_df.to_csv('flat_table.csv', index=False)
```

#### Output: `flat_table.csv`

- **Rows**: 2,200,954 (one row per hour per ICU stay)
- **Columns**: 358 total
  - 4 ID columns: `subject_id`, `hadm_id`, `icustay_id`, `hours_in`
  - 14 outcome columns: Binary intervention flags
  - 312 vitals/labs columns: 104 variables × 3 statistics
  - 28 static columns: Demographics, diagnoses, severity scores
- **Size**: 2.0 GB

---

### Stage 6: Missing Data Cleanup

**Script**: `data/extract/clean_flat_table.py`

Many columns in the extracted data are 100% missing because:
1. The variable exists in the mapping but was never measured in this cohort
2. Specialized tests (pleural fluid, CSF, ascites) are rarely performed
3. Rare cardiac measurements not available for most patients

#### Process

```python
# 1. Read sample to identify 100% missing columns
df_sample = pd.read_csv('flat_table.csv', nrows=50000)
missing = df_sample.isnull().sum()
completely_missing = missing[missing == len(df_sample)].index.tolist()
# Result: 199 columns with 100% missing

# 2. Process full file in chunks, keeping only non-missing columns
cols_to_keep = [c for c in df_sample.columns if c not in completely_missing]
chunk_size = 100000
for chunk in pd.read_csv('flat_table.csv', chunksize=chunk_size, usecols=cols_to_keep):
    chunk.to_csv('flat_table_cleaned.csv', mode='a', header=first_chunk)
```

#### Categories of Dropped Variables

1. **Body Fluid Tests** (~80 columns)
   - Pleural fluid analysis
   - Ascites fluid analysis
   - CSF (cerebrospinal fluid) analysis
   - Other body fluid tests

2. **Specialized Cell Counts** (~40 columns)
   - Atypical cells
   - Percent calculations
   - Rare blood cell subtypes

3. **Rare Cardiac Measurements** (~30 columns)
   - Cardiac output (Thermodilution, Fick)
   - Pulmonary artery pressure
   - Pulmonary capillary wedge pressure
   - Systemic vascular resistance

4. **Other Rare Labs** (~49 columns)
   - Specialized metabolic tests
   - Rare coagulation tests
   - Uncommon chemistry panels

#### Output: `flat_table_cleaned.csv`

- **Rows**: 2,200,954 (unchanged)
- **Columns**: 159 (dropped 199 columns with 100% missing)
- **Size**: 2.0 GB

---

## Final Data Structure

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Patients** | 34,472 unique subjects |
| **Total ICU Stays** | 34,472 (first ICU stay only) |
| **Total Hourly Records** | 2,200,954 |
| **Average Hours per Stay** | ~63.8 hours |
| **Time Period** | First 12-240 hours of ICU stay |
| **File Size** | 2.0 GB |
| **Format** | CSV with mixed data types |

### Column Structure (159 Columns Total)

#### 1. Identifier Columns (4)

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | int | Unique patient identifier |
| `hadm_id` | int | Hospital admission identifier |
| `icustay_id` | int | ICU stay identifier |
| `hours_in` | int | Hours since ICU admission (0, 1, 2, ...) |

#### 2. Outcome/Intervention Columns (11)

Binary flags (0/1) indicating active interventions:

| Column | Description |
|--------|-------------|
| `dobutamine` | Dobutamine infusion active |
| `dopamine` | Dopamine infusion active |
| `epinephrine` | Epinephrine infusion active |
| `isuprel` | Isuprel infusion active |
| `milrinone` | Milrinone infusion active |
| `norepinephrine` | Norepinephrine infusion active |
| `phenylephrine` | Phenylephrine infusion active |
| `vasopressin` | Vasopressin infusion active |
| `colloid_bolus` | Colloid bolus given |
| `crystalloid_bolus` | Crystalloid bolus given |
| `nivdurations` | Non-invasive ventilation active |

**Note**: `vent` (mechanical ventilation) and `vaso` (any vasopressor) columns were in outcomes file but may not be in cleaned version.

#### 3. Vitals/Labs Columns (117 variables × 3 = ~108 kept after cleaning)

Each variable has 3 columns: `(Variable, 'count')`, `(Variable, 'mean')`, `(Variable, 'std')`

**Common Vitals** (High Data Availability):

| Variable | Unit | Typical Range |
|----------|------|---------------|
| Heart Rate | bpm | 0-350 |
| Systolic blood pressure | mmHg | 0-375 |
| Diastolic blood pressure | mmHg | 0-375 |
| Mean blood pressure | mmHg | 14-330 |
| Respiratory rate | breaths/min | 0-300 |
| Temperature | °C | 26-45 |
| Oxygen saturation | % | 0-100 |
| Fraction inspired oxygen | fraction | 0.21-1.0 |

**Common Labs** (Moderate Data Availability):

| Variable | Unit | Description |
|----------|------|-------------|
| Glucose | mg/dL | Blood glucose |
| Creatinine | mg/dL | Kidney function |
| Blood urea nitrogen | mg/dL | Kidney function |
| Bicarbonate | mEq/L | Acid-base balance |
| Chloride | mEq/L | Electrolyte |
| Sodium | mEq/L | Electrolyte |
| Potassium | mEq/L | Electrolyte |
| Hemoglobin | g/dL | Oxygen-carrying capacity |
| Hematocrit | % | Blood cell concentration |
| White blood cell count | K/µL | Immune function |
| Platelets | K/µL | Clotting function |
| pH | pH units | Acid-base balance |
| Partial pressure of oxygen | mmHg | Oxygenation |
| Partial pressure of carbon dioxide | mmHg | Ventilation |
| Lactate | mmol/L | Tissue perfusion |

**Other Vitals** (Lower Data Availability):

- Glasgow Coma Scale (eye opening, motor response, verbal response, total)
- Peak inspiratory pressure
- Central venous pressure
- Height, Weight
- Capillary refill rate
- Various other labs and chemistry panels

#### 4. Static/Demographic Columns (28)

Repeated for each hour of the same ICU stay:

| Column | Type | Description |
|--------|------|-------------|
| `gender` | string | Patient gender (M/F) |
| `age` | float | Age at admission |
| `ethnicity` | string | Patient ethnicity |
| `admission_type` | string | EMERGENCY/ELECTIVE/URGENT |
| `insurance` | string | Insurance type |
| `admittime` | datetime | Hospital admission time |
| `dischtime` | datetime | Hospital discharge time |
| `intime` | datetime | ICU admission time |
| `outtime` | datetime | ICU discharge time |
| `diagnosis_at_admission` | string | Primary diagnosis |
| `discharge_location` | string | Where patient discharged |
| `first_careunit` | string | First ICU type (MICU/SICU/CCU/etc.) |
| `los_icu` | float | ICU length of stay (days) |
| `hospstay_seq` | int | Hospital stay sequence number |
| `hospital_expire_flag` | int | Died in hospital (0/1) |
| `mort_icu` | int | Died in ICU (0/1) |
| `mort_hosp` | int | Died in hospital (0/1) |
| `readmission_30` | int | 30-day readmission (0/1) |
| `fullcode_first` | int | Full code initially (0/1) |
| `dnr_first` | int | DNR initially (0/1) |
| `fullcode` | int | Full code ever (0/1) |
| `dnr` | int | DNR ever (0/1) |
| `sofa` | float | SOFA score (0-24) |
| `sofa_respiration` | float | SOFA respiratory subscore |
| `sofa_coagulation` | float | SOFA coagulation subscore |
| `sofa_liver` | float | SOFA liver subscore |
| `sofa_cardiovascular` | float | SOFA cardiovascular subscore |
| `sofa_cns` | float | SOFA CNS subscore |
| `sofa_renal` | float | SOFA renal subscore |
| `sapsii` | float | SAPS II score |
| `sapsii_prob` | float | SAPS II mortality probability |
| `oasis` | float | OASIS score |
| `oasis_prob` | float | OASIS mortality probability |

---

## Feature Dictionary

### Temporal Structure

- **One row per hour** per ICU stay
- `hours_in` ranges from 0 (admission) to maximum stay length
- Each ICU stay has multiple consecutive hourly records
- Static features are **denormalized** (repeated for each hour)

### Missing Data Patterns

| Data Type | Expected Missingness | Reason |
|-----------|---------------------|---------|
| Vitals (HR, BP, RR, SpO2) | 5-30% | Continuous monitoring, but gaps occur |
| Temperature | 30-50% | Measured periodically, not continuously |
| Common Labs (glucose, creatinine) | 60-80% | Drawn 1-4 times per day |
| Specialized Labs | 80-95% | Ordered only when clinically indicated |
| Glasgow Coma Scale | 40-70% | Assessed periodically |
| Interventions | Sparse (1-20% active) | Only during active treatment |

### Data Quality Notes

1. **Outliers Removed**: Values outside OUTLIER range set to NaN
2. **Values Clamped**: Values outside VALID range clamped to boundary
3. **Units Standardized**: All measurements converted to standard units
4. **Hourly Aggregation**: Multiple measurements per hour aggregated to mean/std/count
5. **Missing Hours Filled**: Hours with no measurements have NaN values
6. **100% Missing Dropped**: Variables never measured in cohort removed

---

## SQL Queries Reference

### Key Concept Table Definitions

#### icustay_detail.sql

Provides comprehensive demographic and administrative information per ICU stay.

**Key Fields**:
- Patient demographics (age, gender, ethnicity)
- Hospital admission details (admittime, dischtime, diagnosis)
- ICU stay details (intime, outtime, los_icu, first_careunit)
- Stay sequences (hospstay_seq, icustay_seq)
- Mortality flags (hospital_expire_flag)

**Source**: `mimic-code/mimic-iii/concepts/demographics/icustay_detail.sql`

#### Severity Scores

**SOFA (Sepsis-related Organ Failure Assessment)**
- Score range: 0-24
- Components: Respiration, Coagulation, Liver, Cardiovascular, CNS, Renal
- Each component: 0-4 points
- **Source**: `mimic-code/mimic-iii/concepts/severityscores/sofa.sql`

**SAPS II (Simplified Acute Physiology Score II)**
- Score range: 0-163
- Includes: Age, vital signs, labs, GCS, chronic diseases
- `sapsii_prob`: Predicted mortality probability
- **Source**: `mimic-code/mimic-iii/concepts/severityscores/sapsii.sql`

**OASIS (Oxford Acute Severity of Illness Score)**
- Score range: 0-64
- Simplified severity score using 10 variables
- `oasis_prob`: Predicted mortality probability
- **Source**: `mimic-code/mimic-iii/concepts/severityscores/oasis.sql`

#### Duration Tables

All duration tables follow similar structure:

```sql
SELECT
    icustay_id,
    ROW_NUMBER() OVER (PARTITION BY icustay_id ORDER BY starttime) as num,
    starttime,
    endtime,
    EXTRACT(EPOCH FROM (endtime - starttime))/60/60 AS duration_hours
FROM (
    -- Complex logic to identify start/stop events
    -- Handles gaps, overlaps, and sequential events
)
```

**Key Duration Concepts**:
- **Mechanical Ventilation**: Any form of invasive ventilation
- **Vasopressors**: Medications to increase blood pressure
- **Fluid Boluses**: Rapid IV fluid administration

**Sources**: `mimic-code/mimic-iii/concepts/durations/*.sql`

---

## Usage Examples

### Loading the Data

```python
import pandas as pd
import numpy as np

# Load full dataset
df = pd.read_csv('flat_table_cleaned.csv')

# Or load in chunks for memory efficiency
chunk_size = 100000
chunks = []
for chunk in pd.read_csv('flat_table_cleaned.csv', chunksize=chunk_size):
    # Process chunk
    chunks.append(chunk)
df = pd.concat(chunks)
```

### Common Analysis Patterns

#### 1. Filter to First 24 Hours

```python
df_24h = df[df['hours_in'] < 24]
```

#### 2. Get Specific Patient Timeline

```python
patient = df[df['subject_id'] == 12345].sort_values('hours_in')
```

#### 3. Extract Vital Signs Time Series

```python
# Get mean respiratory rate over time
resp_rate = df[["subject_id", "icustay_id", "hours_in",
                "('Respiratory rate', 'mean')"]].dropna()
```

#### 4. Identify Patients on Interventions

```python
# Patients who received norepinephrine
norepi_patients = df[df['norepinephrine'] == 1]['subject_id'].unique()

# Hours of mechanical ventilation (if vent column exists)
# vent_hours = df.groupby('icustay_id')['vent'].sum()
```

#### 5. Aggregate to Patient Level

```python
# Get max vital signs per patient
patient_max = df.groupby('subject_id').agg({
    "('Heart Rate', 'mean')": 'max',
    "('Systolic blood pressure', 'mean')": 'max',
    'mort_icu': 'first',  # static, take first
    'age': 'first'
})
```

### Handling MultiIndex Columns

The vitals/labs columns are stored as strings like `"('Variable', 'mean')"`:

```python
# Method 1: Use string directly
heart_rate = df["('Heart Rate', 'mean')"]

# Method 2: Convert to proper MultiIndex
import ast
new_cols = []
for col in df.columns:
    if col.startswith('('):
        new_cols.append(ast.literal_eval(col))
    else:
        new_cols.append(col)
df.columns = pd.Index(new_cols)

# Now can access like:
heart_rate = df[('Heart Rate', 'mean')]
```

### Imputation Strategies

```python
# Forward fill within each ICU stay (carry forward last observation)
df = df.sort_values(['icustay_id', 'hours_in'])
vital_cols = [c for c in df.columns if 'mean' in str(c)]
df[vital_cols] = df.groupby('icustay_id')[vital_cols].ffill()

# Or use median imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[vital_cols] = imputer.fit_transform(df[vital_cols])
```

---

## Data Limitations & Considerations

### 1. Selection Bias

- **First ICU stay only**: Excludes repeat admissions
- **Minimum duration filters**: Excludes very short/long stays
- **Age filter**: Excludes pediatric patients
- **No NICU**: Neonatal patients excluded

### 2. Missing Data

- **Not Missing at Random**: Labs ordered based on clinical suspicion
- **Intermittent Monitoring**: Vitals gaps don't mean patient deteriorated
- **Specialized Tests Rare**: Many variables have >90% missing

### 3. Data Quality Issues

- **CareVue vs MetaVision**: Two different ICU systems with different itemids
- **Unit Inconsistencies**: Addressed by conversion, but some may remain
- **Temporal Granularity**: Hourly aggregation loses within-hour variability
- **Outlier Definitions**: Clinical ranges may not apply to all patients

### 4. Temporal Considerations

- **Right Censoring**: ICU stays truncated at discharge/death
- **Irregular Sampling**: Labs drawn at clinical discretion
- **Intervention Timing**: Start/stop times may be approximate

### 5. Privacy & Ethics

- **De-identified**: All dates shifted, ages >89 set to 300
- **HIPAA Compliant**: Safe Harbor method used
- **Research Use Only**: Not for clinical decision making

---

## References

### Papers

1. **MIMIC-III Database**:
   Johnson, A., Pollard, T., Shen, L. et al. MIMIC-III, a freely accessible critical care database. *Sci Data* 3, 160035 (2016).
   https://doi.org/10.1038/sdata.2016.35

2. **MIMIC-Extract**:
   Wang, S., McDermott, M., Chauhan, G., Ghassemi, M., Hughes, M., & Naumann, T. (2020). MIMIC-Extract: A Data Extraction, Preprocessing, and Representation Pipeline for MIMIC-III. *CHIL 2020*.
   https://arxiv.org/abs/1907.08322

### Code Repositories

- **MIMIC-III Database**: https://physionet.org/content/mimiciii/1.4/
- **mimic-code** (Concepts): https://github.com/MIT-LCP/mimic-code
- **MIMIC_Extract**: https://github.com/MLforHealth/MIMIC_Extract
- **MIMICExtractEasy** (This Project): https://github.com/SphtKr/MIMICExtractEasy

### Documentation

- **MIMIC-III Documentation**: https://mimic.mit.edu/docs/iii/
- **mimic-code Concepts**: https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/concepts

---

## Summary

This dataset represents a highly processed, analysis-ready version of MIMIC-III ICU data:

1. **From 7GB, 26 raw tables** → **2GB, single flat table**
2. **From 330M+ chart events** → **2.2M hourly aggregated records**
3. **From 12K+ item IDs** → **104 standardized variables**
4. **From irregular timestamps** → **Aligned hourly time series**
5. **From diverse units** → **Standardized measurements**
6. **From extreme outliers** → **Validated ranges**

**Key Strengths**:
- Ready for time-series machine learning
- Standardized variable names and units
- Cleaned and validated measurements
- Rich clinical context (outcomes, severity scores, interventions)
- Reproducible pipeline with clear documentation

**Best Use Cases**:
- Mortality prediction models
- Intervention onset/weaning prediction
- Physiological time-series analysis
- Comparative effectiveness studies
- ICU resource utilization studies

**Citation**: If you use this data, please cite both the MIMIC-III database and MIMIC-Extract papers listed above.

---

**Document Version**: 1.0
**Last Updated**: December 8, 2025
**Generated from**: MIMICExtractEasy pipeline
