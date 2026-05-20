"""
preprocessing/clean.py
======================
Expanded feature pipeline for the Patient Digital Twin.

New features vs v1:
  • discharge_disposition_id  → 4-level risk bucket (0=Home, 1=Supervised, 2=Facility, 3=AMA)
  • admission_type_id         → 3-level (0=Elective, 1=Urgent, 2=Emergency)
  • admission_source_id       → 3-level (0=Referral, 1=Transfer, 2=ER)
  • num_procedures            → surgical procedure count (was dropped before)
  • number_diagnoses          → total diagnoses count (comorbidity proxy)
  • change                    → medication regimen changed this visit (0/1)
  • max_glu_serum             → glucose serum result (0=not tested → 3=severe)
  • medical_specialty         → top-12 admitting specialties + "Other" (OHE)
  • 23 individual drug cols   → encoded 0=No, 1=Steady, 2=Changed (Up/Down)
  • drug_changes_count        → # of drugs with dose Up or Down (new aggregate)
  • drugs_active_count        → # of drugs patient is currently on (new aggregate)

Rows excluded (cannot be readmitted within 30 days):
  • Expired patients (discharge_disposition_id in {11, 19, 20, 21})
  • Hospice patients (13, 14)
  • Still admitted (9, 12)
"""

import pandas as pd
import numpy as np
import os
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Columns that MUST be present in the raw CSV
REQUIRED_COLUMNS = {
    # Original core
    'race', 'gender', 'age',
    'time_in_hospital', 'num_lab_procedures', 'num_medications',
    'number_outpatient', 'number_inpatient', 'number_emergency',
    'A1Cresult', 'insulin', 'diabetesMed',
    'diag_1', 'diag_2', 'diag_3', 'readmitted',
    # New additions
    'discharge_disposition_id', 'admission_type_id', 'admission_source_id',
    'num_procedures', 'number_diagnoses', 'change', 'max_glu_serum',
    'medical_specialty',
    # 23 individual drug columns
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
    'tolazamide', 'examide', 'citoglipton',
    'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
}

DRUG_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
    'tolazamide', 'examide', 'citoglipton',
    'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone',
]

# discharge_disposition_id rows to EXCLUDE entirely
# (patient cannot be readmitted — they died, are in hospice, or never left)
EXCLUDE_DISPOSITIONS = {
    9,   # Admitted as inpatient — never actually discharged
    11,  # Expired
    12,  # Still patient / expected return for outpatient only
    13,  # Hospice / home
    14,  # Hospice / medical facility
    19,  # Expired at home (Medicaid hospice)
    20,  # Expired in medical facility (Medicaid hospice)
    21,  # Expired, place unknown
}

# Top medical specialties (covers ~70% of data); rest → "Other"
TOP_SPECIALTIES = [
    'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
    'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics',
    'Orthopedics-Reconstructive', 'Radiologist', 'Pulmonology',
    'Psychiatry', 'ObstetricsandGynecology',
]


# ──────────────────────────────────────────────────────────────────────────────
# Schema validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_schema(df: pd.DataFrame) -> None:
    """Fail loudly if any expected column is missing from the raw CSV."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"[Schema validation failed] Missing columns:\n  {sorted(missing)}\n"
            f"Available: {sorted(df.columns.tolist())}"
        )
    print(f"  Schema OK — all {len(REQUIRED_COLUMNS)} required columns present.")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def clean_data():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir    = os.path.join(project_dir, 'data')
    input_path  = os.path.join(data_dir, 'diabetic_data.csv')
    output_path = os.path.join(data_dir, 'clean_data.csv')

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Raw shape: {df.shape}")

    # ── 1. Schema check ───────────────────────────────────────────────────────
    print("Validating schema...")
    validate_schema(df)

    # ── 2. Replace missing markers ────────────────────────────────────────────
    df.replace('?', np.nan, inplace=True)
    df.replace('Unknown/Invalid', np.nan, inplace=True)

    # ── 3. Exclude rows where readmission is not meaningful ───────────────────
    df['discharge_disposition_id'] = pd.to_numeric(
        df['discharge_disposition_id'], errors='coerce'
    )
    n_before = len(df)
    df = df[~df['discharge_disposition_id'].isin(EXCLUDE_DISPOSITIONS)].copy()
    print(f"  Excluded {n_before - len(df):,} rows (expired/hospice/still-admitted). "
          f"Remaining: {len(df):,}")

    # ── 4. Target engineering ─────────────────────────────────────────────────
    df['target'] = (df['readmitted'] == '<30').astype(int)
    pos_rate = df['target'].mean()
    print(f"  Positive class (<30-day readmission): {pos_rate:.2%}")

    # ── 5. Select & rename columns ────────────────────────────────────────────
    keep = [
        # Demographics
        'race', 'gender', 'age',
        # Hospital stay
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_diagnoses',
        # Visit history
        'number_outpatient', 'number_inpatient', 'number_emergency',
        # Lab results
        'A1Cresult', 'max_glu_serum',
        # Drug-level
        'insulin', 'diabetesMed', 'change',
        # Diagnoses
        'diag_1', 'diag_2', 'diag_3',
        # New categorical IDs
        'discharge_disposition_id', 'admission_type_id',
        'admission_source_id', 'medical_specialty',
        # 23 individual drug cols
        *DRUG_COLS,
        # Target
        'target',
    ]
    df = df[keep].copy()
    df.rename(columns={'A1Cresult': 'HbA1c_result'}, inplace=True)

    # ── 6. Impute missing values ──────────────────────────────────────────────
    print("Imputing missing values...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in num_cols:
        if col != 'target':
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        df[col].fillna('Unknown', inplace=True)

    # ── 7. Encode clinical IDs → risk buckets ─────────────────────────────────
    print("Encoding clinical IDs...")

    # discharge_disposition_id → 4-level risk (0=Home, 1=Supervised, 2=Facility, 3=AMA)
    def map_discharge(x):
        if x == 1:   return 0   # Discharged to home
        if x in {6, 8}:  return 1   # Home with care / Home IV
        if x in {2, 3, 4, 5, 15, 22, 23, 24, 27, 28, 29, 30}: return 2  # Facility / Transfer
        if x == 7:   return 3   # Left AMA — highest readmission risk
        return 1                # Default: supervised

    df['discharge_risk'] = df['discharge_disposition_id'].apply(map_discharge)
    df.drop(columns=['discharge_disposition_id'], inplace=True)

    # admission_type_id → 0=Elective, 1=Urgent, 2=Emergency
    def map_admission_type(x):
        try: x = int(x)
        except: return 1
        if x == 1: return 2   # Emergency
        if x == 2: return 1   # Urgent
        if x == 3: return 0   # Elective
        return 1              # Newborn / Unknown

    df['admission_type'] = df['admission_type_id'].apply(map_admission_type)
    df.drop(columns=['admission_type_id'], inplace=True)

    # admission_source_id → 0=Referral, 1=Transfer, 2=ER
    def map_admission_source(x):
        try: x = int(x)
        except: return 0
        if x == 7: return 2   # Emergency Room
        if x in {4, 5, 6, 10, 18, 22, 25, 26}: return 1  # Transfer
        return 0              # Referral / Other

    df['admission_source'] = df['admission_source_id'].apply(map_admission_source)
    df.drop(columns=['admission_source_id'], inplace=True)

    # ── 8. Binary / ordinal mappings ──────────────────────────────────────────
    print("Encoding binary/ordinal features...")

    insulin_map  = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 2}
    hba1c_map    = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}   # now 4-level, not binary
    glu_map      = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
    change_map   = {'No': 0, 'Ch': 1}
    drug_map     = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 2}

    df['insulin']      = df['insulin'].map(insulin_map).fillna(0).astype(int)
    df['HbA1c_result'] = df['HbA1c_result'].map(hba1c_map).fillna(0).astype(int)
    df['max_glu_serum']= df['max_glu_serum'].map(glu_map).fillna(0).astype(int)
    df['change']       = df['change'].map(change_map).fillna(0).astype(int)
    df['diabetesMed']  = df['diabetesMed'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)

    # 23 individual drug columns → 0/1/2
    print(f"  Encoding {len(DRUG_COLS)} individual drug columns...")
    for col in DRUG_COLS:
        df[col] = df[col].map(drug_map).fillna(0).astype(int)

    # Aggregate drug features
    drug_df = df[DRUG_COLS]
    df['drug_changes_count'] = (drug_df == 2).sum(axis=1)   # # drugs with dose Up/Down
    df['drugs_active_count'] = (drug_df >= 1).sum(axis=1)   # # drugs patient is on

    # ── 9. Simplify ICD diagnosis codes ───────────────────────────────────────
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].astype(str).str[0]
        df[col] = df[col].replace({'n': 'Unknown', 'U': 'Unknown'})

    # ── 10. Medical specialty → top-12 + Other ────────────────────────────────
    print("  Encoding medical_specialty...")
    df['medical_specialty'] = df['medical_specialty'].apply(
        lambda x: x if x in TOP_SPECIALTIES else 'Other'
    )

    # ── 11. One-hot encoding ──────────────────────────────────────────────────
    print("One-hot encoding categorical columns...")
    ohe_cols = ['race', 'gender', 'age', 'diag_1', 'diag_2', 'diag_3', 'medical_specialty']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Ensure no bool columns (XGBoost requires int/float)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    print(f"\nFinal feature count: {df.shape[1] - 1}  (excl. target)")
    print(f"Final row count:     {len(df):,}")
    print(f"Target positive rate: {df['target'].mean():.2%}")
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Preprocessing complete!")

    # ── 12. Print feature summary ─────────────────────────────────────────────
    print("\n── Feature groups ──────────────────────────────────────────────")
    groups = {
        'Demographics':         [c for c in df.columns if any(c.startswith(p) for p in ['race_', 'gender_', 'age_'])],
        'Hospital stay':        ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'number_diagnoses'],
        'Visit history':        ['number_outpatient', 'number_inpatient', 'number_emergency'],
        'Lab results':          ['HbA1c_result', 'max_glu_serum'],
        'Clinical IDs':         ['discharge_risk', 'admission_type', 'admission_source'],
        'Drug summary':         ['insulin', 'diabetesMed', 'change', 'drug_changes_count', 'drugs_active_count'],
        'Individual drugs':     DRUG_COLS,
        'Diagnoses (OHE)':      [c for c in df.columns if any(c.startswith(p) for p in ['diag_1_', 'diag_2_', 'diag_3_'])],
        'Medical specialty':    [c for c in df.columns if c.startswith('medical_specialty_')],
    }
    for grp, cols in groups.items():
        present = [c for c in cols if c in df.columns]
        print(f"  {grp:22s}: {len(present):3d} features")


if __name__ == '__main__':
    clean_data()
