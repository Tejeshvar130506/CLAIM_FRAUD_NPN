# Preprocessing, Data Integration, and Feature Engineering Report

## Executive Summary

This report documents Phase 2 of the **Healthcare Claims Fraud Detection System**: the complete data preprocessing, claims integration, and provider-level feature engineering pipeline.

The target prediction is strictly at the **PROVIDER** level (`PotentialFraud`), scoring individual healthcare providers for potential fraudulent behavior based on Medicare inpatient claims, outpatient claims, and beneficiary demographics.

---

## 1. Cleaning Operations & Date Standardization

### A. Raw Data Preservation
- All 8 original Kaggle CSV files in `data/raw/train/` and `data/raw/test/` were preserved **unaltered**.

### B. Temporal Transformations & Derived Date Features
1. **Date Parsing**:
   - `DOB`, `DOD`, `ClaimStartDt`, `ClaimEndDt`, `AdmissionDt`, `DischargeDt` converted from ISO strings to datetime objects.
2. **Claim Duration**:
   - Calculated as `(ClaimEndDt - ClaimStartDt) + 1 day`. Min duration clamped to 1 day for any edge cases.
3. **Inpatient Stay Duration**:
   - Calculated as `(DischargeDt - AdmissionDt) + 1 day` for inpatient claims. Set to `NaN` for outpatient claims.
4. **Patient Age at Claim Start**:
   - Derived exact patient age in years at `ClaimStartDt`.
5. **IsDeceased Flag**:
   - Set to `1` if beneficiary `DOD` is non-null and `DOD <= ClaimStartDt`, else `0`.

### C. Clinical & Demographic Standardization
- `RenalDiseaseIndicator`: Mapped `'0'` $\rightarrow 0$, `'Y'` $\rightarrow 1$.
- **Chronic Conditions**: Converted raw Kaggle values `1` (Yes) and `2` (No) to standard binary flags `1` (Yes) and `0` (No).
- `ChronicCond_Count`: Calculated sum of 11 chronic condition indicators per beneficiary.

---

## 2. Joins Performed & Row Multiplication Validation

### A. Inpatient and Outpatient Unification
- Inpatient claims (`IsInpatient = 1`) and Outpatient claims (`IsInpatient = 0`) were aligned and concatenated into unified claims datasets:
  - **TRAIN Unified Claims**: `558,211` total claim records (`40,474` Inpatient, `517,737` Outpatient).
  - **TEST Unified Claims**: `135,392` total claim records (`9,551` Inpatient, `125,841` Outpatient).

### B. Beneficiary Reference Join
- Unified claim records were left-joined with cleaned Beneficiary data on primary key `BeneID`.
- **Join Integrity Verification**:
  - `BeneID` is strictly unique (1 row per patient) in Beneficiary reference files.
  - **Result**: `100%` row count preservation.
    - TRAIN: `558,211` input claim rows $\rightarrow$ `558,211` output merged rows (0 row expansion).
    - TEST: `135,392` input claim rows $\rightarrow$ `135,392` output merged rows (0 row expansion).

---

## 3. Missing Value & Duplicate Strategy

### A. Duplicate Handling
- Exact primary key deduplication checked for `BeneID` in Beneficiary files and `ClaimID` in Claims files.
- Zero duplicate key collisions detected.

### B. Imputation Strategy
- **Financial Features**: `IPAnnualReimbursementAmt`, `OPAnnualReimbursementAmt`, `InscClaimAmtReimbursed`, `DeductibleAmtPaid` missing values filled with `0.0`.
- **Diagnosis & Procedure Codes**: Null values preserved as missing indicators, used to compute non-null code density per claim (`NumDiagnosisCodes`, `NumProcedureCodes`).
- **Physician ID Fields**: Preserved for unique count aggregations (`UniqueAttendingPhysicians`, `UniqueOperatingPhysicians`, `UniqueOtherPhysicians`).

---

## 4. Provider-Level Feature Aggregation Strategy

The unified claim records were aggregated to the **Provider** level (`Provider`) across 6 feature categories (43 total features):

| Feature Category | Aggregations & Metrics | Rationale for Fraud Detection |
| :--- | :--- | :--- |
| **Claim Volume & Subgroups** | Total claims, Inpatient/Outpatient claim counts, Inpatient ratio, Unique beneficiaries, Claims per beneficiary. | Unusually high claim velocity or abnormal inpatient/outpatient split relative to beneficiary volume indicates billing inflation. |
| **Financial Metrics** | Total, mean, max, std of claim reimbursement amounts and deductible paid amounts; reimbursement-to-deductible ratio. | Excessive reimbursement requests, uncharacteristically high max claim values, or distorted deductible ratios indicate financial anomaly. |
| **Stay & Duration** | Mean/max claim duration, mean/max inpatient stay length. | Prolonged stay billing for routine procedures or padded claim date windows. |
| **Physician Networks** | Unique attending, operating, and other physicians; physician-to-claim ratio; same attending & operating physician ratio. | Self-referral loops, single-physician monopolies, or fictitious operating physician assignment. |
| **Clinical Code Diversity** | Unique admit diagnosis codes, unique diagnosis group codes, total unique diagnosis codes (across 10 slots), total unique procedure codes (across 6 slots). | Upcoding (coding high-reimbursement complex diagnoses repeatedly) or diagnostic code padding. |
| **Patient Risk Profile** | Mean/std patient age, deceased patient claim count/ratio, mean patient chronic condition count, renal disease ratio, state/county coverage. | Billing for deceased beneficiaries or targeting highly vulnerable/elderly chronic disease cohorts. |

---

## 5. Target Leakage Prevention & Train/Test Isolation

> [!IMPORTANT]
> ### Strict Partitioning Rules Applied
> 1. **Target Isolation**:
>    - Fraud label `PotentialFraud` (`Yes` $\rightarrow 1$, `No` $\rightarrow 0$) was attached **exclusively** to `train_provider_features.parquet`.
>    - `test_provider_features.parquet` was constructed **without** target label awareness.
> 2. **No Data Snooping**:
>    - Feature transformations (date derivations, aggregations, missing value imputations) were calculated independently per group without fitting transformers across combined TRAIN + TEST sets.
> 3. **Unlabeled Test Evaluation**:
>    - Kaggle TEST data will be used strictly for final inference after model training and cross-validation on TRAIN.

---

## 6. Output Files & Parquet Performance Metrics

| Output Dataset | Directory | File Format | Row Count | Column Count | File Size (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train_claims_integrated.parquet` | `data/processed/` | Parquet | `558,211` | 57 | 18.37 MB |
| `test_claims_integrated.parquet` | `data/processed/` | Parquet | `135,392` | 57 | 5.07 MB |
| `train_provider_features.parquet` | `data/features/` | Parquet | `5,410` | 44 | 0.74 MB |
| `test_provider_features.parquet` | `data/features/` | Parquet | `1,353` | 43 | 0.22 MB |
