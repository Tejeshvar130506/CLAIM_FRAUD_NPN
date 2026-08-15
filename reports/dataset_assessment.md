# Kaggle Healthcare Provider Fraud Detection - Dataset Assessment Report

## Executive Summary

This report documents the automated data discovery and structural profiling conducted on the Kaggle Healthcare Provider Fraud Detection Analysis dataset. The discovery phase analyzed all 8 CSV files split across **TRAIN** and **TEST** groups to establish data integrity, relational cardinalities, key relationships, class imbalance, and potential feature engineering hazards.

---

## 1. Discovered Dataset Files & Metrics

| Filename | Group | Size (MB) | Row Count | Column Count | Duplicate Rows |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Train-1542865627584.csv` | TRAIN | 0.08 MB | 5,410 | 2 | 0 |
| `Train_Beneficiarydata-1542865627584.csv` | TRAIN | 10.91 MB | 138,556 | 25 | 0 |
| `Train_Inpatientdata-1542865627584.csv` | TRAIN | 8.18 MB | 40,474 | 30 | 0 |
| `Train_Outpatientdata-1542865627584.csv` | TRAIN | 73.81 MB | 517,737 | 27 | 0 |
| `Test-1542969243754.csv` | TEST | 0.01 MB | 1,353 | 1 | 0 |
| `Test_Beneficiarydata-1542969243754.csv` | TEST | 5.08 MB | 63,968 | 25 | 0 |
| `Test_Inpatientdata-1542969243754.csv` | TEST | 1.93 MB | 9,551 | 30 | 0 |
| `Test_Outpatientdata-1542969243754.csv` | TEST | 17.94 MB | 125,841 | 27 | 0 |

> [!NOTE]
> All raw CSV files have been verified and isolated in `data/raw/train/` and `data/raw/test/` without modifying original contents.

---

## 2. Key Identification & Fraud Target Validation

- **Provider Primary Key**: Identified as column `Provider` in Provider CSVs (`Train-1542865627584.csv` and `Test-1542969243754.csv`).
- **Beneficiary Primary Key**: Identified as column `BeneID` in Beneficiary CSVs.
- **Claim Key**: Identified as column `ClaimID` in Inpatient and Outpatient claims CSVs.
- **Fraud Target Column**:
  - `PotentialFraud` exists in `Train-1542865627584.csv` (TRAIN provider file).
  - **Confirmed**: `PotentialFraud` is **completely absent** in `Test-1542969243754.csv` (TEST provider file), confirming that the TEST dataset is strictly unlabeled for final evaluation.

---

## 3. Fraud Target Class Imbalance Analysis (`PotentialFraud`)

- **Non-Fraud Providers (`No`)**: `4,904` (90.65%)
- **Fraudulent Providers (`Yes`)**: `506` (9.35%)
- **Class Imbalance Ratio**: Approximately **9.69:1** (Non-Fraud to Fraud).

> [!IMPORTANT]
> The target variable exhibits significant class imbalance (~9% fraud rate at the provider level). Appropriate cost-sensitive learning, SMOTE/oversampling, or PR-AUC / ROC-AUC evaluation metrics must be employed during modeling.

---

## 4. Target Leakage & Data Integrity Assessment

1. **Target Leakage**:
   - Scanned all claims and beneficiary files for explicit target markers or post-audit resolution flags.
   - No direct `PotentialFraud` or resolution outcome column exists in claims or beneficiary datasets.
2. **Missing Value Hotspots**:
   - `DOD` (Date of Death): High missing rate (~99% in Beneficiary files), expected as most beneficiaries are alive.
   - Secondary & Tertiary Diagnosis/Procedure Codes (`ClmDiagnosisCode_2..10`, `ClmProcedureCode_1..6`): High null counts, expected as not all claims require full 10 diagnosis or 6 procedure codes.
   - `OperatingPhysician` & `OtherPhysician`: High null percentages in outpatient claims.

---

## 5. Temporal & Categorical Column Inventory

- **Temporal / Date Columns**:
  - `DOB`, `DOD` (Beneficiary birth/death dates)
  - `ClaimStartDt`, `ClaimEndDt` (Claim service window)
  - `AdmissionDt`, `DischargeDt` (Inpatient admission window)
- **Demographic & Medical Categorical Columns**:
  - `Gender`, `Race`, `State`, `County`, `RenalDiseaseIndicator`
  - Chronic condition flags (`ChronicCond_Alzheimer`, `ChronicCond_Heartfailure`, `ChronicCond_KidneyDisease`, `ChronicCond_Cancer`, etc.)
  - `DiagnosisGroupCode`, `ClmAdmitDiagnosisCode`

---

## 6. Recommendations for Downstream Pipeline

1. **Strict Train/Test Partitioning**:
   - DO NOT combine Kaggle TRAIN and TEST datasets.
   - Perform all validation folds strictly within `data/raw/train`.
2. **Provider-Level Feature Aggregation**:
   - Aggregate Inpatient and Outpatient claim metrics (claim counts, mean reimbursement, length of stay, unique beneficiary counts, physician involvement ratios) at the `Provider` level.
3. **Data Dictionary**:
   - Detailed column metadata saved to [`reports/data_dictionary.csv`](file:///C:/Users/tejes/OneDrive/Desktop/Fraud claim/reports/data_dictionary.csv).
4. **Relational Analysis**:
   - Detailed join and entity analysis saved to [`reports/dataset_relationships.md`](file:///C:/Users/tejes/OneDrive/Desktop/Fraud claim/reports/dataset_relationships.md).
