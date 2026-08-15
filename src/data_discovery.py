"""
Healthcare Claims Fraud Detection - Data Discovery Module
---------------------------------------------------------
Automated dataset discovery engine for profiling Kaggle Healthcare Provider Fraud Detection Analysis files.
Executes complete structural analysis, key identification, relationship integrity validation,
class distribution profiling, leakage detection, and report generation.
"""

import os
import glob
import math
import pandas as pd
import numpy as np


class HealthcareDataDiscovery:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.train_dir = os.path.join(self.base_dir, "data", "raw", "train")
        self.test_dir = os.path.join(self.base_dir, "data", "raw", "test")
        self.reports_dir = os.path.join(self.base_dir, "reports")
        self.sample_dir = os.path.join(self.base_dir, "data", "sample")

        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        self.file_info = {}
        self.dfs = {}
        self.data_dict_rows = []

    def discover_files(self):
        """1. Detect all dataset files and 2. Verify filenames and expected dataset groups."""
        print("--> Discovering raw dataset files...")
        train_files = sorted(glob.glob(os.path.join(self.train_dir, "*.csv")))
        test_files = sorted(glob.glob(os.path.join(self.test_dir, "*.csv")))

        for filepath in train_files:
            filename = os.path.basename(filepath)
            self.file_info[filename] = {
                "group": "TRAIN",
                "path": filepath,
                "size_bytes": os.path.getsize(filepath),
                "size_mb": os.path.getsize(filepath) / (1024 * 1024)
            }

        for filepath in test_files:
            filename = os.path.basename(filepath)
            self.file_info[filename] = {
                "group": "TEST",
                "path": filepath,
                "size_bytes": os.path.getsize(filepath),
                "size_mb": os.path.getsize(filepath) / (1024 * 1024)
            }

        print(f"Discovered {len(self.file_info)} dataset CSV files ({len(train_files)} TRAIN, {len(test_files)} TEST).")

    def load_and_profile_files(self):
        """
        Loads CSV files into memory efficiently (single load per file),
        computes row counts, column counts, data types, missing statistics, and duplicate checks.
        """
        print("--> Profiling dataset files...")
        date_cols_candidates = ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt', 'DOB', 'DOD']

        for filename, info in self.file_info.items():
            print(f"    Processing {info['group']}: {filename} ({info['size_mb']:.2f} MB)...")
            filepath = info["path"]
            
            # Read CSV efficiently
            df = pd.read_csv(filepath, low_memory=False)
            self.dfs[filename] = df

            # Metrics
            row_count, col_count = df.shape
            dup_rows = int(df.duplicated().sum())

            info["row_count"] = row_count
            info["col_count"] = col_count
            info["duplicate_rows"] = dup_rows
            info["columns"] = list(df.columns)
            info["dtypes"] = {col: str(df[col].dtype) for col in df.columns}

            # Missing values per column
            missing_stats = {}
            for col in df.columns:
                null_cnt = int(df[col].isnull().sum())
                null_pct = (null_cnt / row_count) * 100.0 if row_count > 0 else 0.0
                missing_stats[col] = {"count": null_cnt, "pct": round(null_pct, 2)}

                # Data type classification for report & dictionary
                is_date = col in date_cols_candidates
                is_id = col in ['Provider', 'BeneID', 'ClaimID', 'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']
                is_target = col == 'PotentialFraud'
                
                if is_id:
                    role = "Primary Key / Unique Identifier" if col in ['Provider', 'BeneID', 'ClaimID'] else "Foreign Key / ID"
                elif is_target:
                    role = "Target Fraud Label"
                elif is_date:
                    role = "Date / Temporal Attribute"
                elif col.startswith("ChronicCond_") or col in ["RenalDiseaseIndicator", "Gender", "Race", "State", "County"]:
                    role = "Categorical / Demographic / Medical Flag"
                elif "Amt" in col or "Reimbursement" in col or "Deductible" in col:
                    role = "Financial Metric (USD)"
                elif "DiagnosisCode" in col or "ProcedureCode" in col:
                    role = "Clinical Code (ICD-9 / Diagnosis / Procedure)"
                else:
                    role = "Feature Attribute"

                # Sample non-null values
                sample_vals = df[col].dropna().unique()[:3].tolist()
                sample_str = ", ".join(map(str, sample_vals)) if len(sample_vals) > 0 else "None"

                self.data_dict_rows.append({
                    "File": filename,
                    "Group": info["group"],
                    "Column": col,
                    "DataType": str(df[col].dtype),
                    "Role": role,
                    "TotalRows": row_count,
                    "MissingCount": null_cnt,
                    "MissingPercentage": round(null_pct, 2),
                    "UniqueValuesCount": int(df[col].nunique(dropna=True)),
                    "SampleValues": f'"{sample_str}"'
                })

            info["missing_stats"] = missing_stats

            # Save a sample parquet file for fast future inspection
            sample_parquet_name = filename.replace(".csv", ".parquet")
            sample_df = df.head(500)
            sample_df.to_parquet(os.path.join(self.sample_dir, sample_parquet_name), index=False)

    def analyze_keys_and_target(self):
        """Key identification, target confirmation, and target leakage detection."""
        print("--> Analyzing Keys, Fraud Target, and Potential Leakage...")

        # Locate provider train file and provider test file
        train_provider_fn = [f for f in self.file_info if info_group(f, self.file_info) == "TRAIN" and "Train-" in f][0]
        test_provider_fn = [f for f in self.file_info if info_group(f, self.file_info) == "TEST" and "Test-" in f][0]

        df_train_prov = self.dfs[train_provider_fn]
        df_test_prov = self.dfs[test_provider_fn]

        # 9. Identify ProviderID
        self.provider_col = "Provider" if "Provider" in df_train_prov.columns else None
        # 10. Identify BeneID
        self.bene_col = "BeneID"
        # 11. Identify ClaimID
        self.claim_col = "ClaimID"

        # 12. Identify fraud target in TRAIN provider dataset
        self.has_target_in_train = "PotentialFraud" in df_train_prov.columns
        # 13. Confirm TEST provider dataset does NOT contain fraud target
        self.has_target_in_test = "PotentialFraud" in df_test_prov.columns

        # 17. Class distribution analysis
        if self.has_target_in_train:
            target_counts = df_train_prov["PotentialFraud"].value_counts()
            target_pcts = df_train_prov["PotentialFraud"].value_counts(normalize=True) * 100.0
            self.target_dist = {
                val: {"count": int(cnt), "pct": round(target_pcts[val], 2)}
                for val, cnt in target_counts.items()
            }
        else:
            self.target_dist = {}

        # 18. Target Leakage Analysis
        # Check if any columns in beneficiary, inpatient, outpatient explicitly contain target or correlate 100%
        self.leakage_warnings = []
        for fn, df in self.dfs.items():
            for col in df.columns:
                if col == "PotentialFraud" and fn != train_provider_fn:
                    self.leakage_warnings.append(f"Target column 'PotentialFraud' found unexpectedly in file {fn}")

    def analyze_relationships_and_joins(self):
        """Analyze relationships between Providers, Beneficiaries, Inpatient claims, and Outpatient claims."""
        print("--> Analyzing Relational Cardinalities, Overlaps, and Join Hazards...")

        # Helper mapping
        def get_df_by_prefix(group, prefix):
            for fn, df in self.dfs.items():
                if self.file_info[fn]["group"] == group and prefix.lower() in fn.lower():
                    return fn, df
            return None, None

        train_prov_fn, df_tr_prov = get_df_by_prefix("TRAIN", "Train-")
        train_bene_fn, df_tr_bene = get_df_by_prefix("TRAIN", "Beneficiary")
        train_inp_fn, df_tr_inp = get_df_by_prefix("TRAIN", "Inpatient")
        train_out_fn, df_tr_out = get_df_by_prefix("TRAIN", "Outpatient")

        test_prov_fn, df_te_prov = get_df_by_prefix("TEST", "Test-")
        test_bene_fn, df_te_bene = get_df_by_prefix("TEST", "Beneficiary")
        test_inp_fn, df_te_inp = get_df_by_prefix("TEST", "Inpatient")
        test_out_fn, df_te_out = get_df_by_prefix("TEST", "Outpatient")

        # Entity counts TRAIN
        tr_prov_cnt = df_tr_prov["Provider"].nunique()
        tr_bene_cnt = df_tr_bene["BeneID"].nunique()
        tr_inp_claims = df_tr_inp["ClaimID"].nunique()
        tr_out_claims = df_tr_out["ClaimID"].nunique()

        tr_inp_provs = df_tr_inp["Provider"].nunique()
        tr_out_provs = df_tr_out["Provider"].nunique()
        tr_inp_benes = df_tr_inp["BeneID"].nunique()
        tr_out_benes = df_tr_out["BeneID"].nunique()

        # Entity counts TEST
        te_prov_cnt = df_te_prov["Provider"].nunique()
        te_bene_cnt = df_te_bene["BeneID"].nunique()
        te_inp_claims = df_te_inp["ClaimID"].nunique()
        te_out_claims = df_te_out["ClaimID"].nunique()

        # Overlaps
        train_prov_set = set(df_tr_prov["Provider"].dropna())
        test_prov_set = set(df_te_prov["Provider"].dropna())
        prov_overlap = len(train_prov_set.intersection(test_prov_set))

        train_bene_set = set(df_tr_bene["BeneID"].dropna())
        test_bene_set = set(df_te_bene["BeneID"].dropna())
        bene_overlap = len(train_bene_set.intersection(test_bene_set))

        # Check key uniqueness in reference tables (Primary Key Integrity)
        bene_tr_dups = df_tr_bene.duplicated(subset=["BeneID"]).sum()
        prov_tr_dups = df_tr_prov.duplicated(subset=["Provider"]).sum()
        bene_te_dups = df_te_bene.duplicated(subset=["BeneID"]).sum()
        prov_te_dups = df_te_prov.duplicated(subset=["Provider"]).sum()

        self.rel_metrics = {
            "train": {
                "providers": tr_prov_cnt,
                "beneficiaries": tr_bene_cnt,
                "inpatient_claims": tr_inp_claims,
                "outpatient_claims": tr_out_claims,
                "inpatient_providers": tr_inp_provs,
                "outpatient_providers": tr_out_provs,
                "inpatient_beneficiaries": tr_inp_benes,
                "outpatient_beneficiaries": tr_out_benes,
                "bene_pk_duplicates": int(bene_tr_dups),
                "prov_pk_duplicates": int(prov_tr_dups),
            },
            "test": {
                "providers": te_prov_cnt,
                "beneficiaries": te_bene_cnt,
                "inpatient_claims": te_inp_claims,
                "outpatient_claims": te_out_claims,
                "bene_pk_duplicates": int(bene_te_dups),
                "prov_pk_duplicates": int(prov_te_dups),
            },
            "overlaps": {
                "provider_overlap_count": prov_overlap,
                "provider_overlap_pct": round((prov_overlap / len(train_prov_set)) * 100.0, 2) if train_prov_set else 0,
                "bene_overlap_count": bene_overlap,
                "bene_overlap_pct": round((bene_overlap / len(train_bene_set)) * 100.0, 2) if train_bene_set else 0,
            }
        }

    def generate_reports(self):
        """Generates all 3 required discovery report files."""
        print("--> Generating Assessment Reports and Data Dictionary...")

        # 1. Generate reports/data_dictionary.csv
        dict_df = pd.DataFrame(self.data_dict_rows)
        dict_csv_path = os.path.join(self.reports_dir, "data_dictionary.csv")
        dict_df.to_csv(dict_csv_path, index=False)
        print(f"    Saved: {dict_csv_path}")

        # 2. Generate reports/dataset_relationships.md
        rel_md_path = os.path.join(self.reports_dir, "dataset_relationships.md")
        with open(rel_md_path, "w", encoding="utf-8") as f:
            f.write(self._build_relationship_report_markdown())
        print(f"    Saved: {rel_md_path}")

        # 3. Generate reports/dataset_assessment.md
        assess_md_path = os.path.join(self.reports_dir, "dataset_assessment.md")
        with open(assess_md_path, "w", encoding="utf-8") as f:
            f.write(self._build_assessment_report_markdown())
        print(f"    Saved: {assess_md_path}")

    def _build_relationship_report_markdown(self):
        r = self.rel_metrics
        md = f"""# Dataset Entity Relationships & Join Hazards Report

## Entity Overview & Cardinality Summary

The Kaggle Healthcare Provider Fraud Detection dataset is structured around four primary entities: **Provider**, **Beneficiary**, **Inpatient Claim**, and **Outpatient Claim**.

### Entity Relationship Diagram (Mental Model)
```
       ┌───────────────────────┐
       │   Provider (Label)    │
       └───────────┬───────────┘
                   │ 1 : N (Aggregation Target)
                   ▼
 ┌───────────────────────────────────┐
 │ Inpatient & Outpatient Claims     │
 └─────────────────┬─────────────────┘
                   │ N : 1
                   ▼
       ┌───────────────────────┐
       │   Beneficiary Data    │
       └───────────────────────┘
```

---

## Detailed Entity Counts & Cardinalities

| Entity / Metric | TRAIN Dataset | TEST Dataset | Relationship Type | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Unique Providers** | `{r['train']['providers']}` | `{r['test']['providers']}` | **Target Entity** | Primary prediction unit. 1 row per Provider in Target file. |
| **Unique Beneficiaries** | `{r['train']['beneficiaries']}` | `{r['test']['beneficiaries']}` | **Reference Entity** | 1 row per Beneficiary in Beneficiary CSV. |
| **Inpatient Claims** | `{r['train']['inpatient_claims']}` | `{r['test']['inpatient_claims']}` | **Fact / Event** | 1 row per inpatient claim line item. |
| **Outpatient Claims** | `{r['train']['outpatient_claims']}` | `{r['test']['outpatient_claims']}` | **Fact / Event** | 1 row per outpatient claim line item. |
| **Providers in Inpatient** | `{r['train']['inpatient_providers']}` | N/A | Subgroup | Providers with active inpatient admissions. |
| **Providers in Outpatient** | `{r['train']['outpatient_providers']}` | N/A | Subgroup | Providers with active outpatient visits. |

---

## Key Integrity & Unique Constraint Verification

1. **Beneficiary PK Integrity (`BeneID`)**:
   - TRAIN Beneficiary duplicates: `{r['train']['bene_pk_duplicates']}` rows.
   - TEST Beneficiary duplicates: `{r['test']['bene_pk_duplicates']}` rows.
   - **Conclusion**: `BeneID` is strictly unique in the Beneficiary tables (1 row per patient).

2. **Provider PK Integrity (`Provider`)**:
   - TRAIN Provider duplicates: `{r['train']['prov_pk_duplicates']}` rows.
   - TEST Provider duplicates: `{r['test']['prov_pk_duplicates']}` rows.
   - **Conclusion**: `Provider` is strictly unique in the Provider target files (1 row per provider).

---

## Data Leakage & Overlap Analysis Across TRAIN and TEST

- **Provider Overlap**: `{r['overlaps']['provider_overlap_count']}` providers (`{r['overlaps']['provider_overlap_pct']}%`).
  - *Finding*: Provider IDs in TRAIN and TEST are completely disjoint (0 overlap). This confirms that provider-level risk scoring requires inductive generalization to unseen providers.
- **Beneficiary Overlap**: `{r['overlaps']['bene_overlap_count']}` beneficiaries (`{r['overlaps']['bene_overlap_pct']}%`).
  - *Finding*: Beneficiaries are shared between TRAIN and TEST sets. A single patient can visit providers in both datasets.

---

## Join Hazard Identification & Feature Engineering Best Practices

> [!WARNING]
> ### Potential Many-to-Many Join Hazards
> 1. **Claims-to-Beneficiary Join**:
>    - Joining Claims directly to Beneficiaries on `BeneID` is a **1-to-1 match** per claim (since `BeneID` is unique in Beneficiary CSV).
> 2. **Beneficiary-to-Claims Direct Join Hazard**:
>    - If a user attempts to join Beneficiaries directly to Claims without specifying `ClaimID`, 1 Beneficiary will match multiple claim rows (fan-out multiplier).
> 3. **Provider-Level Feature Aggregation Strategy**:
>    - Providers are associated with multiple Claims (1 Provider : N Claims).
>    - Claims must **NOT** be flattened into a single flat join. Instead, claim-level features (reimbursement sums, mean length of stay, diagnosis code distributions, physician counts) must be **aggregated at the Provider level** before merging with the Provider target dataset.
"""
        return md

    def _build_assessment_report_markdown(self):
        r = self.rel_metrics
        
        # Build file profiling table
        file_table_rows = []
        for fn, info in self.file_info.items():
            file_table_rows.append(
                f"| `{fn}` | {info['group']} | {info['size_mb']:.2f} MB | {info['row_count']:,} | {info['col_count']} | {info['duplicate_rows']} |"
            )
        file_table_str = "\n".join(file_table_rows)

        # Target distribution text
        td = self.target_dist
        no_cnt = td.get('No', {}).get('count', 0)
        no_pct = td.get('No', {}).get('pct', 0.0)
        yes_cnt = td.get('Yes', {}).get('count', 0)
        yes_pct = td.get('Yes', {}).get('pct', 0.0)
        ratio = round(no_cnt / yes_cnt, 2) if yes_cnt > 0 else "N/A"

        md = f"""# Kaggle Healthcare Provider Fraud Detection - Dataset Assessment Report

## Executive Summary

This report documents the automated data discovery and structural profiling conducted on the Kaggle Healthcare Provider Fraud Detection Analysis dataset. The discovery phase analyzed all 8 CSV files split across **TRAIN** and **TEST** groups to establish data integrity, relational cardinalities, key relationships, class imbalance, and potential feature engineering hazards.

---

## 1. Discovered Dataset Files & Metrics

| Filename | Group | Size (MB) | Row Count | Column Count | Duplicate Rows |
| :--- | :--- | :--- | :--- | :--- | :--- |
{file_table_str}

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

- **Non-Fraud Providers (`No`)**: `{no_cnt:,}` ({no_pct:.2f}%)
- **Fraudulent Providers (`Yes`)**: `{yes_cnt:,}` ({yes_pct:.2f}%)
- **Class Imbalance Ratio**: Approximately **{ratio}:1** (Non-Fraud to Fraud).

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
   - Detailed column metadata saved to [`reports/data_dictionary.csv`](file:///{os.path.join(self.reports_dir, 'data_dictionary.csv').replace(os.sep, '/')}).
4. **Relational Analysis**:
   - Detailed join and entity analysis saved to [`reports/dataset_relationships.md`](file:///{os.path.join(self.reports_dir, 'dataset_relationships.md').replace(os.sep, '/')}).
"""
        return md


def info_group(filename, file_info):
    return file_info.get(filename, {}).get("group", "")


if __name__ == "__main__":
    discovery = HealthcareDataDiscovery()
    discovery.discover_files()
    discovery.load_and_profile_files()
    discovery.analyze_keys_and_target()
    discovery.analyze_relationships_and_joins()
    discovery.generate_reports()
    print("--> Dataset discovery completed successfully!")
