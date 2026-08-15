"""
Healthcare Claims Preprocessing & Data Integration Module
---------------------------------------------------------
Handles raw CSV loading, date conversions, temporal feature creation,
inpatient/outpatient claim unification, beneficiary 1:1 joins, missing value handling,
duplicate checks, logging, and Parquet persistence for TRAIN and TEST datasets.
"""

import os
import glob
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


class HealthcarePreprocessor:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.raw_train_dir = os.path.join(self.base_dir, "data", "raw", "train")
        self.raw_test_dir = os.path.join(self.base_dir, "data", "raw", "test")
        self.processed_dir = os.path.join(self.base_dir, "data", "processed")

        os.makedirs(self.processed_dir, exist_ok=True)

        self.chronic_cols = [
            'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
            'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
            'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
            'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke'
        ]

    def _find_file(self, directory, pattern):
        matches = glob.glob(os.path.join(directory, pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching pattern '{pattern}' found in {directory}")
        return matches[0]

    def load_raw_group(self, group="train"):
        """Load raw CSV files for train or test group."""
        raw_dir = self.raw_train_dir if group == "train" else self.raw_test_dir
        logging.info(f"Loading raw CSVs for group: '{group.upper()}' from {raw_dir}...")

        bene_path = self.find_file_in_dir(raw_dir, "*Beneficiarydata*.csv")
        inp_path = self.find_file_in_dir(raw_dir, "*Inpatientdata*.csv")
        out_path = self.find_file_in_dir(raw_dir, "*Outpatientdata*.csv")

        df_bene = pd.read_csv(bene_path, low_memory=False)
        df_inp = pd.read_csv(inp_path, low_memory=False)
        df_out = pd.read_csv(out_path, low_memory=False)

        logging.info(f"[{group.upper()}] Loaded Beneficiary: {df_bene.shape[0]:,} rows, {df_bene.shape[1]} cols")
        logging.info(f"[{group.upper()}] Loaded Inpatient: {df_inp.shape[0]:,} rows, {df_inp.shape[1]} cols")
        logging.info(f"[{group.upper()}] Loaded Outpatient: {df_out.shape[0]:,} rows, {df_out.shape[1]} cols")

        return df_bene, df_inp, df_out

    def find_file_in_dir(self, directory, pattern):
        matches = glob.glob(os.path.join(directory, pattern))
        if not matches:
            raise FileNotFoundError(f"Pattern {pattern} not found in {directory}")
        return matches[0]

    def clean_beneficiary_data(self, df_bene):
        """Clean beneficiary dataset and construct patient health risk flags."""
        df = df_bene.copy()
        init_rows = len(df)

        # Handle duplicates in Beneficiary data
        dups = df.duplicated(subset=['BeneID']).sum()
        if dups > 0:
            logging.warning(f"Found {dups} duplicate BeneIDs. Dropping duplicates...")
            df = df.drop_duplicates(subset=['BeneID'])
            logging.info(f"Beneficiary rows after deduplication: {len(df):,}")

        # Date parsing
        df['DOB'] = pd.to_datetime(df['DOB'], format='%Y-%m-%d', errors='coerce')
        df['DOD'] = pd.to_datetime(df['DOD'], format='%Y-%m-%d', errors='coerce')

        # Renal Disease Indicator: convert '0' -> 0, 'Y' -> 1
        df['RenalDiseaseIndicator'] = df['RenalDiseaseIndicator'].astype(str).str.strip().map({'0': 0, 'Y': 1}).fillna(0).astype(int)

        # Chronic conditions: convert raw 1 (Yes), 2 (No) -> 1 (Yes), 0 (No)
        for col in self.chronic_cols:
            if col in df.columns:
                df[col] = df[col].map({1: 1, 2: 0}).fillna(0).astype(int)

        # Total Chronic Conditions Count per patient
        df['ChronicCond_Count'] = df[self.chronic_cols].sum(axis=1)

        # Financial missing values (fill NaN with 0.0)
        fin_cols = ['IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt']
        for col in fin_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        logging.info(f"Cleaned Beneficiary Data: {len(df):,} valid records (0 dropped)")
        return df

    def prepare_claims(self, df_inp, df_out):
        """Unify inpatient and outpatient claims into a single consolidated claim representation."""
        df_inp_c = df_inp.copy()
        df_out_c = df_out.copy()

        df_inp_c['IsInpatient'] = 1
        df_out_c['IsInpatient'] = 0

        # Outpatient claims do not have AdmissionDt / DischargeDt / DiagnosisGroupCode
        for col in ['AdmissionDt', 'DischargeDt', 'DiagnosisGroupCode']:
            if col not in df_out_c.columns:
                df_out_c[col] = np.nan

        # Deduplicate exact duplicate claims if any
        inp_dups = df_inp_c.duplicated(subset=['ClaimID']).sum()
        out_dups = df_out_c.duplicated(subset=['ClaimID']).sum()
        if inp_dups > 0:
            logging.warning(f"Inpatient duplicate ClaimIDs found: {inp_dups}. Deduplicating...")
            df_inp_c = df_inp_c.drop_duplicates(subset=['ClaimID'])
        if out_dups > 0:
            logging.warning(f"Outpatient duplicate ClaimIDs found: {out_dups}. Deduplicating...")
            df_out_c = df_out_c.drop_duplicates(subset=['ClaimID'])

        # Concatenate unified claims
        unified_claims = pd.concat([df_inp_c, df_out_c], ignore_index=True)
        logging.info(f"Unified Claims Count: {len(unified_claims):,} ({len(df_inp_c):,} Inpatient, {len(df_out_c):,} Outpatient)")

        # Date parsing
        date_cols = ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt']
        for col in date_cols:
            unified_claims[col] = pd.to_datetime(unified_claims[col], format='%Y-%m-%d', errors='coerce')

        # Feature Creation: Claim Duration (in days)
        unified_claims['ClaimDuration'] = (unified_claims['ClaimEndDt'] - unified_claims['ClaimStartDt']).dt.days + 1
        # Fill any invalid negative or zero durations with 1
        unified_claims['ClaimDuration'] = unified_claims['ClaimDuration'].apply(lambda x: max(1, x) if pd.notnull(x) else 1)

        # Feature Creation: Inpatient Stay Duration (in days)
        unified_claims['InpatientStayDuration'] = np.where(
            unified_claims['IsInpatient'] == 1,
            (unified_claims['DischargeDt'] - unified_claims['AdmissionDt']).dt.days + 1,
            np.nan
        )

        # Financial Amounts
        unified_claims['InscClaimAmtReimbursed'] = pd.to_numeric(unified_claims['InscClaimAmtReimbursed'], errors='coerce').fillna(0.0)
        unified_claims['DeductibleAmtPaid'] = pd.to_numeric(unified_claims['DeductibleAmtPaid'], errors='coerce').fillna(0.0)

        # Count non-null diagnosis & procedure codes per claim
        diag_cols = [c for c in unified_claims.columns if c.startswith('ClmDiagnosisCode_')]
        proc_cols = [c for c in unified_claims.columns if c.startswith('ClmProcedureCode_')]

        unified_claims['NumDiagnosisCodes'] = unified_claims[diag_cols].notnull().sum(axis=1)
        unified_claims['NumProcedureCodes'] = unified_claims[proc_cols].notnull().sum(axis=1)

        return unified_claims

    def merge_claims_and_beneficiaries(self, claims_df, bene_df, group="train"):
        """Merge unified claims with beneficiary reference dataset via strict 1:1 join on BeneID."""
        init_claim_rows = len(claims_df)
        logging.info(f"[{group.upper()}] Joining {init_claim_rows:,} claims with {len(bene_df):,} beneficiaries on BeneID...")

        merged = claims_df.merge(bene_df, on='BeneID', how='left')
        final_rows = len(merged)

        # Validate join integrity (no row multiplication)
        if final_rows != init_claim_rows:
            logging.error(f"[{group.upper()}] Join Integrity Failure! Input rows: {init_claim_rows}, Output rows: {final_rows}")
            raise ValueError(f"Beneficiary join expanded row count from {init_claim_rows} to {final_rows}")
        else:
            logging.info(f"[{group.upper()}] Join Integrity Passed: Exactly {final_rows:,} rows preserved.")

        # Derived Patient-Claim Features
        # Age at claim start date
        merged['PatientAgeAtClaim'] = np.where(
            merged['DOB'].notnull() & merged['ClaimStartDt'].notnull(),
            merged['ClaimStartDt'].dt.year - merged['DOB'].dt.year - (
                (merged['ClaimStartDt'].dt.month < merged['DOB'].dt.month) |
                ((merged['ClaimStartDt'].dt.month == merged['DOB'].dt.month) & (merged['ClaimStartDt'].dt.day < merged['DOB'].dt.day))
            ).astype(int),
            np.nan
        )

        # IsDeceased flag at claim date
        merged['IsDeceased'] = np.where(
            merged['DOD'].notnull() & (merged['DOD'] <= merged['ClaimStartDt']),
            1, 0
        )

        return merged

    def process_and_save(self, group="train"):
        """Full pipeline execution for train or test group."""
        df_bene, df_inp, df_out = self.load_raw_group(group=group)
        clean_bene = self.clean_beneficiary_data(df_bene)
        unified_claims = self.prepare_claims(df_inp, df_out)
        integrated_df = self.merge_claims_and_beneficiaries(unified_claims, clean_bene, group=group)

        out_parquet_path = os.path.join(self.processed_dir, f"{group}_claims_integrated.parquet")
        integrated_df.to_parquet(out_parquet_path, index=False)
        logging.info(f"[{group.upper()}] Saved integrated claims dataset: {out_parquet_path} ({os.path.getsize(out_parquet_path)/(1024*1024):.2f} MB)")

        return integrated_df


if __name__ == "__main__":
    preprocessor = HealthcarePreprocessor()
    df_train_integrated = preprocessor.process_and_save(group="train")
    df_test_integrated = preprocessor.process_and_save(group="test")
    print("--> Data Preprocessing & Integration Pipeline completed successfully!")
