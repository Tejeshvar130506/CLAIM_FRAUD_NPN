"""
Healthcare Provider Risk Scoring - Feature Engineering Module
--------------------------------------------------------------
Aggregates claim-level integrated datasets to the Provider primary key level.
Constructs provider-level risk scoring features across 6 distinct feature categories.
Applies target labeling strictly to TRAIN provider features and guarantees zero target leakage.
"""

import os
import glob
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


class ProviderFeatureExtractor:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.processed_dir = os.path.join(self.base_dir, "data", "processed")
        self.features_dir = os.path.join(self.base_dir, "data", "features")
        self.raw_train_dir = os.path.join(self.base_dir, "data", "raw", "train")
        self.raw_test_dir = os.path.join(self.base_dir, "data", "raw", "test")

        os.makedirs(self.features_dir, exist_ok=True)

    def extract_provider_features(self, group="train"):
        """Extract provider-level feature vectors from integrated claims dataset."""
        parquet_path = os.path.join(self.processed_dir, f"{group}_claims_integrated.parquet")
        logging.info(f"[{group.upper()}] Loading integrated claims from {parquet_path}...")
        df_claims = pd.read_parquet(parquet_path)

        logging.info(f"[{group.upper()}] Computing Provider-level feature aggregations over {len(df_claims):,} claims...")

        # Same attending and operating physician indicator
        df_claims['SameAttendingOperating'] = (
            df_claims['AttendingPhysician'].notnull() &
            df_claims['OperatingPhysician'].notnull() &
            (df_claims['AttendingPhysician'] == df_claims['OperatingPhysician'])
        ).astype(int)

        # Basic GroupBy aggregations
        agg_dict = {
            'ClaimID': ['count'],
            'IsInpatient': ['sum', 'mean'],
            'BeneID': ['nunique'],
            'InscClaimAmtReimbursed': ['sum', 'mean', 'max', 'std'],
            'DeductibleAmtPaid': ['sum', 'mean', 'max'],
            'IPAnnualReimbursementAmt': ['mean'],
            'OPAnnualReimbursementAmt': ['mean'],
            'ClaimDuration': ['mean', 'max', 'std'],
            'InpatientStayDuration': ['mean', 'max'],
            'AttendingPhysician': ['nunique'],
            'OperatingPhysician': ['nunique'],
            'OtherPhysician': ['nunique'],
            'SameAttendingOperating': ['mean'],
            'ClmAdmitDiagnosisCode': ['nunique'],
            'DiagnosisGroupCode': ['nunique'],
            'NumDiagnosisCodes': ['mean'],
            'NumProcedureCodes': ['mean'],
            'PatientAgeAtClaim': ['mean', 'std'],
            'IsDeceased': ['sum', 'mean'],
            'ChronicCond_Count': ['mean'],
            'RenalDiseaseIndicator': ['mean'],
            'Gender': [lambda x: (x == 1).mean(), lambda x: (x == 2).mean()],
            'State': ['nunique'],
            'County': ['nunique'],
        }

        # Flatten column names
        prov_grp = df_claims.groupby('Provider')
        feat_df = prov_grp.agg(agg_dict)

        feat_df.columns = [
            'TotalClaims',
            'InpatientClaimsCount', 'InpatientRatio',
            'UniqueBeneficiaries',
            'TotalInscClaimAmtReimbursed', 'MeanInscClaimAmtReimbursed', 'MaxInscClaimAmtReimbursed', 'StdInscClaimAmtReimbursed',
            'TotalDeductibleAmtPaid', 'MeanDeductibleAmtPaid', 'MaxDeductibleAmtPaid',
            'MeanIPAnnualReimbursementAmt',
            'MeanOPAnnualReimbursementAmt',
            'MeanClaimDuration', 'MaxClaimDuration', 'StdClaimDuration',
            'MeanInpatientStayDuration', 'MaxInpatientStayDuration',
            'UniqueAttendingPhysicians', 'UniqueOperatingPhysicians', 'UniqueOtherPhysicians',
            'SameAttendingOperatingRatio',
            'UniqueAdmitDiagnosisCodes', 'UniqueGroupDiagnosisCodes',
            'MeanNumDiagnosisCodes', 'MeanNumProcedureCodes',
            'MeanPatientAge', 'StdPatientAge',
            'DeceasedPatientCount', 'DeceasedPatientRatio',
            'MeanChronicCondCount',
            'RenalDiseaseRatio',
            'GenderMaleRatio', 'GenderFemaleRatio',
            'UniqueStatesServed', 'UniqueCountiesServed'
        ]

        # Derived Ratios
        feat_df['OutpatientClaimsCount'] = feat_df['TotalClaims'] - feat_df['InpatientClaimsCount']
        feat_df['ClaimsPerBeneficiary'] = feat_df['TotalClaims'] / np.maximum(1, feat_df['UniqueBeneficiaries'])
        feat_df['ReimbursementToDeductibleRatio'] = feat_df['TotalInscClaimAmtReimbursed'] / (feat_df['TotalDeductibleAmtPaid'] + 1.0)
        feat_df['PhysicianToClaimRatio'] = feat_df['UniqueAttendingPhysicians'] / feat_df['TotalClaims']

        # Extract Unique Diagnosis & Procedure Codes across all slots per provider
        diag_cols = [c for c in df_claims.columns if c.startswith('ClmDiagnosisCode_')]
        proc_cols = [c for c in df_claims.columns if c.startswith('ClmProcedureCode_')]

        def count_unique_codes(df_group, code_cols):
            codes = df_group[code_cols].values.ravel()
            codes = codes[pd.notnull(codes)]
            return len(np.unique(codes))

        logging.info(f"[{group.upper()}] Computing unique diagnosis & procedure code diversity per provider...")
        unique_diag_series = prov_grp.apply(lambda g: count_unique_codes(g, diag_cols), include_groups=False)
        unique_proc_series = prov_grp.apply(lambda g: count_unique_codes(g, proc_cols), include_groups=False)

        feat_df['TotalUniqueDiagnosisCodes'] = unique_diag_series
        feat_df['TotalUniqueProcedureCodes'] = unique_proc_series

        # Fill any remaining NaNs in standard deviations or stay metrics with 0.0
        feat_df = feat_df.fillna(0.0).reset_index()

        logging.info(f"[{group.upper()}] Extracted {feat_df.shape[1]} features for {feat_df.shape[0]:,} unique Providers.")

        # Target Labeling for TRAIN dataset ONLY
        if group == "train":
            prov_target_path = glob.glob(os.path.join(self.raw_train_dir, "Train-*.csv"))[0]
            df_target = pd.read_csv(prov_target_path)
            # Map PotentialFraud: 'Yes' -> 1, 'No' -> 0
            df_target['PotentialFraud'] = df_target['PotentialFraud'].map({'Yes': 1, 'No': 0})
            
            feat_df = feat_df.merge(df_target[['Provider', 'PotentialFraud']], on='Provider', how='left')
            logging.info(f"[TRAIN] Target PotentialFraud attached. Positive Fraud Count: {feat_df['PotentialFraud'].sum()} / {len(feat_df)}")
        else:
            logging.info(f"[TEST] Confirmed zero target label attachment. Feature matrix strictly unlabeled.")

        out_features_path = os.path.join(self.features_dir, f"{group}_provider_features.parquet")
        feat_df.to_parquet(out_features_path, index=False)
        logging.info(f"[{group.upper()}] Saved provider features: {out_features_path} ({os.path.getsize(out_features_path)/(1024*1024):.2f} MB)")

        return feat_df


if __name__ == "__main__":
    extractor = ProviderFeatureExtractor()
    df_train_feats = extractor.extract_provider_features(group="train")
    df_test_feats = extractor.extract_provider_features(group="test")
    print("--> Provider Feature Engineering completed successfully!")
