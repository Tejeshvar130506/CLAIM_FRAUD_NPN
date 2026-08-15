"""
Healthcare Provider Risk Scoring - Advanced Feature Engineering Module
------------------------------------------------------------------------
Aggregates claim-level integrated datasets to the Provider primary key level.
Constructs exact requested provider-level risk scoring features, peer benchmarking metrics,
repeat beneficiary concentration, and temporal claim frequencies.
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

    def compute_beneficiary_repeat_and_concentration(self, df_claims):
        """Compute repeat beneficiary counts, ratios, top beneficiary share, and HHI concentration index."""
        logging.info("Computing beneficiary repeat counts and HHI concentration per provider...")
        prov_bene_counts = df_claims.groupby(['Provider', 'BeneID']).size().reset_index(name='bene_claim_cnt')
        
        # Total claims per provider from this table
        prov_totals = prov_bene_counts.groupby('Provider')['bene_claim_cnt'].sum().reset_index(name='total_prov_claims')
        
        # Merge total claims back to compute share per beneficiary
        prov_bene_counts = prov_bene_counts.merge(prov_totals, on='Provider')
        prov_bene_counts['bene_share'] = prov_bene_counts['bene_claim_cnt'] / prov_bene_counts['total_prov_claims']
        prov_bene_counts['bene_share_sq'] = prov_bene_counts['bene_share'] ** 2

        # Aggregations per provider
        repeat_df = prov_bene_counts.groupby('Provider').agg(
            repeat_beneficiary_count=('bene_claim_cnt', lambda x: (x > 1).sum()),
            unique_beneficiaries=('BeneID', 'nunique'),
            top_bene_claim_share=('bene_share', 'max'),
            beneficiary_hhi_concentration=('bene_share_sq', 'sum')
        ).reset_index()

        repeat_df['repeat_beneficiary_ratio'] = (
            repeat_df['repeat_beneficiary_count'] / np.maximum(1, repeat_df['unique_beneficiaries'])
        )
        return repeat_df

    def compute_temporal_frequencies(self, df_claims):
        """Compute active claim span, claims per month, claims per week, and claim frequency."""
        logging.info("Computing temporal claim frequencies and active date spans...")
        dates_df = df_claims.groupby('Provider').agg(
            min_claim_dt=('ClaimStartDt', 'min'),
            max_claim_dt=('ClaimStartDt', 'max'),
            total_claims=('ClaimID', 'count')
        ).reset_index()

        dates_df['active_days'] = (dates_df['max_claim_dt'] - dates_df['min_claim_dt']).dt.days + 1
        dates_df['active_months'] = np.maximum(1.0, dates_df['active_days'] / 30.4375)
        dates_df['active_weeks'] = np.maximum(1.0, dates_df['active_days'] / 7.0)

        dates_df['claims_per_month'] = dates_df['total_claims'] / dates_df['active_months']
        dates_df['claims_per_week'] = dates_df['total_claims'] / dates_df['active_weeks']
        dates_df['claim_frequency'] = dates_df['total_claims'] / dates_df['active_days']

        return dates_df[['Provider', 'active_days', 'claims_per_month', 'claims_per_week', 'claim_frequency']]

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

        # Primary state of provider (mode state)
        prov_state = df_claims.groupby('Provider')['State'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else (x.iloc[0] if len(x) > 0 else 0)
        ).reset_index(name='primary_state')

        # Core GroupBy Aggregations
        prov_grp = df_claims.groupby('Provider')
        
        agg_dict = {
            'ClaimID': ['count'],
            'InscClaimAmtReimbursed': ['sum', 'mean', 'max', 'std'],
            'DeductibleAmtPaid': ['sum', 'mean', 'max'],
            'IsInpatient': ['sum', lambda x: (x == 0).sum(), 'mean'],
            'InpatientStayDuration': ['mean', 'max'],
            'ClaimDuration': ['mean', 'max', 'std'],
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

        feat_df = prov_grp.agg(agg_dict)

        feat_df.columns = [
            'total_claims',
            'total_claim_amount', 'average_claim_amount', 'maximum_claim_amount', 'std_claim_amount',
            'total_deductible_paid', 'average_deductible_paid', 'maximum_deductible_paid',
            'inpatient_claim_count', 'outpatient_claim_count', 'inpatient_ratio',
            'average_length_of_stay', 'max_inpatient_stay_duration',
            'mean_claim_duration', 'max_claim_duration', 'std_claim_duration',
            'unique_attending_physicians', 'unique_operating_physicians', 'unique_other_physicians',
            'same_attending_operating_ratio',
            'unique_admit_diagnosis_codes', 'unique_group_diagnosis_codes',
            'mean_num_diagnosis_codes', 'mean_num_procedure_codes',
            'mean_patient_age', 'std_patient_age',
            'deceased_patient_count', 'deceased_patient_ratio',
            'chronic_cond_score_mean',
            'renal_disease_ratio',
            'gender_male_ratio', 'gender_female_ratio',
            'unique_states_served', 'unique_counties_served'
        ]

        feat_df = feat_df.reset_index()
        feat_df['outpatient_ratio'] = 1.0 - feat_df['inpatient_ratio']
        feat_df['reimbursement_to_deductible_ratio'] = feat_df['total_claim_amount'] / (feat_df['total_deductible_paid'] + 1.0)
        feat_df['physician_to_claim_ratio'] = feat_df['unique_attending_physicians'] / feat_df['total_claims']

        # Code diversity across all 10 diagnosis slots and 6 procedure slots
        diag_cols = [c for c in df_claims.columns if c.startswith('ClmDiagnosisCode_')]
        proc_cols = [c for c in df_claims.columns if c.startswith('ClmProcedureCode_')]

        def count_unique_codes(df_group, code_cols):
            codes = df_group[code_cols].values.ravel()
            codes = codes[pd.notnull(codes)]
            return len(np.unique(codes))

        logging.info(f"[{group.upper()}] Computing unique diagnosis & procedure code diversity per provider...")
        unique_diag_series = prov_grp.apply(lambda g: count_unique_codes(g, diag_cols), include_groups=False)
        unique_proc_series = prov_grp.apply(lambda g: count_unique_codes(g, proc_cols), include_groups=False)

        feat_df['unique_diagnosis_count'] = feat_df['Provider'].map(unique_diag_series)
        feat_df['unique_procedure_count'] = feat_df['Provider'].map(unique_proc_series)

        # Merge Beneficiary concentration and temporal features
        bene_conc_df = self.compute_beneficiary_repeat_and_concentration(df_claims)
        temp_freq_df = self.compute_temporal_frequencies(df_claims)

        feat_df = feat_df.merge(bene_conc_df, on='Provider', how='left')
        feat_df = feat_df.merge(temp_freq_df, on='Provider', how='left')
        feat_df = feat_df.merge(prov_state, on='Provider', how='left')

        # Peer Group Benchmarking (State Level Peer Benchmarks)
        logging.info(f"[{group.upper()}] Computing State-level peer group benchmarks...")
        state_peer_avg = feat_df.groupby('primary_state').agg(
            state_peer_avg_claim=('average_claim_amount', 'mean'),
            state_peer_total_claim=('total_claim_amount', 'mean'),
            state_peer_claims=('total_claims', 'mean'),
            state_peer_std_claims=('total_claims', 'std')
        ).reset_index()

        feat_df = feat_df.merge(state_peer_avg, on='primary_state', how='left')

        feat_df['average_claim_vs_peer_average'] = feat_df['average_claim_amount'] / (feat_df['state_peer_avg_claim'] + 1e-5)
        feat_df['claim_amount_vs_peer_average'] = feat_df['total_claim_amount'] / (feat_df['state_peer_total_claim'] + 1e-5)
        feat_df['peer_claim_volume_zscore'] = (
            (feat_df['total_claims'] - feat_df['state_peer_claims']) / (feat_df['state_peer_std_claims'].fillna(1.0) + 1e-5)
        )

        # Drop temporary state peer columns used in calculation
        feat_df = feat_df.drop(columns=['state_peer_avg_claim', 'state_peer_total_claim', 'state_peer_claims', 'state_peer_std_claims'])

        # Fill remaining NaNs with 0.0
        feat_df = feat_df.fillna(0.0)

        logging.info(f"[{group.upper()}] Extracted {feat_df.shape[1]} provider features for {feat_df.shape[0]:,} unique Providers.")

        # Target Labeling for TRAIN dataset ONLY
        if group == "train":
            prov_target_path = glob.glob(os.path.join(self.raw_train_dir, "Train-*.csv"))[0]
            df_target = pd.read_csv(prov_target_path)
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
    print("--> Advanced Provider Feature Engineering completed successfully!")
