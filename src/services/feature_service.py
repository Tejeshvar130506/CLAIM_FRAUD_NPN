"""
Feature Engineering & Transformation Service
--------------------------------------------
Provides reproducible, leakage-free provider-level behavioral feature extraction for:
1. Batch uploaded claims datasets (Inpatient, Outpatient, Beneficiary integration)
2. Interactive single-provider input forms
3. Precomputed feature vector lookups
Guarantees exact alignment with model training schema (52 feature columns).
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.config import FEATURES_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
from src.features import ProviderFeatureExtractor

logger = logging.getLogger(__name__)

# Canonical 52 feature column schema required by trained models
MODEL_FEATURE_COLUMNS = [
    'total_claims', 'total_claim_amount', 'average_claim_amount',
    'maximum_claim_amount', 'std_claim_amount', 'total_deductible_paid',
    'average_deductible_paid', 'maximum_deductible_paid',
    'inpatient_claim_count', 'outpatient_claim_count', 'inpatient_ratio',
    'average_length_of_stay', 'max_inpatient_stay_duration',
    'mean_claim_duration', 'max_claim_duration', 'std_claim_duration',
    'unique_attending_physicians', 'unique_operating_physicians',
    'unique_other_physicians', 'same_attending_operating_ratio',
    'unique_admit_diagnosis_codes', 'unique_group_diagnosis_codes',
    'mean_num_diagnosis_codes', 'mean_num_procedure_codes', 'mean_patient_age',
    'std_patient_age', 'deceased_patient_count', 'deceased_patient_ratio',
    'chronic_cond_score_mean', 'renal_disease_ratio', 'gender_male_ratio',
    'gender_female_ratio', 'unique_states_served', 'unique_counties_served',
    'outpatient_ratio', 'reimbursement_to_deductible_ratio',
    'physician_to_claim_ratio', 'unique_diagnosis_count',
    'unique_procedure_count', 'repeat_beneficiary_count',
    'unique_beneficiaries', 'top_bene_claim_share',
    'beneficiary_hhi_concentration', 'repeat_beneficiary_ratio', 'active_days',
    'claims_per_month', 'claims_per_week', 'claim_frequency', 'primary_state',
    'average_claim_vs_peer_average', 'claim_amount_vs_peer_average',
    'peer_claim_volume_zscore'
]


class FeatureEngineeringService:
    """
    Service for extracting, normalizing, and formatting provider feature vectors.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.extractor = ProviderFeatureExtractor(base_dir=base_dir)
        self._feature_cache: Optional[pd.DataFrame] = None
        self._state_peer_stats: Optional[Dict[str, Any]] = None

    def get_canonical_feature_columns(self) -> List[str]:
        """Returns the list of 52 canonical feature column names in exact order."""
        return list(MODEL_FEATURE_COLUMNS)

    def extract_from_integrated_claims(self, df_claims: pd.DataFrame, group_name: str = "INFERENCE") -> pd.DataFrame:
        """
        Aggregates claim-level records to Provider level with peer benchmarking and concentrations.
        """
        logger.info(f"Extracting provider features from {len(df_claims):,} claims...")
        
        # Ensure temporary columns
        df = df_claims.copy()
        
        diag_cols = [c for c in df.columns if c.startswith('ClmDiagnosisCode_')]
        proc_cols = [c for c in df.columns if c.startswith('ClmProcedureCode_')]
        
        if 'NumDiagnosisCodes' not in df.columns:
            df['NumDiagnosisCodes'] = df[diag_cols].notnull().sum(axis=1) if diag_cols else 0
        if 'NumProcedureCodes' not in df.columns:
            df['NumProcedureCodes'] = df[proc_cols].notnull().sum(axis=1) if proc_cols else 0

        if 'IsInpatient' not in df.columns:
            df['IsInpatient'] = 0
        if 'InpatientStayDuration' not in df.columns:
            df['InpatientStayDuration'] = 0.0
        if 'ClaimDuration' not in df.columns:
            df['ClaimDuration'] = 1.0
        if 'PatientAgeAtClaim' not in df.columns:
            df['PatientAgeAtClaim'] = 70.0
        if 'IsDeceased' not in df.columns:
            df['IsDeceased'] = 0
        if 'ChronicCond_Count' not in df.columns:
            df['ChronicCond_Count'] = 3.0
        if 'RenalDiseaseIndicator' not in df.columns:
            df['RenalDiseaseIndicator'] = 0
        if 'Gender' not in df.columns:
            df['Gender'] = 1
        if 'State' not in df.columns:
            df['State'] = 0
        if 'County' not in df.columns:
            df['County'] = 0
        if 'DeductibleAmtPaid' not in df.columns:
            df['DeductibleAmtPaid'] = 0.0
        if 'InscClaimAmtReimbursed' not in df.columns:
            df['InscClaimAmtReimbursed'] = 0.0
        if 'AttendingPhysician' not in df.columns:
            df['AttendingPhysician'] = None
        if 'OperatingPhysician' not in df.columns:
            df['OperatingPhysician'] = None
        if 'OtherPhysician' not in df.columns:
            df['OtherPhysician'] = None
        if 'ClmAdmitDiagnosisCode' not in df.columns:
            df['ClmAdmitDiagnosisCode'] = None
        if 'DiagnosisGroupCode' not in df.columns:
            df['DiagnosisGroupCode'] = None

        if 'SameAttendingOperating' not in df.columns:
            df['SameAttendingOperating'] = (
                df['AttendingPhysician'].notnull() &
                df['OperatingPhysician'].notnull() &
                (df['AttendingPhysician'] == df['OperatingPhysician'])
            ).astype(int)

        prov_state = df.groupby('Provider')['State'].agg(
            lambda x: x.mode()[0] if not x.mode().empty else (x.iloc[0] if len(x) > 0 else 0)
        ).reset_index(name='primary_state')

        prov_grp = df.groupby('Provider')

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

        # Code diversity
        diag_cols = [c for c in df.columns if c.startswith('ClmDiagnosisCode_')]
        proc_cols = [c for c in df.columns if c.startswith('ClmProcedureCode_')]

        def count_unique_codes(df_group, code_cols):
            if not code_cols:
                return 0
            codes = df_group[code_cols].values.ravel()
            codes = codes[pd.notnull(codes)]
            return len(np.unique(codes))

        unique_diag_series = prov_grp.apply(lambda g: count_unique_codes(g, diag_cols), include_groups=False)
        unique_proc_series = prov_grp.apply(lambda g: count_unique_codes(g, proc_cols), include_groups=False)

        feat_df['unique_diagnosis_count'] = feat_df['Provider'].map(unique_diag_series).fillna(0)
        feat_df['unique_procedure_count'] = feat_df['Provider'].map(unique_proc_series).fillna(0)

        # Beneficiary concentration & temporal frequencies
        bene_conc_df = self.extractor.compute_beneficiary_repeat_and_concentration(df)
        temp_freq_df = self.extractor.compute_temporal_frequencies(df)

        feat_df = feat_df.merge(bene_conc_df, on='Provider', how='left')
        feat_df = feat_df.merge(temp_freq_df, on='Provider', how='left')
        feat_df = feat_df.merge(prov_state, on='Provider', how='left')

        # Peer Group Benchmarks
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

        feat_df = feat_df.drop(columns=['state_peer_avg_claim', 'state_peer_total_claim', 'state_peer_claims', 'state_peer_std_claims'])
        feat_df = feat_df.fillna(0.0)

        # Enforce canonical schema order
        return self._align_to_model_schema(feat_df)

    def build_feature_vector_from_dict(self, input_dict: Dict[str, Any], provider_id: str = "CUSTOM_PROV") -> pd.DataFrame:
        """
        Builds a single-provider 1-row DataFrame aligned with canonical feature schema from form parameters.
        """
        tot_claims = float(input_dict.get('total_claims', 50))
        tot_amt = float(input_dict.get('total_claim_amount', 45000.0))
        avg_amt = tot_amt / max(1.0, tot_claims)
        inp_cnt = float(input_dict.get('inpatient_claim_count', 5))
        out_cnt = float(input_dict.get('outpatient_claim_count', max(0, tot_claims - inp_cnt)))
        inp_ratio = float(input_dict.get('inpatient_ratio', inp_cnt / max(1.0, tot_claims)))
        out_ratio = 1.0 - inp_ratio

        avg_los = float(input_dict.get('average_length_of_stay', 4.5 if inp_ratio > 0 else 0.0))
        max_los = float(input_dict.get('max_inpatient_stay_duration', avg_los * 1.5 if inp_ratio > 0 else 0.0))
        tot_deduct = float(input_dict.get('total_deductible_paid', tot_amt * 0.08))
        avg_deduct = tot_deduct / max(1.0, tot_claims)

        repeat_ratio = float(input_dict.get('repeat_beneficiary_ratio', 0.25))
        uniq_benes = float(input_dict.get('unique_beneficiaries', max(1, int(tot_claims * (1.0 - repeat_ratio)))))
        repeat_cnt = float(max(0, tot_claims - uniq_benes))
        hhi = float(input_dict.get('beneficiary_hhi_concentration', 0.04))
        same_phys_ratio = float(input_dict.get('same_attending_operating_ratio', 0.15))
        claims_per_month = float(input_dict.get('claims_per_month', 25.0))
        peer_avg_ratio = float(input_dict.get('average_claim_vs_peer_average', 1.0))
        peer_amt_ratio = float(input_dict.get('claim_amount_vs_peer_average', 1.0))

        row_data = {
            'Provider': provider_id,
            'total_claims': tot_claims,
            'total_claim_amount': tot_amt,
            'average_claim_amount': avg_amt,
            'maximum_claim_amount': float(input_dict.get('maximum_claim_amount', avg_amt * 2.5)),
            'std_claim_amount': float(input_dict.get('std_claim_amount', avg_amt * 0.4)),
            'total_deductible_paid': tot_deduct,
            'average_deductible_paid': avg_deduct,
            'maximum_deductible_paid': float(input_dict.get('maximum_deductible_paid', avg_deduct * 2.0)),
            'inpatient_claim_count': inp_cnt,
            'outpatient_claim_count': out_cnt,
            'inpatient_ratio': inp_ratio,
            'average_length_of_stay': avg_los,
            'max_inpatient_stay_duration': max_los,
            'mean_claim_duration': float(input_dict.get('mean_claim_duration', avg_los if inp_ratio > 0 else 1.2)),
            'max_claim_duration': float(input_dict.get('max_claim_duration', max_los if inp_ratio > 0 else 3.0)),
            'std_claim_duration': float(input_dict.get('std_claim_duration', 1.5)),
            'unique_attending_physicians': float(input_dict.get('unique_attending_physicians', 5)),
            'unique_operating_physicians': float(input_dict.get('unique_operating_physicians', 3)),
            'unique_other_physicians': float(input_dict.get('unique_other_physicians', 2)),
            'same_attending_operating_ratio': same_phys_ratio,
            'unique_admit_diagnosis_codes': float(input_dict.get('unique_admit_diagnosis_codes', 8)),
            'unique_group_diagnosis_codes': float(input_dict.get('unique_group_diagnosis_codes', 4)),
            'mean_num_diagnosis_codes': float(input_dict.get('mean_num_diagnosis_codes', 3.2)),
            'mean_num_procedure_codes': float(input_dict.get('mean_num_procedure_codes', 0.8)),
            'mean_patient_age': float(input_dict.get('mean_patient_age', 71.5)),
            'std_patient_age': float(input_dict.get('std_patient_age', 8.2)),
            'deceased_patient_count': float(input_dict.get('deceased_patient_count', 0)),
            'deceased_patient_ratio': float(input_dict.get('deceased_patient_ratio', 0.0)),
            'chronic_cond_score_mean': float(input_dict.get('chronic_cond_score_mean', 3.5)),
            'renal_disease_ratio': float(input_dict.get('renal_disease_ratio', 0.18)),
            'gender_male_ratio': float(input_dict.get('gender_male_ratio', 0.45)),
            'gender_female_ratio': float(input_dict.get('gender_female_ratio', 0.55)),
            'unique_states_served': float(input_dict.get('unique_states_served', 1)),
            'unique_counties_served': float(input_dict.get('unique_counties_served', 2)),
            'outpatient_ratio': out_ratio,
            'reimbursement_to_deductible_ratio': tot_amt / (tot_deduct + 1.0),
            'physician_to_claim_ratio': float(input_dict.get('unique_attending_physicians', 5)) / max(1.0, tot_claims),
            'unique_diagnosis_count': float(input_dict.get('unique_diagnosis_count', 18)),
            'unique_procedure_count': float(input_dict.get('unique_procedure_count', 6)),
            'repeat_beneficiary_count': repeat_cnt,
            'unique_beneficiaries': uniq_benes,
            'top_bene_claim_share': float(input_dict.get('top_bene_claim_share', 0.12)),
            'beneficiary_hhi_concentration': hhi,
            'repeat_beneficiary_ratio': repeat_ratio,
            'active_days': float(input_dict.get('active_days', 365)),
            'claims_per_month': claims_per_month,
            'claims_per_week': claims_per_month / 4.34,
            'claim_frequency': tot_claims / max(1.0, float(input_dict.get('active_days', 365))),
            'primary_state': float(input_dict.get('primary_state', 39)),
            'average_claim_vs_peer_average': peer_avg_ratio,
            'claim_amount_vs_peer_average': peer_amt_ratio,
            'peer_claim_volume_zscore': float(input_dict.get('peer_claim_volume_zscore', 0.0))
        }

        df_single = pd.DataFrame([row_data])
        return self._align_to_model_schema(df_single)

    def _align_to_model_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures exact column presence and ordering as required by models.
        """
        df_out = df.copy()
        for col in MODEL_FEATURE_COLUMNS:
            if col not in df_out.columns:
                df_out[col] = 0.0

        ordered_cols = ['Provider'] + MODEL_FEATURE_COLUMNS if 'Provider' in df_out.columns else MODEL_FEATURE_COLUMNS
        return df_out[ordered_cols]
