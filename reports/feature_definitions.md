# Provider-Level Feature Definitions & Schema Reference

This document provides the formal data dictionary and operational definitions for all 44 provider-level risk scoring features engineered in Phase 3.

---

## Provider Feature Inventory

| Feature Name | Data Type | Feature Category | Description & Rationale |
| :--- | :--- | :--- | :--- |
| `Provider` | `string` | Primary Key | Unique provider identifier. |
| `total_claims` | `int64` | Volume | Total count of claims billed by provider across inpatient & outpatient visits. |
| `total_claim_amount` | `float64` | Financial | Cumulative insurance reimbursement amount ($) requested by provider. |
| `average_claim_amount` | `float64` | Financial | Mean insurance reimbursement amount ($) per claim. |
| `maximum_claim_amount` | `float64` | Financial | Maximum single reimbursement amount ($) billed by provider. |
| `std_claim_amount` | `float64` | Financial | Standard deviation of claim reimbursement amounts ($). |
| `total_deductible_paid` | `float64` | Financial | Cumulative patient deductible paid ($). |
| `average_deductible_paid` | `float64` | Financial | Mean patient deductible paid ($) per claim. |
| `maximum_deductible_paid` | `float64` | Financial | Maximum patient deductible paid ($). |
| `inpatient_claim_count` | `int64` | Subgroup | Number of inpatient admission claims. |
| `outpatient_claim_count` | `int64` | Subgroup | Number of outpatient visit claims. |
| `inpatient_ratio` | `float64` | Subgroup Ratio | Proportion of total claims originating from inpatient admissions (`inpatient_claim_count / total_claims`). |
| `outpatient_ratio` | `float64` | Subgroup Ratio | Proportion of total claims originating from outpatient visits (`outpatient_claim_count / total_claims`). |
| `average_length_of_stay` | `float64` | Clinical Duration | Mean inpatient stay duration in days (`DischargeDt - AdmissionDt + 1`). |
| `max_inpatient_stay_duration` | `float64` | Clinical Duration | Maximum inpatient stay duration in days billed by provider. |
| `mean_claim_duration` | `float64` | Claim Duration | Mean claim active window duration in days (`ClaimEndDt - ClaimStartDt + 1`). |
| `max_claim_duration` | `float64` | Claim Duration | Maximum active claim window duration in days. |
| `std_claim_duration` | `float64` | Claim Duration | Variance/std of claim durations. |
| `unique_beneficiaries` | `int64` | Patient Network | Count of distinct beneficiaries (`BeneID`) served by provider. |
| `repeat_beneficiary_count` | `int64` | Behavioral | Count of unique beneficiaries who submitted >1 claim with this provider. |
| `repeat_beneficiary_ratio` | `float64` | Behavioral | Proportion of provider's patients who are repeat claimants (`repeat_beneficiary_count / unique_beneficiaries`). |
| `top_bene_claim_share` | `float64` | Concentration | Maximum proportion of provider's total claims coming from a single beneficiary. |
| `beneficiary_hhi_concentration` | `float64` | Concentration | Herfindahl-Hirschman Concentration Index of claim volume across beneficiaries. |
| `active_days` | `int64` | Temporal | Total calendar days between provider's earliest and latest claim start dates. |
| `claims_per_month` | `float64` | Temporal Velocity | Claim submission rate per month (`total_claims / (active_days / 30.44)`). |
| `claims_per_week` | `float64` | Temporal Velocity | Claim submission rate per week (`total_claims / (active_days / 7.0)`). |
| `claim_frequency` | `float64` | Temporal Velocity | Daily claim submission frequency (`total_claims / active_days`). |
| `unique_attending_physicians` | `int64` | Physician Network | Count of distinct attending physicians linked to provider. |
| `unique_operating_physicians` | `int64` | Physician Network | Count of distinct operating physicians linked to provider. |
| `unique_other_physicians` | `int64` | Physician Network | Count of distinct other physicians linked to provider. |
| `same_attending_operating_ratio` | `float64` | Physician Pattern | Proportion of claims where attending physician is also operating physician. |
| `physician_to_claim_ratio` | `float64` | Physician Ratio | Attending physician diversity ratio per claim. |
| `unique_admit_diagnosis_codes` | `int64` | Clinical Diversity | Distinct admit diagnosis codes (`ClmAdmitDiagnosisCode`). |
| `unique_group_diagnosis_codes` | `int64` | Clinical Diversity | Distinct DRG group diagnosis codes (`DiagnosisGroupCode`). |
| `unique_diagnosis_count` | `int64` | Clinical Diversity | Total distinct diagnosis codes across all 10 claim diagnosis slots. |
| `unique_procedure_count` | `int64` | Clinical Diversity | Total distinct procedure codes across all 6 claim procedure slots. |
| `mean_num_diagnosis_codes` | `float64` | Clinical Complexity | Average non-null diagnosis codes per claim. |
| `mean_num_procedure_codes` | `float64` | Clinical Complexity | Average non-null procedure codes per claim. |
| `mean_patient_age` | `float64` | Patient Risk | Mean patient age at claim start date. |
| `std_patient_age` | `float64` | Patient Risk | Standard deviation of patient ages. |
| `deceased_patient_count` | `int64` | Patient Mortality | Count of claims billed for deceased beneficiaries. |
| `deceased_patient_ratio` | `float64` | Patient Mortality | Ratio of claims billed for deceased beneficiaries. |
| `chronic_cond_score_mean` | `float64` | Patient Health Risk | Mean patient chronic condition score (0 to 11). |
| `renal_disease_ratio` | `float64` | Patient Health Risk | Ratio of patients with End-Stage Renal Disease. |
| `primary_state` | `int64` | Geographical | Primary state location code of provider. |
| `average_claim_vs_peer_average` | `float64` | Peer Benchmark | Ratio of provider average claim amount to state peer group average claim amount. |
| `claim_amount_vs_peer_average` | `float64` | Peer Benchmark | Ratio of provider total claim amount to state peer group average total claim amount. |
| `peer_claim_volume_zscore` | `float64` | Peer Benchmark | Standardized Z-Score of provider claim volume relative to state peers. |
| `PotentialFraud` | `int64` | Target Label | Provider fraud label (`1` = Potential Fraud, `0` = Non-Fraud). Present **only** in TRAIN. |
