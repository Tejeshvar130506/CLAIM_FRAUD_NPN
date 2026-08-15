# Healthcare Provider Fraud Pattern & Coordinated Behavioral Analysis Report

## Executive Overview

This report presents the findings of the Exploratory Data Analysis (EDA) and pattern identification conducted on the Kaggle Healthcare Provider Fraud Detection dataset. The analysis evaluated provider billing behaviors, financial distributions, beneficiary concentration, state peer group deviations, and **potential coordinated fraud behavior**.

---

## 1. Key Behavioral Differentiators: Fraud vs Non-Fraud Providers

| Metric / Feature | Non-Fraudulent Providers | Fraudulent Providers | Behavioral Divergence |
| :--- | :--- | :--- | :--- |
| **Mean Total Claims** | `70.4` claims | `420.5` claims | Fraudulent providers submit **~3.4x higher claim volume**. |
| **Mean Total Reimbursement ($)** | `$53,193.72` | `$584,350.04` | Fraudulent providers request **~5.8x higher total reimbursements**. |
| **Average Claim vs State Peer Ratio** | `0.87x` | `2.23x` | Fraudulent providers bill significantly higher per claim than local state peers. |
| **Inpatient Ratio** | `0.13` | `0.33` | Fraudulent providers exhibit elevated inpatient admission ratios. |

---

## 2. Analysis of Potential Coordinated Fraud Behavior

> [!WARNING]
> ### Observed Behavioral Patterns
> 1. **High Repeat Beneficiary & Concentration Clusters**:
>    - Fraudulent providers demonstrate elevated `repeat_beneficiary_ratio` combined with high `top_bene_claim_share`. A small subset of beneficiaries accounts for disproportionately high claim volumes at fraudulent provider facilities.
> 2. **Physician Assignment Patterns**:
>    - Fraudulent providers exhibit higher `same_attending_operating_ratio` (attending physician billing as operating physician simultaneously) and lower attending physician diversity per claim.
> 3. **Peer Group Deviation Anomalies**:
>    - Providers exhibiting `peer_claim_volume_zscore > 3.0` and `average_claim_vs_peer_average > 2.5` represent high-risk clusters exhibiting **potential coordinated fraud behavior**.

---

## 3. Summary of Visualizations Generated

- [`reports/figures/01_target_and_claim_volume.png`](file:///C:/Users/tejes/OneDrive/Desktop/Fraud claim/reports/figures/01_target_and_claim_volume.png)
- [`reports/figures/02_financial_distributions.png`](file:///C:/Users/tejes/OneDrive/Desktop/Fraud claim/reports/figures/02_financial_distributions.png)
- [`reports/figures/03_inpatient_vs_outpatient.png`](file:///C:/Users/tejes/OneDrive/Desktop/Fraud claim/reports/figures/03_inpatient_vs_outpatient.png)
- [`reports/figures/04_peer_deviation_analysis.png`](file:///C:/Users/tejes/OneDrive/Desktop/Fraud claim/reports/figures/04_peer_deviation_analysis.png)
- [`reports/figures/05_coordinated_fraud_patterns.png`](file:///C:/Users/tejes/OneDrive/Desktop/Fraud claim/reports/figures/05_coordinated_fraud_patterns.png)
