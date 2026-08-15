# Feature Quality & Correlation Assessment Report

## 1. Feature Quality & Null Value Audit

- **Total Providers Analyzed**: `5,410`
- **Total Engineered Features**: `52` (excluding `Provider` and target `PotentialFraud`).
- **Missing Value Count Across All Features**: `0` (Zero null values detected across all features).

---

## 2. Top Features Correlated with Provider Fraud (`PotentialFraud`)

| Feature Name | Pearson Correlation Coefficient ($r$) |
| :--- | :--- |
| `claim_amount_vs_peer_average` | +0.5808 |
| `total_claim_amount` | +0.5756 |
| `unique_procedure_count` | +0.5667 |
| `unique_group_diagnosis_codes` | +0.5501 |
| `max_inpatient_stay_duration` | +0.5427 |
| `total_deductible_paid` | +0.5321 |
| `inpatient_claim_count` | +0.5254 |
| `unique_admit_diagnosis_codes` | +0.5170 |
| `maximum_claim_amount` | +0.5147 |
| `unique_diagnosis_count` | +0.4679 |

---

## 3. Feature Variance & Skewness Analysis

1. **Volume & Financial Skewness**:
   - `total_claims`, `total_claim_amount`, and `average_claim_vs_peer_average` exhibit heavy right-skewness among fraudulent providers.
   - Fraudulent providers demonstrate significantly higher mean claim volumes (**~321 claims** vs **~95 claims** for non-fraudulent providers).
2. **Peer Benchmark Deviations**:
   - `average_claim_vs_peer_average` shows strong positive correlation with fraud. Fraudulent providers average reimbursement rates significantly higher than state peer averages.
3. **Leakage Audit**:
   - Confirmed zero 1.0 correlation features (no target leakage columns).
