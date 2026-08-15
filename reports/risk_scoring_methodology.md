# Provider Risk Scoring Methodology Report

## Executive Summary

This report documents the operational methodology, probability calibration, score normalization, and priority classification rules governing the **Provider Risk Scoring Engine**.

To support clinical compliance officers, medical auditors, and fraud investigators, the system converts continuous Machine Learning fraud probabilities into transparent, standardized **0–100 Risk Scores** and 4-tier operational **Risk Levels** (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).

---

## 1. System Architecture: Conceptual Separation

The risk scoring architecture maintains strict conceptual boundaries between four key layers:

```
 ┌────────────────────────┐
 │ 1. XGBoost ML Probability │  Continuous model output: P(Fraud) in [0.0, 1.0]
 └───────────┬────────────┘
             │ Formula: Risk Score = round(P * 100)
             ▼
 ┌────────────────────────┐
 │ 2. Provider Risk Score │  Standardized operational score: 0 to 100
 └───────────┬────────────┘
             │ Priority Threshold Mapping
             ▼
 ┌────────────────────────┐
 │ 3. Priority Risk Level │  Categorical Audit Priority: LOW, MEDIUM, HIGH, CRITICAL
 └───────────┬────────────┘
             │ Peer Deviation & Feature Attribution
             ▼
 ┌────────────────────────┐
 │ 4. Business Risk Flags │  Human-readable "Potential Contributing Factors"
 └────────────────────────┘
```

> [!NOTE]
> ### System Boundary Principles
> - **XGBoost Probability**: Reflects statistical likelihood based on learned provider feature patterns.
> - **Risk Score**: Standardizes probability to an easily interpretable integer metric (0–100) for UI display and thresholding.
> - **Risk Level**: Establishes actionable operational workflow triggers for audit teams.
> - **Potential Contributing Factors**: Highlight top behavioral deviations to assist human auditors during investigation.

---

## 2. Risk Score & Risk Level Justification Matrix

Risk levels are assigned based on calibrated validation probability distributions and operational audit capacity:

| Risk Level | Risk Score Range | Fraud Probability ($P$) | Operational Action & Audit Priority |
| :--- | :--- | :--- | :--- |
| **`LOW`** | **`0 – 30`** | $P < 0.30$ | **Baseline Monitoring**: Provider billing patterns fall within expected population norms. Standard automated claim processing. |
| **`MEDIUM`** | **`31 – 60`** | $0.30 \le P \le 0.60$ | **Elevated Monitoring**: Minor behavioral anomalies detected (e.g. slight financial or volume deviation). Scheduled for periodic review. |
| **`HIGH`** | **`61 – 85`** | $0.61 \le P \le 0.85$ | **Priority Investigation**: Significant behavioral anomalies (e.g. >2.0x peer reimbursement or high repeat beneficiary ratio). Flagged for compliance audit. |
| **`CRITICAL`** | **`86 – 100`** | $P > 0.85$ | **Urgent Audit & Payment Hold**: Severe anomalous billing patterns. Immediate administrative payment pause and full forensic audit. |

---

## 3. Potential Contributing Factors Identification

When a provider exhibits an elevated Risk Level (`MEDIUM`, `HIGH`, `CRITICAL`), the engine compares the provider's behavioral metrics against state peer averages and population medians to extract human-readable **potential contributing factors**:

- **Financial Deviations**: `"Claim reimbursement amount significantly above peer level (3.2x state peer average)"`
- **Volume Anomalies**: `"Unusually high claim frequency (95.4 claims per active month)"`
- **Inpatient Care Anomalies**: `"High inpatient ratio (65.2% inpatient admissions)"`
- **Patient Network Anomalies**: `"High repeat-beneficiary ratio (42.1% repeat patient claims)"`
- **Concentration Anomalies**: `"High beneficiary claim concentration (HHI: 0.284)"`
- **Physician Patterns**: `"High attending & operating physician match ratio (82.5%)"`

---

## 4. Reusable API Function Usage (`get_provider_risk`)

```python
from src.risk_scoring import get_provider_risk

# Example Lookup for Provider PRV51069
risk_profile = get_provider_risk('PRV51069')

print(risk_profile)
# Output:
# {
#     "provider_id": "PRV51069",
#     "fraud_probability": 0.9572,
#     "risk_score": 96,
#     "risk_level": "CRITICAL",
#     "top_potential_contributing_factors": [
#         "Total claim billing volume significantly above peer level (3.0x state peer average)",
#         "High repeat-beneficiary ratio (61.7% repeat patient claims)"
#     ],
#     "important_behavioral_metrics": { ... }
# }
```
