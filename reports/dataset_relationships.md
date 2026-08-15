# Dataset Entity Relationships & Join Hazards Report

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
| **Unique Providers** | `5410` | `1353` | **Target Entity** | Primary prediction unit. 1 row per Provider in Target file. |
| **Unique Beneficiaries** | `138556` | `63968` | **Reference Entity** | 1 row per Beneficiary in Beneficiary CSV. |
| **Inpatient Claims** | `40474` | `9551` | **Fact / Event** | 1 row per inpatient claim line item. |
| **Outpatient Claims** | `517737` | `125841` | **Fact / Event** | 1 row per outpatient claim line item. |
| **Providers in Inpatient** | `2092` | N/A | Subgroup | Providers with active inpatient admissions. |
| **Providers in Outpatient** | `5012` | N/A | Subgroup | Providers with active outpatient visits. |

---

## Key Integrity & Unique Constraint Verification

1. **Beneficiary PK Integrity (`BeneID`)**:
   - TRAIN Beneficiary duplicates: `0` rows.
   - TEST Beneficiary duplicates: `0` rows.
   - **Conclusion**: `BeneID` is strictly unique in the Beneficiary tables (1 row per patient).

2. **Provider PK Integrity (`Provider`)**:
   - TRAIN Provider duplicates: `0` rows.
   - TEST Provider duplicates: `0` rows.
   - **Conclusion**: `Provider` is strictly unique in the Provider target files (1 row per provider).

---

## Data Leakage & Overlap Analysis Across TRAIN and TEST

- **Provider Overlap**: `0` providers (`0.0%`).
  - *Finding*: Provider IDs in TRAIN and TEST are completely disjoint (0 overlap). This confirms that provider-level risk scoring requires inductive generalization to unseen providers.
- **Beneficiary Overlap**: `54452` beneficiaries (`39.3%`).
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
