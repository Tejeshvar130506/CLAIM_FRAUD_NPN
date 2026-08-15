# Healthcare Provider Fraud Detection - Model Evaluation & Risk Scoring Report

## Executive Summary

This report documents the development, validation, performance comparison, retraining, and test inference for the **Healthcare Provider Fraud Detection System**.

The target prediction entity is **PROVIDER-LEVEL** `PotentialFraud`. Supervised learning was conducted strictly on the **5,410 labeled TRAIN providers**. Unlabeled Kaggle TEST providers (**1,353 providers**) were set aside for final risk score inference and were **never** exposed during model validation or tuning.

---

## 1. Dataset & Splitting Strategy

- **Supervised Training Set**: `5,410` unique provider feature vectors (`data/features/train_provider_features.parquet`).
- **Class Distribution**:
  - Non-Fraudulent Providers (`0`): `4,904` (90.65%)
  - Fraudulent Providers (`1`): `506` (9.35%)
  - Class Imbalance Ratio: **~9.69 : 1** (Non-Fraud to Fraud).
- **Validation Partitioning**:
  - 80/20 Stratified Train/Validation split (`4,328` Train providers, `1,082` Validation providers).
  - Class imbalance handled via XGBoost `scale_pos_weight = 9.69` and Random Forest `class_weight='balanced'`.

---

## 2. Model Performance Comparison

| Evaluation Metric | XGBoost Classifier (Primary Candidate) | Random Forest Classifier (Baseline) | Performance Divergence |
| :--- | :--- | :--- | :--- |
| **ROC-AUC Score** | **`0.9692`** | `0.9651` | XGBoost demonstrates superior ranking performance (+0.0041 ROC-AUC). |
| **PR-AUC Score** | **`0.7795`** | `0.7537` | XGBoost maintains higher precision under high recall settings. |
| **Precision** | `0.6724` | `0.6496` | Precision at optimal decision threshold. |
| **Recall** | `0.7723` | `0.7525` | High recall captures a vast majority of fraudulent provider networks. |
| **F1-Score** | **`0.7189`** | `0.6972` | Balanced harmonic mean of Precision and Recall. |
| **Decision Threshold** | `0.6011` | `0.5000` | Optimal threshold tuned on validation PR curve. |

---

## 3. Confusion Matrix Breakdown (Validation Set: 1,082 Providers)

### XGBoost Classifier
- **True Negatives (TN)**: `943`
- **False Positives (FP)**: `38`
- **False Negatives (FN)**: `23`
- **True Positives (TP)**: `78`

### Random Forest Classifier
- **True Negatives (TN)**: `940`
- **False Positives (FP)**: `41`
- **False Negatives (FN)**: `25`
- **True Positives (TP)**: `76`

---

## 4. Final Selected Model & Full Retraining

- **Selected Model**: **XGBoost Classifier** was selected as the final production candidate based on superior ROC-AUC (`0.9692`), PR-AUC (`0.7795`), and F1-score.
- **Full Retraining**: Retrained on 100% of labeled TRAIN provider data (`5,410` providers).
- **Saved Model Artifacts**:
  - `models/xgboost_fraud_model.pkl`
  - `models/random_forest_fraud_model.pkl`
  - `models/final_fraud_model.pkl`

---

## 5. Kaggle Test Dataset Inference Results

- **Total Test Providers Evaluated**: `1,353`
- **Predicted High-Risk Fraudulent Providers**: `158` (11.68%)
- **Outputs Saved**:
  - Parquet dataset: [`data/processed/test_provider_predictions.parquet`](file:///C:/Users/tejes/OneDrive/Desktop/Fraud claim/data/processed/test_provider_predictions.parquet)
  - CSV dataset: [`reports/test_provider_predictions.csv`](file:///C:/Users/tejes/OneDrive/Desktop/Fraud claim/reports/test_provider_predictions.csv)

---

## 6. Model Limitations & Operational Scope Disclaimers

> [!IMPORTANT]
> ### System Scope & Operational Disclaimer
> 1. **Provider Risk Prediction Target**:
>    - This machine learning model predicts **PROVIDER-LEVEL** risk of potential fraud.
> 2. **Not Individual Claim Fraud Proof**:
>    - The model does **NOT** directly prove that an individual medical bill or claim line item is fraudulent.
> 3. **Future Billing Risk Assessment**:
>    - The future medical bill upload workflow will evaluate uploaded claims in context with the provider's overall risk score and claim-level anomaly indicators to generate comprehensive risk assessments.
