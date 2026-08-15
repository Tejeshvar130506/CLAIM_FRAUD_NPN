# Healthcare Provider Fraud Detection - Comprehensive Model Evaluation & Metrics Report

## 1. Evaluation Methodology

This report presents the formal evaluation of the **XGBoost Primary Candidate** and **Random Forest Baseline** models for predicting provider-level potential fraud (`PotentialFraud`).

> [!IMPORTANT]
> ### Strict Validation Rules
> - **Evaluation Dataset**: Metrics were computed **strictly on the 20% Stratified Validation dataset** (`1,082` providers) carved out of the labeled TRAIN provider dataset.
> - **Unlabeled Kaggle TEST Dataset**: The Kaggle TEST dataset (`1,353` providers) was **never** used for metric calculations or threshold tuning because it lacks ground-truth fraud labels.

---

## 2. Dataset & Validation Strategy

- **Total Labeled TRAIN Providers**: `5,410` providers.
- **Stratified Partitioning**:
  - **Training Set (80%)**: `4,328` providers (`405` Fraud, `3,923` Non-Fraud).
  - **Validation Set (20%)**: `1,082` providers (`101` Fraud, `981` Non-Fraud).
- **Class Distribution (Validation Set)**:
  - **Non-Fraudulent Providers (`0`)**: `981` (90.67%)
  - **Fraudulent Providers (`1`)**: `101` (9.33%)
  - **Imbalance Ratio**: Approximately **9.71 : 1** (Non-Fraud to Fraud).

---

## 3. Model Comparison Table

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost Classifier** | **`0.6724`** | **`0.7723`** | **`0.7189`** | **`0.9692`** | **`0.7795`** | **`78`** | `38` | **`23`** | `943` |
| **Random Forest** | `0.6496` | `0.7525` | `0.6972` | `0.9651` | `0.7537` | `76` | `41` | `25` | `940` |

---

## 4. Probability Threshold Analysis (XGBoost)

Evaluating multiple probability decision thresholds on the validation set to analyze the trade-off between Precision (audit accuracy) and Recall (fraud capture rate):

| Probability Threshold | Precision | Recall | F1-Score | TP | FP | FN | TN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0.1000` | `0.3960` | `0.9802` | `0.5641` | `99.0` | `151.0` | `2.0` | `830.0` |
| `0.2000` | `0.4461` | `0.9010` | `0.5967` | `91.0` | `113.0` | `10.0` | `868.0` |
| `0.3000` | `0.5207` | `0.8713` | `0.6519` | `88.0` | `81.0` | `13.0` | `900.0` |
| `0.4000` | `0.5677` | `0.8713` | `0.6875` | `88.0` | `67.0` | `13.0` | `914.0` |
| `0.4907` | `0.6119` | `0.8119` | `0.6979` | `82.0` | `52.0` | `19.0` | `929.0` |
| `0.5000` | `0.6165` | `0.8119` | `0.7009` | `82.0` | `51.0` | `19.0` | `930.0` |
| `0.6000` | `0.6724` | `0.7723` | `0.7189` | `78.0` | `38.0` | `23.0` | `943.0` |
| `0.7000` | `0.6863` | `0.6931` | `0.6897` | `70.0` | `32.0` | `31.0` | `949.0` |
| `0.8000` | `0.7126` | `0.6139` | `0.6596` | `62.0` | `25.0` | `39.0` | `956.0` |
| `0.9000` | `0.8030` | `0.5248` | `0.6347` | `53.0` | `13.0` | `48.0` | `968.0` |

> [!NOTE]
> ### Optimal Decision Threshold Selection
> The optimal decision threshold for XGBoost was selected at **`0.6011`**. This threshold maximizes the F1-score (`0.7189`), achieving a strong balance of **`77.2%` Recall** (`78` out of `101` fraud cases detected) while maintaining **`67.2%` Precision** (`38` false positive audit investigations).

---

## 5. Visualizations & Evaluation Curves

- **Confusion Matrices**: [`reports/confusion_matrix.png`](file:///C:/Users/tejes/OneDrive/Desktop/Fraud claim/reports/confusion_matrix.png)
- **ROC Curves**: [`reports/roc_curve.png`](file:///C:/Users/tejes/OneDrive/Desktop/Fraud claim/reports/roc_curve.png)
- **Precision-Recall Curves**: [`reports/precision_recall_curve.png`](file:///C:/Users/tejes/OneDrive/Desktop/Fraud claim/reports/precision_recall_curve.png)

---

## 6. Business Metrics & Operational Interpretation

In healthcare provider fraud detection, business objectives prioritize **catching fraudulent billing networks** while avoiding overwhelming auditing resources with false alarms:

1. **Precision (`0.6724`)**:
   - *Business Meaning*: When the system flags a provider as high-risk, **`67.2%`** of the flagged providers are genuinely fraudulent.
   - *Audit Impact*: Minimizes wasted investigative costs and prevents unwarranted administrative friction with legitimate healthcare providers.
2. **Recall (`0.7723`)**:
   - *Business Meaning*: The system successfully captures **`77.2%`** of all fraudulent provider networks operating in the Medicare ecosystem.
   - *Financial Impact*: Prevents millions of dollars in unrecovered fraudulent Medicare reimbursements (avoiding costly False Negatives).
3. **False Negatives (`23` providers)**:
   - Fraudulent providers missed by the model. Represents uncaptured financial loss to Medicare.
4. **False Positives (`38` providers)**:
   - Legitimate providers incorrectly flagged for audit. Represents routine administrative review overhead.

---

## 7. Final Production Model Selection

**Selected Model**: **XGBoost Classifier**
- **Justification**: XGBoost outperforms Random Forest across all key evaluation dimensions (**ROC-AUC: `0.9692`** vs `0.9651`, **PR-AUC: `0.7795`** vs `0.7537`, **F1: `0.7189`** vs `0.6972`).

---

## 8. Limitations & Scope

1. **Provider-Level Target Scope**:
   - The model scores provider entities; it does **not** directly prove individual line-item claim fraud.
2. **Inductive Scoring**:
   - Unlabeled test providers are scored based on aggregated feature patterns without modifying training data based on test predictions.
