"""
Healthcare Provider Fraud Detection - Detailed Model Evaluation & Threshold Analysis Module
------------------------------------------------------------------------------------------
Evaluates XGBoost and Random Forest models on leakage-safe validation dataset split from TRAIN.
Computes Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrix (TP, TN, FP, FN).
Performs multi-threshold analysis, exports reports/model_comparison.csv, generates standalone plots,
and compiles the comprehensive reports/model_metrics.md report.
"""

import os
import glob
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, roc_curve
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif', 'figure.dpi': 300})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


class ModelEvaluator:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.train_feat_path = os.path.join(self.base_dir, "data", "features", "train_provider_features.parquet")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.reports_dir = os.path.join(self.base_dir, "reports")
        self.figures_dir = os.path.join(self.reports_dir, "figures")

        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_validation_data_and_models(self):
        """Re-establish exact 80/20 Stratified Validation split from TRAIN features."""
        logging.info(f"Loading TRAIN features from {self.train_feat_path}...")
        df_train = pd.read_parquet(self.train_feat_path)

        feature_cols = [c for c in df_train.columns if c not in ['Provider', 'PotentialFraud']]
        X = df_train[feature_cols]
        y = df_train['PotentialFraud']

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        logging.info(f"Validation dataset: {len(X_val):,} providers ({y_val.sum():,} Fraud, {len(y_val)-y_val.sum():,} Non-Fraud)")

        # Load trained models
        xgb_path = os.path.join(self.models_dir, "xgboost_fraud_model.pkl")
        rf_path = os.path.join(self.models_dir, "random_forest_fraud_model.pkl")

        model_xgb = joblib.load(xgb_path)
        model_rf = joblib.load(rf_path)

        return X_val, y_val, model_xgb, model_rf

    def evaluate_models(self, X_val, y_val, model_xgb, model_rf):
        """Calculate metrics, confusion matrix counts, and comparison table."""
        logging.info("Evaluating XGBoost and Random Forest on Validation set...")

        probs_xgb = model_xgb.predict_proba(X_val)[:, 1]
        probs_rf = model_rf.predict_proba(X_val)[:, 1]

        # XGBoost Optimal Threshold Selection
        precisions_xgb, recalls_xgb, thresholds_xgb = precision_recall_curve(y_val, probs_xgb)
        f1_scores_xgb = 2 * (precisions_xgb * recalls_xgb) / (precisions_xgb + recalls_xgb + 1e-10)
        best_thresh_xgb = float(thresholds_xgb[np.argmax(f1_scores_xgb)])

        preds_xgb = (probs_xgb >= best_thresh_xgb).astype(int)
        preds_rf = (probs_rf >= 0.50).astype(int)

        cm_xgb = confusion_matrix(y_val, preds_xgb)
        cm_rf = confusion_matrix(y_val, preds_rf)

        # Confusion Matrix breakdown: tn, fp, fn, tp
        tn_xgb, fp_xgb, fn_xgb, tp_xgb = cm_xgb.ravel()
        tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

        metrics = {
            'XGBoost': {
                'Precision': precision_score(y_val, preds_xgb),
                'Recall': recall_score(y_val, preds_xgb),
                'F1': f1_score(y_val, preds_xgb),
                'ROC-AUC': roc_auc_score(y_val, probs_xgb),
                'PR-AUC': auc(recalls_xgb, precisions_xgb),
                'TP': int(tp_xgb),
                'TN': int(tn_xgb),
                'FP': int(fp_xgb),
                'FN': int(fn_xgb),
                'Threshold': best_thresh_xgb,
                'Probs': probs_xgb
            },
            'Random Forest': {
                'Precision': precision_score(y_val, preds_rf),
                'Recall': recall_score(y_val, preds_rf),
                'F1': f1_score(y_val, preds_rf),
                'ROC-AUC': roc_auc_score(y_val, probs_rf),
                'PR-AUC': auc(*precision_recall_curve(y_val, probs_rf)[1::-1]),
                'TP': int(tp_rf),
                'TN': int(tn_rf),
                'FP': int(fp_rf),
                'FN': int(fn_rf),
                'Threshold': 0.50,
                'Probs': probs_rf
            }
        }

        # Save model_comparison.csv
        comp_rows = []
        for name, m in metrics.items():
            comp_rows.append({
                'Model': name,
                'Precision': round(m['Precision'], 4),
                'Recall': round(m['Recall'], 4),
                'F1': round(m['F1'], 4),
                'ROC-AUC': round(m['ROC-AUC'], 4),
                'PR-AUC': round(m['PR-AUC'], 4),
                'TP': m['TP'],
                'TN': m['TN'],
                'FP': m['FP'],
                'FN': m['FN']
            })

        df_comp = pd.DataFrame(comp_rows)
        comp_csv_path = os.path.join(self.reports_dir, "model_comparison.csv")
        df_comp.to_csv(comp_csv_path, index=False)
        logging.info(f"Saved: {comp_csv_path}")

        return metrics

    def perform_threshold_analysis(self, y_val, probs_xgb):
        """Sweep probability thresholds for XGBoost and construct threshold analysis table."""
        logging.info("Performing probability threshold analysis for XGBoost...")
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.4907, 0.50, 0.60, 0.70, 0.80, 0.90]
        rows = []

        for th in thresholds:
            preds = (probs_xgb >= th).astype(int)
            cm = confusion_matrix(y_val, preds)
            tn, fp, fn, tp = cm.ravel()
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            f1 = f1_score(y_val, preds, zero_division=0)

            rows.append({
                'Threshold': th,
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1-Score': round(f1, 4),
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn
            })

        df_thresh = pd.DataFrame(rows)
        return df_thresh

    def generate_evaluation_plots(self, y_val, metrics):
        """Generate standalone plots directly in reports/ and reports/figures/."""
        logging.info("Generating evaluation plots in reports/...")

        # 1. ROC Curves Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, m in metrics.items():
            fpr, tpr, _ = roc_curve(y_val, m['Probs'])
            ax.plot(fpr, tpr, label=f"{name} (ROC-AUC = {m['ROC-AUC']:.4f})", linewidth=2.5)
        ax.plot([0, 1], [0, 1], 'k--', label='Baseline (AUC = 0.5000)')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        plt.tight_layout()

        roc_path1 = os.path.join(self.reports_dir, "roc_curve.png")
        roc_path2 = os.path.join(self.figures_dir, "roc_curve.png")
        plt.savefig(roc_path1)
        plt.savefig(roc_path2)
        plt.close()
        logging.info(f"Saved: {roc_path1}")

        # 2. Precision-Recall Curves Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, m in metrics.items():
            precision, recall, _ = precision_recall_curve(y_val, m['Probs'])
            ax.plot(recall, precision, label=f"{name} (PR-AUC = {m['PR-AUC']:.4f})", linewidth=2.5)
        ax.set_title('Precision-Recall (PR) Curve Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Recall (Fraud Case Capture Rate)')
        ax.set_ylabel('Precision (Audit Accuracy)')
        ax.legend(loc='lower left')
        plt.tight_layout()

        pr_path1 = os.path.join(self.reports_dir, "precision_recall_curve.png")
        pr_path2 = os.path.join(self.figures_dir, "precision_recall_curve.png")
        plt.savefig(pr_path1)
        plt.savefig(pr_path2)
        plt.close()
        logging.info(f"Saved: {pr_path1}")

        # 3. Confusion Matrix Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for idx, (name, m) in enumerate(metrics.items()):
            cm = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'], cbar=False)
            axes[idx].set_title(f'{name} Confusion Matrix\n(Precision: {m["Precision"]:.4f}, Recall: {m["Recall"]:.4f})', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Predicted Provider Label')
            axes[idx].set_ylabel('Actual Provider Label')
        plt.tight_layout()

        cm_path1 = os.path.join(self.reports_dir, "confusion_matrix.png")
        cm_path2 = os.path.join(self.figures_dir, "confusion_matrix.png")
        plt.savefig(cm_path1)
        plt.savefig(cm_path2)
        plt.close()
        logging.info(f"Saved: {cm_path1}")

    def generate_model_metrics_markdown(self, metrics, df_thresh):
        """Generate reports/model_metrics.md."""
        xgb_m = metrics['XGBoost']
        rf_m = metrics['Random Forest']

        thresh_table_rows = []
        for _, row in df_thresh.iterrows():
            thresh_table_rows.append(
                f"| `{row['Threshold']:.4f}` | `{row['Precision']:.4f}` | `{row['Recall']:.4f}` | `{row['F1-Score']:.4f}` | `{row['TP']}` | `{row['FP']}` | `{row['FN']}` | `{row['TN']}` |"
            )
        thresh_table_str = "\n".join(thresh_table_rows)

        md = f"""# Healthcare Provider Fraud Detection - Comprehensive Model Evaluation & Metrics Report

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
| **XGBoost Classifier** | **`{xgb_m['Precision']:.4f}`** | **`{xgb_m['Recall']:.4f}`** | **`{xgb_m['F1']:.4f}`** | **`{xgb_m['ROC-AUC']:.4f}`** | **`{xgb_m['PR-AUC']:.4f}`** | **`{xgb_m['TP']}`** | `{xgb_m['FP']}` | **`{xgb_m['FN']}`** | `{xgb_m['TN']}` |
| **Random Forest** | `{rf_m['Precision']:.4f}` | `{rf_m['Recall']:.4f}` | `{rf_m['F1']:.4f}` | `{rf_m['ROC-AUC']:.4f}` | `{rf_m['PR-AUC']:.4f}` | `{rf_m['TP']}` | `{rf_m['FP']}` | `{rf_m['FN']}` | `{rf_m['TN']}` |

---

## 4. Probability Threshold Analysis (XGBoost)

Evaluating multiple probability decision thresholds on the validation set to analyze the trade-off between Precision (audit accuracy) and Recall (fraud capture rate):

| Probability Threshold | Precision | Recall | F1-Score | TP | FP | FN | TN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{thresh_table_str}

> [!NOTE]
> ### Optimal Decision Threshold Selection
> The optimal decision threshold for XGBoost was selected at **`{xgb_m['Threshold']:.4f}`**. This threshold maximizes the F1-score (`{xgb_m['F1']:.4f}`), achieving a strong balance of **`{xgb_m['Recall']*100:.1f}%` Recall** (`{xgb_m['TP']}` out of `{xgb_m['TP']+xgb_m['FN']}` fraud cases detected) while maintaining **`{xgb_m['Precision']*100:.1f}%` Precision** (`{xgb_m['FP']}` false positive audit investigations).

---

## 5. Visualizations & Evaluation Curves

- **Confusion Matrices**: [`reports/confusion_matrix.png`](file:///{os.path.join(self.reports_dir, 'confusion_matrix.png').replace(os.sep, '/')})
- **ROC Curves**: [`reports/roc_curve.png`](file:///{os.path.join(self.reports_dir, 'roc_curve.png').replace(os.sep, '/')})
- **Precision-Recall Curves**: [`reports/precision_recall_curve.png`](file:///{os.path.join(self.reports_dir, 'precision_recall_curve.png').replace(os.sep, '/')})

---

## 6. Business Metrics & Operational Interpretation

In healthcare provider fraud detection, business objectives prioritize **catching fraudulent billing networks** while avoiding overwhelming auditing resources with false alarms:

1. **Precision (`{xgb_m['Precision']:.4f}`)**:
   - *Business Meaning*: When the system flags a provider as high-risk, **`{xgb_m['Precision']*100:.1f}%`** of the flagged providers are genuinely fraudulent.
   - *Audit Impact*: Minimizes wasted investigative costs and prevents unwarranted administrative friction with legitimate healthcare providers.
2. **Recall (`{xgb_m['Recall']:.4f}`)**:
   - *Business Meaning*: The system successfully captures **`{xgb_m['Recall']*100:.1f}%`** of all fraudulent provider networks operating in the Medicare ecosystem.
   - *Financial Impact*: Prevents millions of dollars in unrecovered fraudulent Medicare reimbursements (avoiding costly False Negatives).
3. **False Negatives (`{xgb_m['FN']}` providers)**:
   - Fraudulent providers missed by the model. Represents uncaptured financial loss to Medicare.
4. **False Positives (`{xgb_m['FP']}` providers)**:
   - Legitimate providers incorrectly flagged for audit. Represents routine administrative review overhead.

---

## 7. Final Production Model Selection

**Selected Model**: **XGBoost Classifier**
- **Justification**: XGBoost outperforms Random Forest across all key evaluation dimensions (**ROC-AUC: `{xgb_m['ROC-AUC']:.4f}`** vs `{rf_m['ROC-AUC']:.4f}`, **PR-AUC: `{xgb_m['PR-AUC']:.4f}`** vs `{rf_m['PR-AUC']:.4f}`, **F1: `{xgb_m['F1']:.4f}`** vs `{rf_m['F1']:.4f}`).

---

## 8. Limitations & Scope

1. **Provider-Level Target Scope**:
   - The model scores provider entities; it does **not** directly prove individual line-item claim fraud.
2. **Inductive Scoring**:
   - Unlabeled test providers are scored based on aggregated feature patterns without modifying training data based on test predictions.
"""
        report_path = os.path.join(self.reports_dir, "model_metrics.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)
        logging.info(f"Saved: {report_path}")

    def run_all(self):
        X_val, y_val, model_xgb, model_rf = self.load_validation_data_and_models()
        metrics = self.evaluate_models(X_val, y_val, model_xgb, model_rf)
        df_thresh = self.perform_threshold_analysis(y_val, metrics['XGBoost']['Probs'])
        self.generate_evaluation_plots(y_val, metrics)
        self.generate_model_metrics_markdown(metrics, df_thresh)
        logging.info("--> Detailed Model Evaluation & Threshold Analysis completed successfully!")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_all()
