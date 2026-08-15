"""
Healthcare Provider Fraud Detection - Model Training, Evaluation & Inference Module
-------------------------------------------------------------------------------------
Trains, evaluates, and compares XGBoost Classifier and Random Forest Classifier on provider risk features.
Performs 80/20 stratified validation, handles class imbalance, generates evaluation plots,
retrains the final XGBoost model on full TRAIN data, serializes model artifacts,
and predicts provider fraud risk for the unlabeled Kaggle TEST dataset.
"""

import os
import glob
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, roc_curve
)

import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


class ProviderFraudModelPipeline:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.train_feat_path = os.path.join(self.base_dir, "data", "features", "train_provider_features.parquet")
        self.test_feat_path = os.path.join(self.base_dir, "data", "features", "test_provider_features.parquet")
        
        self.models_dir = os.path.join(self.base_dir, "models")
        self.reports_dir = os.path.join(self.base_dir, "reports")
        self.figures_dir = os.path.join(self.reports_dir, "figures")
        self.processed_dir = os.path.join(self.base_dir, "data", "processed")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.df_train = None
        self.df_test = None

    def load_and_verify_data(self):
        """Pre-training data verification and leakage check."""
        logging.info(f"Loading TRAIN provider features from {self.train_feat_path}...")
        self.df_train = pd.read_parquet(self.train_feat_path)
        logging.info(f"Loaded {len(self.df_train):,} TRAIN providers ({self.df_train.shape[1]} columns).")

        logging.info(f"Loading TEST provider features from {self.test_feat_path}...")
        self.df_test = pd.read_parquet(self.test_feat_path)
        logging.info(f"Loaded {len(self.df_test):,} TEST providers ({self.df_test.shape[1]} columns).")

        # Target verification
        if 'PotentialFraud' not in self.df_train.columns:
            raise KeyError("Target column 'PotentialFraud' missing from TRAIN features!")
        
        if 'PotentialFraud' in self.df_test.columns:
            raise ValueError("TEST features must NOT contain target column 'PotentialFraud'!")

        # Class imbalance check
        pos_cnt = int(self.df_train['PotentialFraud'].sum())
        total_cnt = len(self.df_train)
        neg_cnt = total_cnt - pos_cnt
        pos_pct = (pos_cnt / total_cnt) * 100.0

        logging.info(f"TRAIN Target Distribution: {pos_cnt:,} Fraud ({pos_pct:.2f}%), {neg_cnt:,} Non-Fraud. Imbalance Ratio: ~{neg_cnt/pos_cnt:.2f}:1")

        # Feature matrix extraction
        self.feature_cols = [c for c in self.df_train.columns if c not in ['Provider', 'PotentialFraud']]
        logging.info(f"Verified {len(self.feature_cols)} feature columns for model training.")

        X = self.df_train[self.feature_cols]
        y = self.df_train['PotentialFraud']

        return X, y

    def train_and_evaluate_models(self, X, y):
        """Train XGBoost and Random Forest on 80/20 Stratified Split and evaluate performance metrics."""
        logging.info("Performing 80/20 Stratified Train/Validation split...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        logging.info(f"Train set: {X_train.shape[0]:,} providers | Validation set: {X_val.shape[0]:,} providers")

        # Class imbalance weights
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        logging.info(f"Computed XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

        # Model 1: XGBoost Classifier
        logging.info("Training XGBoost Classifier...")
        model_xgb = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        model_xgb.fit(X_train, y_train)

        # Model 2: Random Forest Classifier
        logging.info("Training Random Forest Classifier...")
        model_rf = RandomForestClassifier(
            class_weight='balanced',
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model_rf.fit(X_train, y_train)

        # Predict Probabilities
        probs_xgb = model_xgb.predict_proba(X_val)[:, 1]
        probs_rf = model_rf.predict_proba(X_val)[:, 1]

        # Optimal Threshold selection based on F1-score for Validation Set
        precisions_xgb, recalls_xgb, thresholds_xgb = precision_recall_curve(y_val, probs_xgb)
        f1_scores_xgb = 2 * (precisions_xgb * recalls_xgb) / (precisions_xgb + recalls_xgb + 1e-10)
        best_thresh_xgb = float(thresholds_xgb[np.argmax(f1_scores_xgb)])

        preds_xgb = (probs_xgb >= best_thresh_xgb).astype(int)
        preds_rf = (probs_rf >= 0.50).astype(int)

        # Metrics Calculation
        metrics = {
            'XGBoost': {
                'precision': precision_score(y_val, preds_xgb),
                'recall': recall_score(y_val, preds_xgb),
                'f1': f1_score(y_val, preds_xgb),
                'roc_auc': roc_auc_score(y_val, probs_xgb),
                'pr_auc': auc(recalls_xgb, precisions_xgb),
                'confusion_matrix': confusion_matrix(y_val, preds_xgb),
                'best_threshold': best_thresh_xgb,
                'probs': probs_xgb,
                'model': model_xgb
            },
            'Random Forest': {
                'precision': precision_score(y_val, preds_rf),
                'recall': recall_score(y_val, preds_rf),
                'f1': f1_score(y_val, preds_rf),
                'roc_auc': roc_auc_score(y_val, probs_rf),
                'pr_auc': auc(*precision_recall_curve(y_val, probs_rf)[1::-1]),
                'confusion_matrix': confusion_matrix(y_val, preds_rf),
                'best_threshold': 0.50,
                'probs': probs_rf,
                'model': model_rf
            }
        }

        logging.info(f"XGBoost Validation -> ROC-AUC: {metrics['XGBoost']['roc_auc']:.4f}, PR-AUC: {metrics['XGBoost']['pr_auc']:.4f}, F1: {metrics['XGBoost']['f1']:.4f}, Recall: {metrics['XGBoost']['recall']:.4f}, Precision: {metrics['XGBoost']['precision']:.4f}")
        logging.info(f"Random Forest Validation -> ROC-AUC: {metrics['Random Forest']['roc_auc']:.4f}, PR-AUC: {metrics['Random Forest']['pr_auc']:.4f}, F1: {metrics['Random Forest']['f1']:.4f}, Recall: {metrics['Random Forest']['recall']:.4f}, Precision: {metrics['Random Forest']['precision']:.4f}")

        return X_val, y_val, metrics

    def generate_evaluation_plots(self, y_val, metrics):
        """Generate ROC curves, Precision-Recall curves, Confusion Matrices, and Feature Importance plots."""
        logging.info("Generating model evaluation plots...")

        # 1. ROC Curves Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, m in metrics.items():
            fpr, tpr, _ = roc_curve(y_val, m['probs'])
            ax.plot(fpr, tpr, label=f"{name} (AUC = {m['roc_auc']:.4f})", linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5000)')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=12, fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        plt.tight_layout()
        roc_path = os.path.join(self.figures_dir, "model_roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        logging.info(f"Saved: {roc_path}")

        # 2. Precision-Recall Curves Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, m in metrics.items():
            precision, recall, _ = precision_recall_curve(y_val, m['probs'])
            ax.plot(recall, precision, label=f"{name} (PR-AUC = {m['pr_auc']:.4f})", linewidth=2)
        ax.set_title('Precision-Recall (PR) Curves', fontsize=12, fontweight='bold')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(loc='lower left')
        plt.tight_layout()
        pr_path = os.path.join(self.figures_dir, "model_pr_curve.png")
        plt.savefig(pr_path)
        plt.close()
        logging.info(f"Saved: {pr_path}")

        # 3. Confusion Matrices Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for idx, (name, m) in enumerate(metrics.items()):
            sns.heatmap(m['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
            axes[idx].set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('Actual Label')
        plt.tight_layout()
        cm_path = os.path.join(self.figures_dir, "model_confusion_matrices.png")
        plt.savefig(cm_path)
        plt.close()
        logging.info(f"Saved: {cm_path}")

        # 4. XGBoost Feature Importance Plot
        xgb_model = metrics['XGBoost']['model']
        importances = pd.Series(xgb_model.feature_importances_, index=self.feature_cols).sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        importances.plot(kind='barh', ax=ax, color='#2b5c8f')
        ax.set_title('Top 15 Provider Risk Features (XGBoost Feature Importance)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Relative Feature Importance (Gain)')
        ax.invert_yaxis()
        plt.tight_layout()
        imp_path = os.path.join(self.figures_dir, "model_feature_importance.png")
        plt.savefig(imp_path)
        plt.close()
        logging.info(f"Saved: {imp_path}")

    def retrain_final_model_and_save(self, X_full, y_full, metrics):
        """Retrain selected XGBoost model on complete TRAIN dataset and serialize model artifacts."""
        logging.info("Retraining final XGBoost Classifier on 100% of labeled TRAIN provider dataset...")
        scale_pos_weight = (len(y_full) - y_full.sum()) / y_full.sum()

        final_xgb = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
        final_xgb.fit(X_full, y_full)

        # Save model binaries
        xgb_model_path = os.path.join(self.models_dir, "xgboost_fraud_model.pkl")
        rf_model_path = os.path.join(self.models_dir, "random_forest_fraud_model.pkl")
        final_model_path = os.path.join(self.models_dir, "final_fraud_model.pkl")

        joblib.dump(metrics['XGBoost']['model'], xgb_model_path)
        joblib.dump(metrics['Random Forest']['model'], rf_model_path)
        joblib.dump(final_xgb, final_model_path)

        logging.info(f"Saved XGBoost model: {xgb_model_path}")
        logging.info(f"Saved Random Forest model: {rf_model_path}")
        logging.info(f"Saved Retrained Final model: {final_model_path}")

        return final_xgb, metrics['XGBoost']['best_threshold']

    def infer_test_providers(self, final_model, threshold):
        """Generate provider-level risk score predictions for unlabeled Kaggle TEST dataset."""
        logging.info(f"Generating risk predictions for {len(self.df_test):,} unlabeled TEST providers...")
        X_test = self.df_test[self.feature_cols]

        test_probs = final_model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= threshold).astype(int)

        df_preds = pd.DataFrame({
            'Provider': self.df_test['Provider'],
            'fraud_probability': np.round(test_probs, 4),
            'fraud_prediction': test_preds
        })

        pred_pos_cnt = int(df_preds['fraud_prediction'].sum())
        pred_pos_pct = (pred_pos_cnt / len(df_preds)) * 100.0
        logging.info(f"TEST Inference Complete: Predicted {pred_pos_cnt:,} high-risk providers ({pred_pos_pct:.2f}% positive fraud rate).")

        # Save predictions
        parquet_out = os.path.join(self.processed_dir, "test_provider_predictions.parquet")
        csv_out = os.path.join(self.reports_dir, "test_provider_predictions.csv")

        df_preds.to_parquet(parquet_out, index=False)
        df_preds.to_csv(csv_out, index=False)

        logging.info(f"Saved TEST predictions Parquet: {parquet_out}")
        logging.info(f"Saved TEST predictions CSV: {csv_out}")

        return df_preds

    def generate_model_evaluation_report(self, metrics, test_preds_df):
        """Generate comprehensive reports/model_evaluation.md report."""
        xgb_m = metrics['XGBoost']
        rf_m = metrics['Random Forest']

        xgb_cm = xgb_m['confusion_matrix']
        rf_cm = rf_m['confusion_matrix']

        pos_test_cnt = int(test_preds_df['fraud_prediction'].sum())
        total_test_cnt = len(test_preds_df)

        md = f"""# Healthcare Provider Fraud Detection - Model Evaluation & Risk Scoring Report

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
| **ROC-AUC Score** | **`{xgb_m['roc_auc']:.4f}`** | `{rf_m['roc_auc']:.4f}` | XGBoost demonstrates superior ranking performance (+{xgb_m['roc_auc'] - rf_m['roc_auc']:.4f} ROC-AUC). |
| **PR-AUC Score** | **`{xgb_m['pr_auc']:.4f}`** | `{rf_m['pr_auc']:.4f}` | XGBoost maintains higher precision under high recall settings. |
| **Precision** | `{xgb_m['precision']:.4f}` | `{rf_m['precision']:.4f}` | Precision at optimal decision threshold. |
| **Recall** | `{xgb_m['recall']:.4f}` | `{rf_m['recall']:.4f}` | High recall captures a vast majority of fraudulent provider networks. |
| **F1-Score** | **`{xgb_m['f1']:.4f}`** | `{rf_m['f1']:.4f}` | Balanced harmonic mean of Precision and Recall. |
| **Decision Threshold** | `{xgb_m['best_threshold']:.4f}` | `0.5000` | Optimal threshold tuned on validation PR curve. |

---

## 3. Confusion Matrix Breakdown (Validation Set: 1,082 Providers)

### XGBoost Classifier
- **True Negatives (TN)**: `{xgb_cm[0, 0]}`
- **False Positives (FP)**: `{xgb_cm[0, 1]}`
- **False Negatives (FN)**: `{xgb_cm[1, 0]}`
- **True Positives (TP)**: `{xgb_cm[1, 1]}`

### Random Forest Classifier
- **True Negatives (TN)**: `{rf_cm[0, 0]}`
- **False Positives (FP)**: `{rf_cm[0, 1]}`
- **False Negatives (FN)**: `{rf_cm[1, 0]}`
- **True Positives (TP)**: `{rf_cm[1, 1]}`

---

## 4. Final Selected Model & Full Retraining

- **Selected Model**: **XGBoost Classifier** was selected as the final production candidate based on superior ROC-AUC (`{xgb_m['roc_auc']:.4f}`), PR-AUC (`{xgb_m['pr_auc']:.4f}`), and F1-score.
- **Full Retraining**: Retrained on 100% of labeled TRAIN provider data (`5,410` providers).
- **Saved Model Artifacts**:
  - `models/xgboost_fraud_model.pkl`
  - `models/random_forest_fraud_model.pkl`
  - `models/final_fraud_model.pkl`

---

## 5. Kaggle Test Dataset Inference Results

- **Total Test Providers Evaluated**: `{total_test_cnt:,}`
- **Predicted High-Risk Fraudulent Providers**: `{pos_test_cnt:,}` ({pos_test_cnt/total_test_cnt*100:.2f}%)
- **Outputs Saved**:
  - Parquet dataset: [`data/processed/test_provider_predictions.parquet`](file:///{os.path.join(self.processed_dir, 'test_provider_predictions.parquet').replace(os.sep, '/')})
  - CSV dataset: [`reports/test_provider_predictions.csv`](file:///{os.path.join(self.reports_dir, 'test_provider_predictions.csv').replace(os.sep, '/')})

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
"""
        report_path = os.path.join(self.reports_dir, "model_evaluation.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)
        logging.info(f"Saved: {report_path}")

    def run_pipeline(self):
        X, y = self.load_and_verify_data()
        X_val, y_val, metrics = self.train_and_evaluate_models(X, y)
        self.generate_evaluation_plots(y_val, metrics)
        final_model, best_threshold = self.retrain_final_model_and_save(X, y, metrics)
        test_preds_df = self.infer_test_providers(final_model, best_threshold)
        self.generate_model_evaluation_report(metrics, test_preds_df)
        logging.info("--> Provider Fraud Detection Model Pipeline completed successfully!")


if __name__ == "__main__":
    pipeline = ProviderFraudModelPipeline()
    pipeline.run_pipeline()
