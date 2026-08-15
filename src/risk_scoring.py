"""
Provider Risk Scoring & Reusable Lookup Engine
------------------------------------------------
Computes transparent Risk Scores (0-100), Risk Levels (LOW, MEDIUM, HIGH, CRITICAL),
top potential contributing factors, and important provider behavioral metrics.
Exposes the reusable lookup API function `get_provider_risk(provider_id)`.
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Global Cache for Models & Feature Matrices
_MODEL_CACHE = {}


def _load_resources(base_dir=None):
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if 'xgb_model' not in _MODEL_CACHE:
        xgb_path = os.path.join(base_dir, "models", "final_fraud_model.pkl")
        if not os.path.exists(xgb_path):
            xgb_path = os.path.join(base_dir, "models", "xgboost_fraud_model.pkl")
        
        ebm_path = os.path.join(base_dir, "models", "ebm_model.pkl")
        
        train_feat_path = os.path.join(base_dir, "data", "features", "train_provider_features.parquet")
        test_feat_path = os.path.join(base_dir, "data", "features", "test_provider_features.parquet")

        logging.info(f"Loading final model from {xgb_path}...")
        _MODEL_CACHE['xgb_model'] = joblib.load(xgb_path)

        if os.path.exists(ebm_path):
            _MODEL_CACHE['ebm_model'] = joblib.load(ebm_path)
        else:
            _MODEL_CACHE['ebm_model'] = None

        df_train = pd.read_parquet(train_feat_path)
        df_test = pd.read_parquet(test_feat_path)

        # Combine datasets for lookup
        if 'PotentialFraud' in df_train.columns:
            df_train_no_target = df_train.drop(columns=['PotentialFraud'])
        else:
            df_train_no_target = df_train

        combined_df = pd.concat([df_train_no_target, df_test], ignore_index=True).drop_duplicates(subset=['Provider'])
        _MODEL_CACHE['combined_features'] = combined_df.set_index('Provider')
        _MODEL_CACHE['train_medians'] = df_train.select_dtypes(include=[np.number]).median()

    return _MODEL_CACHE


def calculate_risk_level(risk_score: int) -> str:
    """
    Categorizes Risk Score (0-100) into justified risk priority levels:
    - 0  to 30 : LOW       (Baseline routine monitoring)
    - 31 to 60 : MEDIUM    (Elevated risk requiring periodic review)
    - 61 to 85 : HIGH      (High risk requiring priority audit investigation)
    - 86 to 100: CRITICAL  (Urgent risk requiring immediate payment hold & audit)
    """
    if risk_score <= 30:
        return "LOW"
    elif risk_score <= 60:
        return "MEDIUM"
    elif risk_score <= 85:
        return "HIGH"
    else:
        return "CRITICAL"


def get_provider_risk(provider_id: str, base_dir=None) -> dict:
    """
    Reusable lookup function for Provider Risk Scoring & Explanation.
    Returns:
    - provider_id
    - fraud_probability
    - risk_score (0-100)
    - risk_level (LOW, MEDIUM, HIGH, CRITICAL)
    - top_potential_contributing_factors
    - important_behavioral_metrics
    """
    res = _load_resources(base_dir=base_dir)
    df_lookup = res['combined_features']
    model = res['xgb_model']

    if provider_id not in df_lookup.index:
        raise KeyError(f"Provider ID '{provider_id}' not found in provider feature repository.")

    # Extract feature vector
    row = df_lookup.loc[[provider_id]]
    feature_cols = [c for c in row.columns if c not in ['Provider', 'PotentialFraud']]
    X_prov = row[feature_cols]

    # Model Probability Prediction
    fraud_prob = float(model.predict_proba(X_prov)[0, 1])
    risk_score = int(round(fraud_prob * 100.0))
    risk_level = calculate_risk_level(risk_score)

    # Extract behavioral metrics
    row_dict = row.iloc[0].to_dict()
    
    tot_claims = int(row_dict.get('total_claims', 0))
    tot_amt = float(row_dict.get('total_claim_amount', 0.0))
    avg_amt = float(row_dict.get('average_claim_amount', 0.0))
    inp_ratio = float(row_dict.get('inpatient_ratio', 0.0))
    repeat_ratio = float(row_dict.get('repeat_beneficiary_ratio', 0.0))
    peer_avg_ratio = float(row_dict.get('average_claim_vs_peer_average', 1.0))
    peer_amt_ratio = float(row_dict.get('claim_amount_vs_peer_average', 1.0))
    bene_hhi = float(row_dict.get('beneficiary_hhi_concentration', 0.0))
    same_phys_ratio = float(row_dict.get('same_attending_operating_ratio', 0.0))
    active_month_claims = float(row_dict.get('claims_per_month', 0.0))

    important_metrics = {
        "total_claims": tot_claims,
        "total_claim_amount": round(tot_amt, 2),
        "average_claim_amount": round(avg_amt, 2),
        "inpatient_ratio": round(inp_ratio, 4),
        "repeat_beneficiary_ratio": round(repeat_ratio, 4),
        "average_claim_vs_peer_average": round(peer_avg_ratio, 2),
        "claim_amount_vs_peer_average": round(peer_amt_ratio, 2),
        "beneficiary_hhi_concentration": round(bene_hhi, 4),
        "same_attending_operating_ratio": round(same_phys_ratio, 4),
        "claims_per_month": round(active_month_claims, 2)
    }

    # Generate Top Potential Contributing Factors (Exact Wording)
    factors = []
    
    if peer_avg_ratio > 1.5:
        factors.append(f"Claim amount significantly above peer level ({peer_avg_ratio:.1f}x state peer average)")
    elif peer_amt_ratio > 2.0:
        factors.append(f"Total claim billing volume significantly above peer level ({peer_amt_ratio:.1f}x state peer average)")
        
    if active_month_claims > 80.0:
        factors.append(f"Unusually high claim frequency ({active_month_claims:.1f} claims per active month)")

    if inp_ratio > 0.35:
        factors.append(f"High inpatient ratio ({inp_ratio*100.0:.1f}% inpatient admissions)")

    if repeat_ratio > 0.30:
        factors.append(f"High repeat-beneficiary ratio ({repeat_ratio*100.0:.1f}% repeat patient claims)")

    if bene_hhi > 0.15:
        factors.append(f"High beneficiary claim concentration (HHI: {bene_hhi:.3f})")

    if same_phys_ratio > 0.50:
        factors.append(f"High attending & operating physician match ratio ({same_phys_ratio*100.0:.1f}%)")

    if not factors:
        factors.append("Behavioral metrics within standard population peer norms")

    return {
        "provider_id": provider_id,
        "fraud_probability": round(fraud_prob, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "top_potential_contributing_factors": factors[:4],
        "important_behavioral_metrics": important_metrics
    }


if __name__ == "__main__":
    # Test execution for a high-risk and low-risk provider
    test_high = get_provider_risk('PRV51069')
    print("=== HIGH RISK PROVIDER LOOKUP ===")
    print(f"Provider ID: {test_high['provider_id']}")
    print(f"Fraud Probability: {test_high['fraud_probability']*100:.1f}%")
    print(f"Risk Score: {test_high['risk_score']}/100")
    print(f"Risk Level: {test_high['risk_level']}")
    print("Potential contributing factors:")
    for factor in test_high['top_potential_contributing_factors']:
        print(f"  * {factor}")

    test_low = get_provider_risk('PRV51002')
    print("\n=== LOW RISK PROVIDER LOOKUP ===")
    print(f"Provider ID: {test_low['provider_id']}")
    print(f"Fraud Probability: {test_low['fraud_probability']*100:.1f}%")
    print(f"Risk Score: {test_low['risk_score']}/100")
    print(f"Risk Level: {test_low['risk_level']}")
    print("Potential contributing factors:")
    for factor in test_low['top_potential_contributing_factors']:
        print(f"  * {factor}")
