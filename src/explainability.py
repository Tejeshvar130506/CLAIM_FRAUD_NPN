"""
Explainable Boosting Machine (EBM) Training & Interpretability Module
----------------------------------------------------------------------
Trains an Explainable Boosting Machine (EBM) glass-box classifier on provider-level risk features.
Provides interpretable surrogate feature attributions for global and local provider risk explanations.
Serializes model artifact to models/ebm_model.pkl.
"""

import os
import glob
import logging
import joblib
import pandas as pd
import numpy as np

from interpret.glassbox import ExplainableBoostingClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


class EBMExplainerPipeline:
    def __init__(self, base_dir=None):
        if base_dir is None:
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        else:
            self.base_dir = os.path.abspath(base_dir)

        self.train_feat_path = os.path.join(self.base_dir, "data", "features", "train_provider_features.parquet")
        self.models_dir = os.path.join(self.base_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.ebm_model = None

    def train_ebm_model(self):
        """Train Explainable Boosting Machine on provider features."""
        logging.info(f"Loading provider features from {self.train_feat_path}...")
        df_train = pd.read_parquet(self.train_feat_path)

        feature_cols = [c for c in df_train.columns if c not in ['Provider', 'PotentialFraud']]
        X = df_train[feature_cols]
        y = df_train['PotentialFraud']

        logging.info(f"Training Explainable Boosting Classifier (EBM) over {len(X):,} providers and {len(feature_cols)} features...")
        ebm = ExplainableBoostingClassifier(
            random_state=42,
            n_jobs=-1
        )
        ebm.fit(X, y)
        self.ebm_model = ebm

        # Serialize EBM model
        ebm_save_path = os.path.join(self.models_dir, "ebm_model.pkl")
        joblib.dump(ebm, ebm_save_path)
        logging.info(f"Saved EBM Model: {ebm_save_path}")

        return ebm


if __name__ == "__main__":
    pipeline = EBMExplainerPipeline()
    pipeline.train_ebm_model()
    print("--> EBM Model Training & Serialization completed successfully!")
