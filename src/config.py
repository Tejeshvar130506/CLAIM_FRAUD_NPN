"""
Centralized Configuration Module
--------------------------------
Provides typed, validated application settings, directory paths, risk score thresholds,
database paths, and LLM configuration settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Base paths
BASE_DIR = Path(os.getenv("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))))
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_DIR = DATA_DIR / "db"

# Ensure runtime directories exist
for path in [DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR, MODELS_DIR, REPORTS_DIR, DATABASE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Database Configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", str(DATABASE_DIR / "fraud_intelligence.db"))
DATABASE_TIMEOUT_SECONDS = int(os.getenv("DATABASE_TIMEOUT_SECONDS", 10))

# Application Info
APP_TITLE = "Multi-Agent Healthcare Provider Fraud Intelligence Platform"
APP_VERSION = "2.0.0"
APP_ENV = os.getenv("APP_ENV", "development")

# Security / Authentication
SECRET_KEY = os.getenv("SECRET_KEY", "anti-fraud-platform-secret-key-2026")
PASSWORD_SALT = os.getenv("PASSWORD_SALT", "healthcare-audit-salt")
SESSION_EXPIRY_HOURS = int(os.getenv("SESSION_EXPIRY_HOURS", 12))

# Risk Score Thresholds (0 - 100)
RISK_THRESHOLD_LOW_MAX = 30
RISK_THRESHOLD_MEDIUM_MAX = 60
RISK_THRESHOLD_HIGH_MAX = 85
# 86-100 is CRITICAL

# Model Configuration
DEFAULT_XGB_MODEL_PATH = str(MODELS_DIR / "final_fraud_model.pkl")
DEFAULT_EBM_MODEL_PATH = str(MODELS_DIR / "ebm_model.pkl")
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_CLASSIFICATION_THRESHOLD", 0.50))

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash")
ENABLE_LLM_REASONING = bool(os.getenv("ENABLE_LLM_REASONING", True))
