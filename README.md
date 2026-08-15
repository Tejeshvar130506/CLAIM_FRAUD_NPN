# AI Healthcare Claims Fraud Detection & Provider Risk Scoring System

An end-to-end Machine Learning and Provider Risk Assessment system for detecting healthcare claims fraud and scoring provider risk based on Medicare inpatient, outpatient, and beneficiary datasets.

## Project Structure

```
claims-fraud-detection/
├── data/
│   ├── raw/
│   │   ├── train/        # Raw Kaggle Train CSV files
│   │   └── test/         # Raw Kaggle Test CSV files (unlabeled for final inference)
│   ├── processed/        # Parquet processed data storage
│   ├── features/         # Engineered feature sets
│   └── sample/           # Representative dataset samples
├── models/               # Model artifacts & checkpoints
├── reports/              # Data discovery & assessment reports
│   ├── dataset_assessment.md
│   ├── data_dictionary.csv
│   └── dataset_relationships.md
├── notebooks/            # Exploratory analysis notebooks
├── src/                  # Core Python modules & pipelines
│   └── data_discovery.py # Automated data discovery module
├── backend/              # API microservices (FastAPI)
├── frontend/             # Dashboard application
├── tests/                # Automated pytest test suites
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
└── docker-compose.yml    # Multi-container orchestration
```

## Quick Start

### 1. Requirements Setup
```bash
pip install -r requirements.txt
```

### 2. Run Data Discovery Module
```bash
python src/data_discovery.py
```

### 3. Run Test Suite
```bash
pytest tests/
```

## Important Dataset Isolation Rules
- **TRAIN** and **TEST** dataset groups are strictly partitioned.
- Kaggle **TEST** data is reserved strictly for final unlabeled provider-level prediction/scoring and is **never** used for model validation or cross-validation.
