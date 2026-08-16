# Multi-Agent Healthcare Provider Fraud Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![SQLite WAL](https://img.shields.io/badge/database-SQLite%20WAL-success.svg)](https://www.sqlite.org/wal.html)
[![Multi-Agent](https://img.shields.io/badge/architecture-Multi--Agent%20Systems-orange.svg)]()
[![Explainability](https://img.shields.io/badge/explainability-EBM%20%2B%20SHAP-blueviolet.svg)]()

A multi-agent decision support platform transforming raw Medicare inpatient, outpatient, and beneficiary claims into explainable fraud intelligence for Special Investigations Units (SIU), clinical auditors, and healthcare compliance executives.

---

## 🏛️ Target System Architecture

```
Raw Medicare Claims (Inpatient, Outpatient, Beneficiary)
                          │
                          ▼
            [ 1. PERCEPTION AGENT ]
    (Profiles schemas, checks key integrity, scores data quality 0–100)
                          │
                          ▼
         [ 2. FRAUD ANALYSIS AGENT ]
    (XGBoost ROC-AUC 0.9692 + EBM Glass-Box GAM Explanations)
                          │
                          ▼
          [ 3. NEGOTIATION AGENT ]
    (Adversarial Reasoning: Examine ➔ Argue ➔ Challenge ➔ Propose)
                          │
                          ▼
             [ 4. ARBITRATOR AGENT ]
    (Independent Synthesis, Conflict Analysis, Epistemic Uncertainty)
                          │
                          ▼
          [ INVESTIGATION CANDIDATE ]
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      [ PATH A: Finding ]     [ PATH B: Escalation ]
 (Sufficient Evidence ➔ Resolved) (Insufficient Evidence ➔ Manager Review)
                                       │
                                       ▼
                             [ MANAGEMENT DECISION ]
```

---

## 🤖 Specialized AI Agents & Contracts

1. **Perception Agent (`src/agents/perception_agent.py`)**:
   - Automated profiling of claims files and in-memory DataFrames
   - Key integrity & referential consistency validation
   - 0–100 Data Quality Scoring with automated anomaly warnings
   - Produces typed `PerceptionResult` contract

2. **Fraud Analysis Agent (`src/agents/fraud_analysis_agent.py`)**:
   - Dual-mode feature extraction (batch parquets & interactive single-provider form)
   - Calibrated fraud probability estimation (XGBoost Classifier)
   - Local additive score decomposition via Explainable Boosting Machine (EBM)
   - Produces typed `EvidencePackage` contract

3. **Negotiation Agent (`src/agents/negotiation_agent.py`)**:
   - 4-Stage Adversarial Workflow:
     - **EXAMINE**: Ingests billing metrics, volume deviations, and network concentrations
     - **ARGUE**: Builds strongest pro-investigation evidence points
     - **CHALLENGE**: Formulates skeptical counter-hypotheses (specialization, patient acuity, geography)
     - **PROPOSE**: Synthesizes balanced structured action (`HIGH_PRIORITY_INVESTIGATION`, `AUDIT_REVIEW`, `ROUTINE_MONITORING`)
   - Produces typed `NegotiationResult` contract

4. **Arbitrator Agent (`src/agents/arbitrator_agent.py`)**:
   - Independent cross-agent synthesis (Perception + ML Evidence + Adversarial Challenges)
   - Evaluates conflicting evidence and epistemic uncertainty
   - Determines **Investigation Candidate** qualification (auto-provisioned into SQLite)
   - Produces typed `ArbitratorResult` contract

5. **Multi-Agent Orchestrator (`src/agents/orchestrator.py`)**:
   - End-to-end execution coordinator running the complete multi-agent pipeline

---

## 👥 Role-Based Access Control (RBAC) & Persona Views

| Role | Workspace | Core Permissions |
|---|---|---|
| **`USER`** | Claims Explorer | Upload datasets, evaluate individual providers, plain-language risk lookup |
| **`INVESTIGATOR`** | SIU Console | Case queue triage, full multi-agent evidence inspection, clinical notes, **Path A Findings**, **Path B Escalation** |
| **`MANAGER`** | Operations Portal | Executive KPIs, $ risk exposure analytics, escalated case review, **Management Decision adjudication** |
| **`ADMIN`** | Governance Portal | SQLite WAL diagnostics, user CRUD & role assignment, Model Registry, AI agent telemetry, Audit Log JSON ledger |

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment
```bash
# Python 3.10+ required
pip install -r requirements.txt
```

### 2. Run Complete Test Suite
```bash
python -m pytest tests/ -v
```

### 3. Launch Streamlit Multi-Role Application
```bash
streamlit run app.py
```

### 4. Default Demo Accounts
- **Claims Analyst**: Username `user` / Password `UserPass@2026`
- **SIU Investigator**: Username `investigator` / Password `InvestigatorPass@2026`
- **SIU Manager**: Username `manager` / Password `ManagerPass@2026`
- **System Administrator**: Username `admin` / Password `AdminPass@2026`

---

## 🔒 Security, Compliance & Audit Trail

- **Password Hashing**: PBKDF2-HMAC-SHA256 with cryptographically random per-user salt and 600,000 iterations
- **Database Concurrency**: SQLite with Write-Ahead Logging (`WAL`), `busy_timeout = 5000ms`, `synchronous = NORMAL`, and foreign key integrity
- **Audit Logging**: Immutable, queryable ledger recording all authentication, agent runs, case notes, investigator findings, and managerial decisions with automatic secret redaction
- **Decision-Support Terminology**: Strict non-accusatory language ("Elevated fraud risk identified", "Potentially suspicious provider", "Investigation candidate") ensuring ethical and legally defensible human-in-the-loop governance.
