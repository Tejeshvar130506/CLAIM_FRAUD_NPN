-- ==============================================================================
-- Multi-Agent Healthcare Provider Fraud Intelligence Platform
-- SQLite Database Schema (WAL Mode Enabled)
-- ==============================================================================

-- 1. Users Table (Authentication & RBAC)
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('USER', 'INVESTIGATOR', 'MANAGER', 'ADMIN')),
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- 2. Providers Table (Risk Profile Repository)
CREATE TABLE IF NOT EXISTS providers (
    provider_id TEXT PRIMARY KEY,
    primary_state TEXT,
    total_claims INTEGER DEFAULT 0,
    total_claim_amount REAL DEFAULT 0.0,
    average_claim_amount REAL DEFAULT 0.0,
    inpatient_ratio REAL DEFAULT 0.0,
    repeat_beneficiary_ratio REAL DEFAULT 0.0,
    average_claim_vs_peer_average REAL DEFAULT 1.0,
    claim_amount_vs_peer_average REAL DEFAULT 1.0,
    beneficiary_hhi_concentration REAL DEFAULT 0.0,
    same_attending_operating_ratio REAL DEFAULT 0.0,
    claims_per_month REAL DEFAULT 0.0,
    fraud_probability REAL DEFAULT 0.0,
    risk_score INTEGER DEFAULT 0,
    risk_level TEXT DEFAULT 'LOW' CHECK(risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    investigation_priority TEXT DEFAULT 'LOW' CHECK(investigation_priority IN ('LOW', 'NORMAL', 'HIGH', 'CRITICAL')),
    status TEXT DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Analysis Runs Table (Batch Analysis Run History)
CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    triggered_by TEXT NOT NULL,
    total_providers_analyzed INTEGER DEFAULT 0,
    high_risk_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'COMPLETED' CHECK(status IN ('STARTED', 'RUNNING', 'COMPLETED', 'FAILED')),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    config_json TEXT,
    FOREIGN KEY(triggered_by) REFERENCES users(username)
);

-- 4. Investigations Table (Investigation Candidates & Cases)
CREATE TABLE IF NOT EXISTS investigations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT NOT NULL,
    case_number TEXT UNIQUE NOT NULL,
    priority TEXT DEFAULT 'NORMAL' CHECK(priority IN ('LOW', 'NORMAL', 'HIGH', 'CRITICAL')),
    status TEXT DEFAULT 'NEW' CHECK(status IN ('NEW', 'ASSIGNED', 'IN_REVIEW', 'ESCALATED', 'RESOLVED_VALIDATED', 'RESOLVED_CLEARED', 'CLOSED')),
    assigned_to TEXT,
    ai_risk_score INTEGER DEFAULT 0,
    ai_risk_level TEXT DEFAULT 'HIGH',
    ai_fraud_probability REAL DEFAULT 0.0,
    escalation_reason TEXT,
    manager_decision TEXT,
    manager_reasoning TEXT,
    final_outcome TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(provider_id) REFERENCES providers(provider_id),
    FOREIGN KEY(assigned_to) REFERENCES users(username)
);

-- 5. Investigation Events Table (Timeline & Audit History for Cases)
CREATE TABLE IF NOT EXISTS investigation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('CREATED', 'ASSIGNED', 'NOTE_ADDED', 'EVIDENCE_REVIEWED', 'FINDING_RECORDED', 'ESCALATED_TO_MANAGEMENT', 'MANAGEMENT_DECISION', 'CASE_CLOSED')),
    actor_username TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    notes TEXT,
    decision TEXT,
    rationale TEXT,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(investigation_id) REFERENCES investigations(id)
);

-- 6. Agent Runs Table (Agent Execution History & Outputs)
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    provider_id TEXT,
    agent_name TEXT NOT NULL CHECK(agent_name IN ('PERCEPTION_AGENT', 'FRAUD_ANALYSIS_AGENT', 'NEGOTIATION_AGENT', 'ARBITRATOR', 'ORCHESTRATOR')),
    status TEXT DEFAULT 'COMPLETED' CHECK(status IN ('STARTED', 'RUNNING', 'COMPLETED', 'FAILED')),
    input_summary TEXT,
    output_json TEXT,
    execution_time_ms INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Audit Logs Table (Immutable System-Wide Event Ledger)
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    username TEXT NOT NULL,
    role TEXT NOT NULL,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT,
    status TEXT NOT NULL CHECK(status IN ('SUCCESS', 'FAILURE', 'WARNING', 'UNAUTHORIZED')),
    details_json TEXT,
    ip_address TEXT
);

-- 8. Model Versions Table (Model Governance & Registry)
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_tag TEXT UNIQUE NOT NULL,
    model_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    training_dataset TEXT NOT NULL,
    features_count INTEGER NOT NULL,
    roc_auc REAL,
    pr_auc REAL,
    f1_score REAL,
    is_active INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 9. Watchlist Table (Monitored Providers)
CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id TEXT NOT NULL,
    added_by TEXT NOT NULL,
    reason TEXT NOT NULL,
    alert_threshold REAL DEFAULT 0.70,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(provider_id) REFERENCES providers(provider_id)
);

-- 10. Alerts Table (System and Fraud Risk Notifications)
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('INFO', 'WARNING', 'HIGH', 'CRITICAL')),
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    entity_id TEXT,
    is_read INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_providers_risk_score ON providers(risk_score);
CREATE INDEX IF NOT EXISTS idx_investigations_status ON investigations(status);
CREATE INDEX IF NOT EXISTS idx_investigations_provider_id ON investigations(provider_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_username ON audit_logs(username);
CREATE INDEX IF NOT EXISTS idx_agent_runs_provider_id ON agent_runs(provider_id);
