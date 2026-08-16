"""
System Administration & Model Governance Dashboard
--------------------------------------------------
Provides administrators with comprehensive system governance:
1. System Health & SQLite WAL diagnostic monitor
2. User Management & RBAC role assignments
3. Machine Learning Model Registry & Performance Governance
4. AI Agent execution monitoring & telemetry
5. Immutable System Audit Log Ledger with JSON inspection
6. Provider Watchlist & System Alerts
"""

import os
import json
import streamlit as st
import pandas as pd
from typing import Dict, Any

from src.auth.security import list_users, create_user, update_user_role, toggle_user_active
from src.services.audit_service import get_audit_logs, get_audit_summary_metrics
from src.database.connection import db_transaction, get_db_connection


def render_admin_view(username: str, role: str, db_path: str) -> None:
    """
    Renders the System Administration Dashboard.
    """
    st.markdown("## ⚙️ System Administration & Governance Portal")
    st.caption("Manage user access controls, oversee machine learning model artifacts, monitor AI agents, and inspect audit logs.")

    tab_users, tab_models, tab_agents, tab_audit, tab_health, tab_watch = st.tabs([
        "👥 User Management & RBAC",
        "🧠 Model Registry",
        "🤖 AI Agent Telemetry",
        "📜 System Audit Logs",
        "🖥️ System Health",
        "👁️ Watchlist & Alerts"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: User Management & RBAC
    # --------------------------------------------------------------------------
    with tab_users:
        st.markdown("### 👥 System Users & Role Assignments")
        users = list_users(db_path=db_path)
        if users:
            st.dataframe(pd.DataFrame(users), use_container_width=True, hide_index=True)

        st.markdown("#### ➕ Create New System User")
        with st.form("create_user_form"):
            ucol1, ucol2 = st.columns(2)
            with ucol1:
                new_uname = st.text_input("Username")
                new_pwd = st.text_input("Temporary Password", type="password")
                new_role = st.selectbox("Role", ["USER", "INVESTIGATOR", "MANAGER", "ADMIN"])
            with ucol2:
                new_fname = st.text_input("Full Name")
                new_email = st.text_input("Email Address")

            if st.form_submit_button("Create User Account", type="primary"):
                if new_uname and new_pwd and new_fname and new_email:
                    try:
                        create_user(new_uname, new_pwd, new_role, new_fname, new_email, db_path=db_path)
                        st.success(f"User account `{new_uname}` ({new_role}) created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create user: {e}")
                else:
                    st.error("All user fields are required.")

        st.markdown("#### 🔄 Modify User Role / Active State")
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            target_uname = st.selectbox("Select User", [u["username"] for u in users], key="target_user_sel")
            target_role = st.selectbox("New Role", ["USER", "INVESTIGATOR", "MANAGER", "ADMIN"], key="target_role_sel")
            if st.button("Update Role", key="btn_upd_role"):
                update_user_role(target_uname, target_role, db_path=db_path)
                st.success(f"Updated `{target_uname}` role to `{target_role}`.")
                st.rerun()

        with mcol2:
            st.write(f"Toggle Active status for `{target_uname}`:")
            if st.button("Toggle Active / Inactive State", key="btn_toggle_act"):
                toggle_user_active(target_uname, db_path=db_path)
                st.success(f"Toggled active state for `{target_uname}`.")
                st.rerun()

    # --------------------------------------------------------------------------
    # TAB 2: Model Registry
    # --------------------------------------------------------------------------
    with tab_models:
        st.markdown("### 🧠 Production Machine Learning Model Registry")
        st.caption("Active models generating provider fraud probability and glass-box explainability attributions.")

        with db_transaction(db_path) as conn:
            models_rows = conn.execute("SELECT * FROM model_versions ORDER BY id DESC").fetchall()

        if models_rows:
            st.dataframe(pd.DataFrame([dict(r) for r in models_rows]), use_container_width=True, hide_index=True)
        else:
            st.info("No models currently registered in model_versions table.")

    # --------------------------------------------------------------------------
    # TAB 3: AI Agent Telemetry
    # --------------------------------------------------------------------------
    with tab_agents:
        st.markdown("### 🤖 Multi-Agent Execution Telemetry & Latency")
        with db_transaction(db_path) as conn:
            agent_runs = conn.execute(
                """
                SELECT id, run_id, provider_id, agent_name, status, execution_time_ms, created_at, input_summary
                FROM agent_runs
                ORDER BY id DESC LIMIT 50
                """
            ).fetchall()

        if agent_runs:
            df_runs = pd.DataFrame([dict(r) for r in agent_runs])
            st.dataframe(df_runs, use_container_width=True, hide_index=True)
        else:
            st.info("No agent execution history recorded yet.")

    # --------------------------------------------------------------------------
    # TAB 4: System Audit Logs
    # --------------------------------------------------------------------------
    with tab_audit:
        st.markdown("### 📜 System-Wide Immutable Audit Ledger")
        
        # Summary Metrics
        audit_metrics = get_audit_summary_metrics(db_path=db_path)
        acol1, acol2, acol3, acol4 = st.columns(4)
        with acol1:
            st.metric("Total Logged Events", audit_metrics["total_events"])
        with acol2:
            st.metric("Success Rate", f"{audit_metrics['success_rate_pct']:.1f}%")
        with acol3:
            st.metric("Failure / Security Events", audit_metrics["failure_events"])
        with acol4:
            st.metric("Unique Active Actors", audit_metrics["unique_active_users"])

        # Filter Logs
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            act_filter = st.text_input("Filter Action (e.g. LOGIN, FINDING, AGENT)", "")
        with fcol2:
            user_filter = st.text_input("Filter Username", "")

        logs = get_audit_logs(
            username=user_filter if user_filter else None,
            action=act_filter if act_filter else None,
            limit=50,
            db_path=db_path
        )

        if logs:
            df_logs = pd.DataFrame(logs)
            st.dataframe(
                df_logs[["id", "timestamp", "username", "role", "action", "entity_type", "entity_id", "status"]],
                use_container_width=True,
                hide_index=True
            )

            with st.expander("🔍 Inspect Full Audit Event Details (JSON)"):
                selected_log_id = st.selectbox("Select Log ID to Inspect", [l["id"] for l in logs])
                selected_log = next((l for l in logs if l["id"] == selected_log_id), None)
                if selected_log:
                    st.json(selected_log)
        else:
            st.info("No audit logs match criteria.")

    # --------------------------------------------------------------------------
    # TAB 5: System Health
    # --------------------------------------------------------------------------
    with tab_health:
        st.markdown("### 🖥️ SQLite Database & Concurrency Diagnostic Health")
        
        conn = get_db_connection(db_path)
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        sync = conn.execute("PRAGMA synchronous;").fetchone()[0]
        timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        conn.close()

        hcol1, hcol2, hcol3 = st.columns(3)
        with hcol1:
            st.success(f"**Journal Mode**: `{mode.upper()}` (WAL Enabled)")
        with hcol2:
            st.info(f"**Synchronous Level**: `{sync}` (NORMAL)")
        with hcol3:
            st.info(f"**Busy Timeout**: `{timeout} ms`")

        st.markdown("#### 📊 Database Table Record Counts")
        table_counts = []
        with db_transaction(db_path) as conn:
            tables = ["users", "providers", "investigations", "investigation_events", "agent_runs", "audit_logs", "model_versions", "watchlist", "alerts"]
            for t in tables:
                cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                table_counts.append({"Table Name": t, "Record Count": f"{cnt:,}"})

        st.dataframe(pd.DataFrame(table_counts), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 6: Watchlist & Alerts
    # --------------------------------------------------------------------------
    with tab_watch:
        st.markdown("### 👁️ Monitored Provider Watchlist")
        with db_transaction(db_path) as conn:
            watch_items = conn.execute("SELECT * FROM watchlist ORDER BY id DESC").fetchall()
            alerts = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 20").fetchall()

        wcol1, wcol2 = st.columns(2)
        with wcol1:
            st.markdown("#### 📌 Add Provider to Watchlist")
            with st.form("add_watch_form"):
                w_prov_id = st.text_input("Provider ID (e.g. PRV51069)")
                w_reason = st.text_input("Watch Reason", value="Under strategic review")
                w_thresh = st.slider("Alert Probability Threshold", 0.1, 0.95, 0.70)
                if st.form_submit_button("Add to Watchlist", type="primary"):
                    if w_prov_id:
                        with db_transaction(db_path) as conn:
                            conn.execute(
                                "INSERT INTO watchlist (provider_id, added_by, reason, alert_threshold) VALUES (?, ?, ?, ?)",
                                (w_prov_id.strip().upper(), username, w_reason, w_thresh)
                            )
                        st.success(f"Added `{w_prov_id}` to watchlist.")
                        st.rerun()

            if watch_items:
                st.dataframe(pd.DataFrame([dict(r) for r in watch_items]), use_container_width=True, hide_index=True)
            else:
                st.info("Watchlist is currently empty.")

        with wcol2:
            st.markdown("#### 🚨 Recent System & SIU Alerts")
            if alerts:
                for a in alerts:
                    st.warning(f"**{a['title']}** ({a['created_at']})\n\n{a['message']}")
            else:
                st.info("No active system alerts.")
