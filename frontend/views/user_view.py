"""
User Claims & Risk Explorer View
--------------------------------
Provides general users, claims contributors, and data analysts with:
1. Dataset upload & validation pipeline (Perception Agent execution)
2. Interactive individual provider input form
3. Plain-language provider risk search & overview
4. Own analysis execution history
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from src.agents.perception_agent import PerceptionAgent
from src.agents.orchestrator import MultiAgentOrchestrator
from src.frontend_utils import render_badge
from frontend.components.agent_timeline import render_agent_timeline
from frontend.components.provider_intelligence import render_provider_intelligence
from src.database.connection import db_transaction


def render_user_view(username: str, role: str, db_path: str) -> None:
    """
    Renders the User Claims & Risk Explorer dashboard.
    """
    st.markdown("## 📊 Claims Explorer & Provider Risk Assessment")
    st.caption("Upload claim datasets or submit provider practice metrics to run automated multi-agent risk intelligence.")

    tab_upload, tab_form, tab_search, tab_history = st.tabs([
        "📁 Dataset Batch Upload",
        "📝 Individual Provider Entry",
        "🔍 Provider Risk Lookup",
        "📜 Analysis History"
    ])

    orchestrator = MultiAgentOrchestrator(db_path=db_path)
    perception_agent = PerceptionAgent(db_path=db_path)

    # --------------------------------------------------------------------------
    # TAB 1: Dataset Batch Upload
    # --------------------------------------------------------------------------
    with tab_upload:
        st.markdown("### 📤 Upload Claims & Beneficiary Datasets")
        st.info("Upload CSV files for Inpatient Claims, Outpatient Claims, or Beneficiary data.")

        uploaded_files = st.file_uploader(
            "Select CSV files to upload",
            type=["csv"],
            accept_multiple_files=True,
            key="user_batch_uploader"
        )

        if uploaded_files:
            st.success(f"Selected {len(uploaded_files)} file(s).")
            dfs_dict = {}
            for uf in uploaded_files:
                try:
                    dfs_dict[uf.name] = pd.read_csv(uf)
                except Exception as e:
                    st.error(f"Error reading {uf.name}: {e}")

            if st.button("🚀 Run Perception Agent Validation & Profiling", type="primary"):
                with st.spinner("Perception Agent profiling schema, keys, and data quality..."):
                    perc_result = perception_agent.analyze_uploaded_dataframes(
                        dfs_dict=dfs_dict,
                        group_name="USER_UPLOAD",
                        actor_username=username
                    )

                st.markdown("#### 📋 Perception Agent Quality Report")
                q = perc_result.quality_report
                qcol1, qcol2, qcol3 = st.columns(3)
                with qcol1:
                    st.metric("Data Quality Score", f"{q.overall_quality_score:.0f} / 100")
                with qcol2:
                    st.metric("Total Records Profiled", f"{q.total_records:,}")
                with qcol3:
                    st.metric("Key Integrity Status", q.key_integrity_status)

                if q.warnings:
                    for w in q.warnings:
                        st.warning(f"⚠️ {w}")
                else:
                    st.success("✅ All schema requirements, primary identifiers, and data types validated successfully.")

                # Profile Table
                file_rows = []
                for f in perc_result.files_profiled:
                    file_rows.append({
                        "File": f.filename,
                        "Rows": f.row_count,
                        "Columns": f.col_count,
                        "Duplicates": f.duplicate_rows,
                        "Size (MB)": f.size_mb
                    })
                st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 2: Individual Provider Form Entry
    # --------------------------------------------------------------------------
    with tab_form:
        st.markdown("### 📝 Interactive Provider Risk Evaluation Form")
        st.caption("Input practice metrics to evaluate estimated fraud risk and generate multi-agent decision support.")

        with st.form("single_provider_eval_form"):
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                form_prov_id = st.text_input("Provider Identifier", value="PRV-CUSTOM-901")
                form_state = st.number_input("Primary State Code", min_value=1, max_value=55, value=39)
                form_total_claims = st.number_input("Total Claim Volume", min_value=1, max_value=50000, value=120)
                form_total_amount = st.number_input("Total Reimbursement Amount ($)", min_value=0.0, max_value=10000000.0, value=380000.0)
                form_inpatient_claims = st.number_input("Inpatient Claims Count", min_value=0, max_value=form_total_claims, value=35)

            with pcol2:
                form_repeat_ratio = st.slider("Repeat Patient Ratio (%)", 0.0, 100.0, 45.0) / 100.0
                form_same_phys = st.slider("Attending & Operating Physician Overlap (%)", 0.0, 100.0, 55.0) / 100.0
                form_peer_avg_mult = st.slider("Average Claim vs State Peer Multiplier", 0.2, 5.0, 2.1)
                form_cpm = st.number_input("Monthly Claim Velocity (Claims/Month)", min_value=1.0, max_value=1000.0, value=65.0)
                form_avg_los = st.number_input("Average Inpatient Stay (Days)", min_value=0.0, max_value=60.0, value=5.5)

            submit_eval = st.form_submit_button("🔍 Execute Multi-Agent Risk Assessment", type="primary")

        if submit_eval:
            input_dict = {
                "total_claims": form_total_claims,
                "total_claim_amount": form_total_amount,
                "inpatient_claim_count": form_inpatient_claims,
                "repeat_beneficiary_ratio": form_repeat_ratio,
                "same_attending_operating_ratio": form_same_phys,
                "average_claim_vs_peer_average": form_peer_avg_mult,
                "claims_per_month": form_cpm,
                "average_length_of_stay": form_avg_los,
                "primary_state": form_state
            }

            with st.spinner("Executing Perception -> Fraud Analysis -> Negotiation -> Arbitrator..."):
                orch_result = orchestrator.run_pipeline_from_dict(
                    input_dict=input_dict,
                    provider_id=form_prov_id,
                    actor_username=username
                )

            st.success(f"Assessment complete for `{form_prov_id}`!")
            render_agent_timeline(orch_result)

            st.markdown("---")
            render_provider_intelligence(
                provider_id=form_prov_id,
                user_role=role,
                user_username=username
            )

    # --------------------------------------------------------------------------
    # TAB 3: Provider Risk Lookup
    # --------------------------------------------------------------------------
    with tab_search:
        st.markdown("### 🔍 Search Provider Risk Directory")
        search_prov_id = st.text_input("Enter Provider ID (e.g., PRV51069, PRV51002)", value="PRV51069")
        
        if st.button("Lookup Provider Risk", key="btn_lookup_user"):
            render_provider_intelligence(
                provider_id=search_prov_id.strip().upper(),
                user_role=role,
                user_username=username
            )

    # --------------------------------------------------------------------------
    # TAB 4: Analysis History
    # --------------------------------------------------------------------------
    with tab_history:
        st.markdown("### 📜 Your Analysis Runs & Activity")
        with db_transaction(db_path) as conn:
            runs = conn.execute(
                """
                SELECT id, timestamp, action, entity_id, status, details_json
                FROM audit_logs
                WHERE username = ?
                ORDER BY id DESC LIMIT 20
                """,
                (username,)
            ).fetchall()

        if runs:
            st.dataframe(pd.DataFrame([dict(r) for r in runs]), use_container_width=True, hide_index=True)
        else:
            st.info("No prior personal activity logs recorded.")
