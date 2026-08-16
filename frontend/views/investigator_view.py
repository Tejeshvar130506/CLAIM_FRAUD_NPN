"""
Special Investigations Unit (SIU) Investigator Console
------------------------------------------------------
Provides clinical and healthcare fraud investigators with:
1. Investigation queue with status & priority filtering
2. Multi-agent evidence inspection & EBM attributions
3. Case triage & assignment
4. PATH A (Sufficient Evidence Finding & Resolution)
5. PATH B (Insufficient Evidence Escalation to Management)
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

from src.services.investigation_service import InvestigationService
from frontend.components.provider_intelligence import render_provider_intelligence
from src.database.connection import db_transaction


def render_investigator_view(username: str, role: str, db_path: str) -> None:
    """
    Renders the Investigator Dashboard.
    """
    st.markdown("## 🔍 Special Investigations Unit (SIU) Console")
    st.caption("Review prioritized investigation candidates, inspect multi-agent evidence chains, and record findings or escalations.")

    inv_service = InvestigationService(db_path=db_path)

    # 1. Top KPI Summary
    with db_transaction(db_path) as conn:
        total_cases = conn.execute("SELECT COUNT(*) FROM investigations").fetchone()[0]
        my_cases = conn.execute("SELECT COUNT(*) FROM investigations WHERE assigned_to = ?", (username,)).fetchone()[0]
        escalated_cases = conn.execute("SELECT COUNT(*) FROM investigations WHERE status = 'ESCALATED'").fetchone()[0]
        critical_cases = conn.execute("SELECT COUNT(*) FROM investigations WHERE priority = 'CRITICAL' AND status NOT IN ('RESOLVED_VALIDATED', 'RESOLVED_CLEARED')").fetchone()[0]

    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    with kcol1:
        st.metric("Total Investigation Candidates", total_cases)
    with kcol2:
        st.metric("Assigned to Me", my_cases)
    with kcol3:
        st.metric("Escalated to Management", escalated_cases)
    with kcol4:
        st.metric("Critical Priority Queue", critical_cases)

    st.markdown("---")

    # 2. Queue Filters
    st.markdown("### 📋 Active Investigation Queue")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        status_filter = st.selectbox(
            "Filter by Case Status",
            ["ALL", "NEW", "ASSIGNED", "IN_REVIEW", "ESCALATED", "RESOLVED_VALIDATED", "RESOLVED_CLEARED"],
            index=0
        )
    with fcol2:
        priority_filter = st.selectbox(
            "Filter by Priority",
            ["ALL", "CRITICAL", "HIGH", "NORMAL", "LOW"],
            index=0
        )
    with fcol3:
        assign_filter = st.selectbox(
            "Assignment Filter",
            ["ALL CASES", "MY ASSIGNED CASES", "UNASSIGNED"],
            index=0
        )

    # Apply filters
    stat_param = None if status_filter == "ALL" else status_filter
    prio_param = None if priority_filter == "ALL" else priority_filter
    assign_param = username if assign_filter == "MY ASSIGNED CASES" else None

    cases = inv_service.get_investigation_queue(
        status=stat_param,
        priority=prio_param,
        assigned_to=assign_param,
        limit=100
    )

    if assign_filter == "UNASSIGNED":
        cases = [c for c in cases if not c.get("assigned_to")]

    if cases:
        df_queue = pd.DataFrame(cases)
        display_cols = [
            "case_number", "provider_id", "priority", "status", "ai_risk_score",
            "ai_risk_level", "assigned_to", "total_claims", "total_claim_amount", "primary_state"
        ]
        available_cols = [c for c in display_cols if c in df_queue.columns]
        
        st.dataframe(
            df_queue[available_cols],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### 🔍 Select Provider for Detailed Investigation")
        case_options = {f"{c['case_number']} — {c['provider_id']} (Score: {c['ai_risk_score']}/100 | {c['status']})": c for c in cases}
        selected_label = st.selectbox("Choose Case to Investigate", list(case_options.keys()))
        selected_case = case_options[selected_label]

        # Case Quick Action: Self-Assign
        if not selected_case.get("assigned_to") or selected_case["assigned_to"] != username:
            if st.button(f"📌 Assign Case {selected_case['case_number']} to Myself ({username})"):
                inv_service.assign_case(selected_case["id"], username, username)
                st.success("Case assigned to you.")
                st.rerun()

        st.markdown("---")
        render_provider_intelligence(
            provider_id=selected_case["provider_id"],
            user_role=role,
            user_username=username,
            inv_service=inv_service
        )
    else:
        st.info("No investigation candidates match the selected filter criteria.")
