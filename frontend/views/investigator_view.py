"""
Special Investigations Unit (SIU) Investigator Console
------------------------------------------------------
1. Prioritized Investigation Queue with Chronological Ordering (Newest to Oldest or Oldest to Newest)
2. Single-ID and Batch Multi-Agent Analysis (Evidence Package + Negotiation + Arbitrator Outcome)
3. Dedicated Claim ID / Provider ID Status Search Bar
4. Sufficient Evidence Decision: Classify as FRAUD vs NOT FRAUD (Path A)
5. Insufficient Evidence Escalation: Send Case to Manager Database with Reasoning (Path B)
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

from src.services.investigation_service import InvestigationService
from src.agents.orchestrator import MultiAgentOrchestrator
from frontend.components.provider_intelligence import render_provider_intelligence
from frontend.components.agent_timeline import render_agent_timeline
from src.database.connection import db_transaction


def render_investigator_view(username: str, role: str, db_path: str) -> None:
    """
    Renders the enhanced Investigator Dashboard.
    """
    st.markdown("## 🔍 Special Investigations Unit (SIU) Console")
    st.caption("Review prioritized investigation candidates in chronological order, run multi-agent evidence synthesis, search claim IDs, and classify cases as Fraud / Not Fraud or escalate to Management.")

    inv_service = InvestigationService(db_path=db_path)
    orchestrator = MultiAgentOrchestrator(db_path=db_path)

    # 1. Top KPI Summary
    with db_transaction(db_path) as conn:
        total_cases = conn.execute("SELECT COUNT(*) FROM investigations").fetchone()[0]
        my_cases = conn.execute("SELECT COUNT(*) FROM investigations WHERE assigned_to = ?", (username,)).fetchone()[0]
        escalated_cases = conn.execute("SELECT COUNT(*) FROM investigations WHERE status = 'ESCALATED'").fetchone()[0]
        resolved_fraud = conn.execute("SELECT COUNT(*) FROM investigations WHERE status = 'RESOLVED_VALIDATED'").fetchone()[0]
        resolved_cleared = conn.execute("SELECT COUNT(*) FROM investigations WHERE status = 'RESOLVED_CLEARED'").fetchone()[0]

    kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)
    with kcol1:
        st.metric("Total Candidates", total_cases)
    with kcol2:
        st.metric("Assigned to Me", my_cases)
    with kcol3:
        st.metric("🚨 Escalated to Mgr", escalated_cases)
    with kcol4:
        st.metric("🔴 Confirmed Fraud", resolved_fraud)
    with kcol5:
        st.metric("🟢 Cleared (Not Fraud)", resolved_cleared)

    st.markdown("---")

    # 2. Investigation Queue with Newest/Oldest Order Toggle
    st.markdown("### 📋 Investigation Queue")
    
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    with qcol1:
        order_choice = st.selectbox(
            "Sort Queue Chronology:",
            ["Newest to Oldest (Latest First)", "Oldest to Newest (Earliest First / FIFO)", "Highest Risk Score First"],
            index=0
        )
    with qcol2:
        status_filter = st.selectbox(
            "Filter by Case Status:",
            ["ALL", "NEW", "ASSIGNED", "IN_REVIEW", "ESCALATED", "RESOLVED_VALIDATED", "RESOLVED_CLEARED"],
            index=0
        )
    with qcol3:
        priority_filter = st.selectbox(
            "Filter by Priority:",
            ["ALL", "CRITICAL", "HIGH", "NORMAL", "LOW"],
            index=0
        )
    with qcol4:
        assign_filter = st.selectbox(
            "Assignment:",
            ["ALL CASES", "MY ASSIGNED CASES", "UNASSIGNED"],
            index=0
        )

    # Order translation
    order_key = "NEWEST"
    if order_choice == "Oldest to Newest (Earliest First / FIFO)":
        order_key = "OLDEST"
    elif order_choice == "Highest Risk Score First":
        order_key = "RISK_SCORE"

    stat_param = None if status_filter == "ALL" else status_filter
    prio_param = None if priority_filter == "ALL" else priority_filter
    assign_param = username if assign_filter == "MY ASSIGNED CASES" else None

    cases = inv_service.get_investigation_queue(
        status=stat_param,
        priority=prio_param,
        assigned_to=assign_param,
        order_by=order_key,
        limit=100
    )

    if assign_filter == "UNASSIGNED":
        cases = [c for c in cases if not c.get("assigned_to")]

    if cases:
        df_queue = pd.DataFrame(cases)
        display_cols = [
            "id", "case_number", "provider_id", "priority", "status", "ai_risk_score",
            "ai_risk_level", "assigned_to", "total_claims", "total_claim_amount", "created_at"
        ]
        available_cols = [c for c in display_cols if c in df_queue.columns]
        st.dataframe(df_queue[available_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No investigation candidates match the current filter criteria.")

    st.markdown("---")

    # 3. Multi-Agent Evidence & Arbitrator Analysis Console
    st.markdown("### 🤖 Multi-Agent Evidence & Arbitrator Analysis")
    st.caption("Execute full Evidence Package, adversarial Negotiation, and Arbitrator synthesis for a single ID or the entire active queue.")

    if cases:
        case_options = {f"{c['provider_id']} (Case: {c['case_number']} | Score: {c['ai_risk_score']}/100 | {c['status']})": c for c in cases}
        selected_label = st.selectbox("Select Provider ID to Inspect & Analyze:", list(case_options.keys()))
        selected_case = case_options[selected_label]
        target_provider_id = selected_case["provider_id"]

        acol1, acol2 = st.columns(2)
        with acol1:
            # Self-assign button
            if not selected_case.get("assigned_to") or selected_case["assigned_to"] != username:
                if st.button(f"📌 Assign Case {selected_case['case_number']} to Myself ({username})"):
                    inv_service.assign_case(selected_case["id"], username, username)
                    st.success(f"Case assigned to {username}.")
                    st.rerun()

        st.markdown("---")

        # Provider Intelligence with Evidence & Arbitrator
        render_provider_intelligence(
            provider_id=target_provider_id,
            user_role=role,
            user_username=username,
            inv_service=inv_service
        )

        st.markdown("---")

        # 4. Classification & Decision Console (Path A vs Path B)
        st.markdown("### ⚖️ Case Classification & Decision Console")
        st.caption("Classify this claim/provider as Fraud or Not Fraud with sufficient evidence, or escalate to Management if ambiguous.")

        tab_classify, tab_escalate, tab_notes = st.tabs([
            "🟢 Path A: Classify Case (Sufficient Evidence)",
            "🟡 Path B: Escalate to Management Database (Insufficient Evidence)",
            "📝 Case Notes & History"
        ])

        # PATH A: Classify as FRAUD vs NOT FRAUD
        with tab_classify:
            st.markdown("#### Path A — Sufficient Evidence Determination")
            st.info("When clinical and billing evidence is sufficient, record the definitive classification outcome:")

            decision_type = st.radio(
                "Classification Outcome:",
                [
                    "🔴 FRAUD (Elevated Fraud Risk Confirmed — Initiate Overpayment Recovery)",
                    "🟢 NOT FRAUD (Suspicion Cleared — Legitimate Clinical Practice)",
                    "🟡 IN REVIEW (Continue Routine Surveillance)"
                ]
            )

            clinical_reasoning = st.text_area(
                "Clinical / Audit Findings & Reasoning (Required):",
                placeholder="Detail specific evidence (e.g. unbundled surgery codes, patient recycling loops, or legitimate tertiary specialization)...",
                key=f"reasoning_{target_provider_id}"
            )

            follow_up = st.text_input(
                "Follow-up Action:",
                value="Initiate payment hold and request chart audit for full reimbursement recovery.",
                key=f"follow_up_{target_provider_id}"
            )

            if st.button("💾 Record Classification & Resolve Case", type="primary", key=f"btn_record_{target_provider_id}"):
                if len(clinical_reasoning.strip()) < 10:
                    st.error("Please provide substantive clinical / audit findings of at least 10 characters.")
                else:
                    finding_key = "ELEVATED_RISK_VALIDATED" if "FRAUD (" in decision_type else ("SUSPICION_CLEARED_LEGITIMATE" if "NOT FRAUD" in decision_type else "MONITORING_CONTINUED")
                    inv_service.record_investigator_finding(
                        investigation_id=selected_case["id"],
                        finding_type=finding_key,
                        reasoning=clinical_reasoning,
                        follow_up_action=follow_up,
                        actor_username=username
                    )
                    st.success(f"Case {selected_case['case_number']} successfully classified and recorded!")
                    st.rerun()

        # PATH B: Insufficient Evidence Escalation
        with tab_escalate:
            st.markdown("#### Path B — Insufficient Evidence: Send to Manager Database")
            st.caption("If available claims evidence is insufficient to confidently classify, transmit this ID to the Manager database with structured reasoning.")

            esc_reason = st.text_area(
                "Reasoning for Management Escalation (Mandatory):",
                placeholder="e.g., Statistical billing deviations present, but provider operates as sole specialized regional pediatric trauma clinic. Clinical ambiguity requires managerial sign-off for clinical chart subpoena.",
                key=f"esc_reason_txt_{target_provider_id}"
            )

            if st.button("🚨 Transmit Case to Manager Database", type="secondary", key=f"btn_esc_mgr_{target_provider_id}"):
                if len(esc_reason.strip()) < 10:
                    st.error("Please provide structured justification of at least 10 characters for management escalation.")
                else:
                    inv_service.escalate_to_management(
                        investigation_id=selected_case["id"],
                        escalation_reason=esc_reason,
                        actor_username=username
                    )
                    st.success(f"Case {selected_case['case_number']} successfully transmitted to Management Review Database!")
                    st.rerun()

        # Tab Notes
        with tab_notes:
            note_content = st.text_area("Add Timestamped Note to Case Timeline:", key=f"inv_note_{target_provider_id}")
            if st.button("Save Note", key=f"btn_save_note_{target_provider_id}"):
                if note_content.strip():
                    inv_service.add_case_note(selected_case["id"], note_content, username, role)
                    st.success("Note saved to timeline.")
                    st.rerun()

    st.markdown("---")

    # 5. Dedicated Claim ID / Provider ID Search Bar & Status Viewer
    st.markdown("### 🔍 Search Claim ID / Provider ID Status")
    st.caption("Enter any Provider ID, Claim ID, or Case Number to inspect its real-time investigation status and history.")

    scol1, scol2 = st.columns([3, 1])
    with scol1:
        search_term = st.text_input(
            "Enter Provider ID or Case Number:",
            value="",
            placeholder="e.g. PRV51069, INV-PRV51069-1001",
            key="inv_claim_search_input"
        )
    with scol2:
        st.write("")
        st.write("")
        btn_search = st.button("Search Case Status", use_container_width=True, type="primary")

    if search_term.strip() or btn_search:
        term = search_term.strip().upper()
        found_cases = inv_service.get_investigation_queue(search_query=term, limit=10)

        if found_cases:
            st.success(f"Found {len(found_cases)} matching record(s):")
            for fc in found_cases:
                with st.expander(f"📌 Case: **{fc['case_number']}** — Provider: `{fc['provider_id']}` | Status: **{fc['status']}** | Score: **{fc['ai_risk_score']}/100**", expanded=True):
                    rcol1, rcol2, rcol3, rcol4 = st.columns(4)
                    with rcol1:
                        st.write(f"**Investigation Status**:\n\n`{fc['status']}`")
                    with rcol2:
                        st.write(f"**AI Risk Score**:\n\n`{fc['ai_risk_score']}/100` ({fc['ai_risk_level']})")
                    with rcol3:
                        st.write(f"**Assigned To**:\n\n`{fc.get('assigned_to') or 'Unassigned'}`")
                    with rcol4:
                        st.write(f"**Priority**:\n\n`{fc['priority']}`")

                    if fc.get("final_outcome"):
                        st.info(f"**Final Outcome Determination**:\n\n{fc['final_outcome']}")

                    if fc.get("escalation_reason"):
                        st.warning(f"**Management Escalation Rationale**:\n\n{fc['escalation_reason']}")

                    # Case Timeline Events
                    events = inv_service.get_case_events(fc["id"])
                    if events:
                        st.markdown("##### 📜 Case History Timeline:")
                        for ev in events:
                            st.markdown(f"- **{ev['created_at']}** | `{ev['actor_role']}` **{ev['actor_username']}** | `{ev['event_type']}`: {ev.get('notes') or ev.get('rationale') or ''}")
        else:
            st.warning(f"No active investigation case found matching `{search_term}`.")
