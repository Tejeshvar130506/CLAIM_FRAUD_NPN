"""
Shared Provider Intelligence Component
---------------------------------------
Reusable, multi-layered provider drill-down component featuring Progressive Disclosure:
1. Executive / Plain Language Risk Overview (Accessible to all roles)
2. Detailed Behavioral Metrics & Peer Comparisons
3. Glass-box Technical Explainability (EBM additive effects & XGBoost attributions)
4. Multi-Agent Reasoning Chain (Negotiation Arguments & Challenges)
5. Case Timeline & Investigation History
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

from src.agents.contracts import EvidencePackage, NegotiationResult, ArbitratorResult
from src.services.explainability_service import ExplainabilityService
from src.services.feature_service import FeatureEngineeringService
from src.services.investigation_service import InvestigationService
from src.auth.rbac import has_permission, Permission


def render_provider_intelligence(
    provider_id: str,
    user_role: str,
    user_username: str,
    df_features_row: Optional[pd.DataFrame] = None,
    inv_service: Optional[InvestigationService] = None
) -> None:
    """
    Renders the unified Provider Intelligence module tailored by RBAC permissions.
    """
    if inv_service is None:
        inv_service = InvestigationService()

    # Load provider details
    explain_service = ExplainabilityService()
    feat_service = FeatureEngineeringService()

    if df_features_row is None:
        # Load from precomputed feature matrix or DB
        from src.config import FEATURES_DATA_DIR
        import os
        train_path = os.path.join(FEATURES_DATA_DIR, "train_provider_features.parquet")
        test_path = os.path.join(FEATURES_DATA_DIR, "test_provider_features.parquet")
        
        found_df = None
        if os.path.exists(train_path):
            df_tr = pd.read_parquet(train_path)
            match = df_tr[df_tr['Provider'] == provider_id]
            if not match.empty:
                found_df = match.iloc[[0]]
                
        if found_df is None and os.path.exists(test_path):
            df_te = pd.read_parquet(test_path)
            match = df_te[df_te['Provider'] == provider_id]
            if not match.empty:
                found_df = match.iloc[[0]]

        if found_df is None:
            st.error(f"Provider `{provider_id}` features not found in feature repository.")
            return
        df_features_row = found_df

    evidence: EvidencePackage = explain_service.generate_evidence_package(
        provider_id=provider_id,
        df_feature_row=df_features_row
    )

    # --------------------------------------------------------------------------
    # LEVEL 1: Plain Language Executive Risk Overview (All Roles)
    # --------------------------------------------------------------------------
    st.markdown(f"## 🏥 Provider Intelligence: `{provider_id}`")
    
    # Top KPI Badges
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        st.metric("Model Fraud Probability", f"{evidence.fraud_probability*100:.1f}%")
    with bcol2:
        st.metric("Operational Risk Score", f"{evidence.risk_score} / 100")
    with bcol3:
        level_colors = {"LOW": "🟢 LOW", "MEDIUM": "🟡 MEDIUM", "HIGH": "🟠 HIGH", "CRITICAL": "🔴 CRITICAL"}
        st.metric("Risk Priority Level", level_colors.get(evidence.risk_level, evidence.risk_level))
    with bcol4:
        st.metric("Investigation Priority", evidence.investigation_priority)

    # Plain Language Summary Box
    st.info(
        f"**Why is this provider flagged?**\n\n"
        f"The AI multi-agent system identified elevated statistical risk patterns in this provider's billing history. "
        f"This assessment represents a **decision-support investigation recommendation** to assist human auditors, "
        f"and does not constitute proof of fraudulent intent."
    )

    # Key Plain Language Findings
    st.markdown("#### 🔍 Primary Plain-Language Findings")
    for finding in evidence.key_findings:
        st.markdown(f"- 📌 **{finding}**")

    # --------------------------------------------------------------------------
    # LEVEL 2: Detailed Behavioral Metrics & Peer Comparison (Progressive)
    # --------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("📊 **View Behavioral Metrics & Peer Group Benchmarking**", expanded=(user_role != "USER")):
        st.markdown("##### Provider Operational Profile vs Peer Baselines")
        
        m_cols = st.columns(3)
        metrics_list = evidence.behavioral_metrics
        for idx, m in enumerate(metrics_list):
            col_target = m_cols[idx % 3]
            with col_target:
                sev_icon = "🔴" if m.severity == "CRITICAL" else ("🟠" if m.severity == "HIGH" else ("🟡" if m.severity == "ELEVATED" else "🟢"))
                col_target.metric(
                    label=f"{sev_icon} {m.display_name}",
                    value=f"{m.value:,.2f}" if isinstance(m.value, float) else f"{m.value:,}",
                    delta=f"{m.peer_ratio:.1f}x peer avg" if m.peer_ratio is not None else None
                )
                col_target.caption(m.plain_language_explanation)

        # Plotly Peer Comparison Bar Chart
        peer_metrics_data = [
            {"Metric": "Reimbursement Ratio", "Ratio": float(df_features_row['average_claim_vs_peer_average'].values[0]), "Benchmark": 1.0},
            {"Metric": "Volume Deviation", "Ratio": float(df_features_row['claim_amount_vs_peer_average'].values[0]), "Benchmark": 1.0},
            {"Metric": "Repeat Patient Ratio", "Ratio": float(df_features_row['repeat_beneficiary_ratio'].values[0]) / 0.18, "Benchmark": 1.0},
            {"Metric": "Inpatient Share", "Ratio": float(df_features_row['inpatient_ratio'].values[0]) / 0.15, "Benchmark": 1.0}
        ]
        df_peer_chart = pd.DataFrame(peer_metrics_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_peer_chart["Metric"],
            y=df_peer_chart["Ratio"],
            name="Provider Metric / Peer Benchmark",
            marker_color=['#dc2626' if r > 2.0 else '#ea580c' if r > 1.4 else '#0284c7' for r in df_peer_chart["Ratio"]]
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="State Peer Group Baseline (1.0x)")
        fig.update_layout(
            title="Provider Deviation Multipliers Relative to State Peer Group (1.0x Baseline)",
            yaxis_title="Ratio Multiplier",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------
    # LEVEL 3: Technical Explainability & EBM Contributions (Investigator/Manager/Admin)
    # --------------------------------------------------------------------------
    if has_permission(user_role, Permission.VIEW_EBM_SHAP_ANALYSIS):
        with st.expander("🧪 **View Technical Machine Learning Explainability (EBM Glass-Box)**", expanded=False):
            st.markdown("##### Explainable Boosting Machine (EBM) Additive Feature Contributions")
            st.caption(
                "EBM Generalized Additive Model decomposition: shows exact additive score attributions "
                "increasing (+) or decreasing (-) the model risk score."
            )

            ebm_items = evidence.ebm_contributions
            if ebm_items:
                df_ebm = pd.DataFrame([e.model_dump() for e in ebm_items])
                
                fig_ebm = px.bar(
                    df_ebm,
                    x="score_contribution",
                    y="display_name",
                    orientation="h",
                    color="direction",
                    color_discrete_map={
                        "INCREASES_RISK": "#dc2626",
                        "DECREASES_RISK": "#16a34a",
                        "NEUTRAL": "#94a3b8"
                    },
                    title="Top Additive Feature Score Contributions (EBM GAM Decomposition)"
                )
                fig_ebm.update_layout(yaxis=dict(autorange="reversed"), height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_ebm, use_container_width=True)

                st.dataframe(
                    df_ebm[["display_name", "actual_value", "score_contribution", "direction", "plain_summary"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("EBM local contribution attributions loaded within normal parameters.")

    # --------------------------------------------------------------------------
    # LEVEL 4: Multi-Agent Adversarial Reasoning (Investigator/Manager/Admin)
    # --------------------------------------------------------------------------
    if has_permission(user_role, Permission.VIEW_AGENT_CHAIN):
        with st.expander("🤖 **View Multi-Agent Reasoning Chain (Negotiation & Arbitrator)**", expanded=False):
            from src.agents.negotiation_agent import NegotiationAgent
            from src.agents.arbitrator_agent import Arbitrator

            neg_agent = NegotiationAgent(db_path=inv_service.db_path)
            arb_agent = Arbitrator(db_path=inv_service.db_path)

            neg_res = neg_agent.negotiate(evidence)
            arb_res = arb_agent.arbitrate(evidence, neg_res)

            st.markdown("#### 1. Negotiation Agent — Adversarial Evidence Debate")
            ncol1, ncol2 = st.columns(2)
            with ncol1:
                st.markdown("##### ⚖️ Case Supporting Investigation (Arguments)")
                for arg in neg_res.arguments_supporting_investigation:
                    st.warning(f"**{arg.title}** ({arg.strength})\n\n{arg.evidence_basis}")

            with ncol2:
                st.markdown("##### 🛡️ Skeptical Counter-Challenges & Clinical Alternatives")
                for chal in neg_res.counter_challenges_and_mitigations:
                    st.info(f"**{chal.alternative_hypothesis}** ({chal.clinical_plausibility})\n\n" + "\n".join([f"- {m}" for m in chal.mitigating_factors]))

            st.markdown("#### 2. Arbitrator — Independent AI Risk Synthesis")
            st.success(f"**AI Risk Assessment**: {arb_res.ai_risk_assessment}")
            st.markdown(f"**Conflicting Signals Evaluation**: {arb_res.conflicting_evidence_analysis}")
            st.markdown(f"**Epistemic Uncertainty Rating**: {arb_res.uncertainty_assessment}")

    # --------------------------------------------------------------------------
    # LEVEL 5: Investigation Case Management & Decision Console
    # --------------------------------------------------------------------------
    if has_permission(user_role, Permission.RECORD_INVESTIGATION_FINDING) or has_permission(user_role, Permission.RECORD_MANAGEMENT_DECISION):
        st.markdown("---")
        st.markdown("### 📋 Investigation Workflow & Decision Console")

        case = inv_service.get_investigation_by_provider(provider_id)
        if case:
            st.markdown(f"**Case Number**: `{case['case_number']}` | **Current Status**: `{case['status']}` | **Assigned To**: `{case.get('assigned_to') or 'Unassigned'}`")

            # Timeline History
            events = inv_service.get_case_events(case["id"])
            with st.expander(f"📜 View Case Timeline History ({len(events)} events)", expanded=False):
                for ev in events:
                    st.markdown(f"- **{ev['created_at']}** | `{ev['actor_role']}` **{ev['actor_username']}** | `{ev['event_type']}`: {ev.get('notes') or ev.get('rationale') or ''}")

            # Investigator Decision Actions (Path A vs Path B)
            if has_permission(user_role, Permission.RECORD_INVESTIGATION_FINDING) and case["status"] not in {"RESOLVED_VALIDATED", "RESOLVED_CLEARED", "CLOSED"}:
                st.markdown("#### 🧑‍⚖️ Investigator Action Console")
                
                tab_a, tab_b, tab_note = st.tabs(["🟢 Path A: Record Finding (Sufficient Evidence)", "🟡 Path B: Escalate to Management (Insufficient Evidence)", "📝 Add Note"])

                # PATH A
                with tab_a:
                    st.markdown("##### Path A — Sufficient Evidence for Investigation Conclusion")
                    f_type = st.selectbox(
                        "Investigation Outcome Finding",
                        ["ELEVATED_RISK_VALIDATED", "SUSPICION_CLEARED_LEGITIMATE", "MONITORING_CONTINUED"],
                        key=f"finding_type_{provider_id}"
                    )
                    f_reason = st.text_area("Clinical Findings & Investigation Reasoning", key=f"f_reason_{provider_id}")
                    f_action = st.text_input("Recommended Follow-up Action", value="Initiate overpayment audit and demand records recovery.", key=f"f_act_{provider_id}")
                    
                    if st.button("Submit Finding & Resolve Case", key=f"btn_path_a_{provider_id}", type="primary"):
                        if len(f_reason.strip()) < 10:
                            st.error("Please provide substantive clinical investigation findings.")
                        else:
                            inv_service.record_investigator_finding(
                                investigation_id=case["id"],
                                finding_type=f_type,
                                reasoning=f_reason,
                                follow_up_action=f_action,
                                actor_username=user_username
                            )
                            st.success("Investigator finding recorded successfully!")
                            st.rerun()

                # PATH B
                with tab_b:
                    st.markdown("##### Path B — Insufficient Evidence: Escalate to Management Review")
                    st.caption("Select this path if claims analysis exhibits ambiguity that cannot be resolved without management intervention.")
                    esc_reason = st.text_area(
                        "Reasoning for Management Escalation (Mandatory)",
                        placeholder="e.g., Statistical deviations present, but provider is sole specialized burn center in county; clinical ambiguity requires managerial guidance on record subpoena.",
                        key=f"esc_reason_{provider_id}"
                    )
                    if st.button("🚨 Escalate Case to Management Queue", key=f"btn_path_b_{provider_id}", type="secondary"):
                        if len(esc_reason.strip()) < 10:
                            st.error("Please provide structured justification for management escalation.")
                        else:
                            inv_service.escalate_to_management(
                                investigation_id=case["id"],
                                escalation_reason=esc_reason,
                                actor_username=user_username
                            )
                            st.success("Case successfully escalated to Management Review Queue!")
                            st.rerun()

                # Note Tab
                with tab_note:
                    note_text = st.text_area("Add Investigation Case Note", key=f"note_{provider_id}")
                    if st.button("Save Note to Case Timeline", key=f"btn_note_{provider_id}"):
                        if note_text.strip():
                            inv_service.add_case_note(case["id"], note_text, user_username, user_role)
                            st.success("Note added to timeline.")
                            st.rerun()

            # Manager Decision Action Console
            if has_permission(user_role, Permission.RECORD_MANAGEMENT_DECISION) and case["status"] == "ESCALATED":
                st.markdown("#### 👔 Executive Management Review & Decision Console")
                st.warning(f"**Escalated by Investigator with Rationale**:\n\n_{case.get('escalation_reason')}_")
                
                mgr_action = st.selectbox(
                    "Management Decision Action",
                    [
                        "ACCEPT_INVESTIGATOR_ASSESSMENT",
                        "REQUEST_ADDITIONAL_CLINICAL_RECORDS",
                        "REFER_TO_PAYMENT_INTEGRITY_AUDIT",
                        "REFER_TO_LAW_ENFORCEMENT_SIU",
                        "CLOSE_NO_FURTHER_ACTION"
                    ],
                    key=f"mgr_act_{provider_id}"
                )
                mgr_reason = st.text_area("Management Decision Rationale (Mandatory)", key=f"mgr_reason_{provider_id}")

                if st.button("Record Executive Management Decision", key=f"btn_mgr_dec_{provider_id}", type="primary"):
                    if len(mgr_reason.strip()) < 10:
                        st.error("Please provide executive decision rationale.")
                    else:
                        from src.services.manager_service import ManagerService
                        mgr_service = ManagerService(db_path=inv_service.db_path)
                        mgr_service.record_management_decision(
                            investigation_id=case["id"],
                            decision_action=mgr_action,
                            reasoning=mgr_reason,
                            actor_username=user_username
                        )
                        st.success("Executive Management Decision recorded successfully!")
                        st.rerun()
        else:
            st.info(f"Provider `{provider_id}` is not currently an open investigation case.")
