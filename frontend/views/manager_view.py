"""
Executive Fraud Operations & Manager Dashboard
----------------------------------------------
Provides SIU leadership with:
1. Executive operational KPIs & risk exposure metrics
2. Escalated case review console
3. Policy-configurable Management Decision execution
4. Team workload distribution & fraud risk analytics
5. Exportable executive compliance reports
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from src.services.manager_service import ManagerService
from src.services.investigation_service import InvestigationService
from frontend.components.provider_intelligence import render_provider_intelligence
from src.database.connection import db_transaction


def render_manager_view(username: str, role: str, db_path: str) -> None:
    """
    Renders the Executive Management Dashboard.
    """
    st.markdown("## 👔 Executive Fraud Operations Management")
    st.caption("Strategic decision-making, escalated case adjudication, risk exposure analytics, and team workload oversight.")

    mgr_service = ManagerService(db_path=db_path)
    inv_service = InvestigationService(db_path=db_path)

    # 1. Executive KPI Summary Cards
    kpis = mgr_service.get_management_kpis()
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    with kcol1:
        st.metric("Total Active Investigations", kpis["total_investigations"])
    with kcol2:
        st.metric("🚨 Escalated to Management", kpis["escalated_cases"])
    with kcol3:
        st.metric("Total High-Risk Exposure ($)", f"${kpis['total_risk_exposure_dollars']:,.2f}")
    with kcol4:
        st.metric("Validated Risk Outcomes", kpis["validated_fraud_risk_cases"])

    st.markdown("---")

    tab_escalated, tab_analytics, tab_team, tab_report = st.tabs([
        "🚨 Escalated Cases Review",
        "📈 Fraud Risk Analytics",
        "👥 Team & Workload Overview",
        "📄 Export Executive Report"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: Escalated Cases Review Console
    # --------------------------------------------------------------------------
    with tab_escalated:
        st.markdown("### 🚨 Cases Escalated for Management Adjudication")
        st.caption("Cases where clinical investigators identified elevated risk but require managerial decision guidance.")

        escalated_cases = mgr_service.get_escalated_cases()
        if escalated_cases:
            df_esc = pd.DataFrame(escalated_cases)
            display_cols = ["case_number", "provider_id", "ai_risk_score", "ai_risk_level", "assigned_to", "escalation_reason", "updated_at"]
            available = [c for c in display_cols if c in df_esc.columns]
            st.dataframe(df_esc[available], use_container_width=True, hide_index=True)

            st.markdown("#### 🔍 Select Escalated Case for Adjudication")
            esc_options = {f"{c['case_number']} — {c['provider_id']} (Risk Score: {c['ai_risk_score']}/100)": c for c in escalated_cases}
            chosen_label = st.selectbox("Choose Case to Adjudicate", list(esc_options.keys()))
            chosen_case = esc_options[chosen_label]

            st.markdown("---")
            render_provider_intelligence(
                provider_id=chosen_case["provider_id"],
                user_role=role,
                user_username=username,
                inv_service=inv_service
            )
        else:
            st.success("✅ Zero escalated cases currently pending management decision.")

    # --------------------------------------------------------------------------
    # TAB 2: Fraud Risk Analytics
    # --------------------------------------------------------------------------
    with tab_analytics:
        st.markdown("### 📈 Provider Risk Distribution Analytics")
        with db_transaction(db_path) as conn:
            prov_rows = conn.execute(
                """
                SELECT provider_id, primary_state, total_claims, total_claim_amount,
                       average_claim_amount, risk_score, risk_level, investigation_priority
                FROM providers
                ORDER BY risk_score DESC
                LIMIT 500
                """
            ).fetchall()

        if prov_rows:
            df_provs = pd.DataFrame([dict(r) for r in prov_rows])

            acol1, acol2 = st.columns(2)
            with acol1:
                # Risk Score Histogram
                fig_hist = px.histogram(
                    df_provs,
                    x="risk_score",
                    nbins=20,
                    color="risk_level",
                    title="Provider Risk Score Distribution",
                    color_discrete_map={"LOW": "#16a34a", "MEDIUM": "#eab308", "HIGH": "#ea580c", "CRITICAL": "#dc2626"}
                )
                fig_hist.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_hist, use_container_width=True)

            with acol2:
                # Top Billing vs Risk Score Scatter
                fig_scat = px.scatter(
                    df_provs,
                    x="total_claims",
                    y="total_claim_amount",
                    color="risk_level",
                    size="risk_score",
                    hover_name="provider_id",
                    title="Claims Volume vs Billing Exposure by Risk Level",
                    color_discrete_map={"LOW": "#16a34a", "MEDIUM": "#eab308", "HIGH": "#ea580c", "CRITICAL": "#dc2626"}
                )
                fig_scat.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_scat, use_container_width=True)

            st.markdown("#### 🔝 Top High-Risk Providers by Exposure")
            top_high = df_provs[df_provs["risk_score"] >= 60].head(15)
            st.dataframe(top_high, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 3: Team Workload Overview
    # --------------------------------------------------------------------------
    with tab_team:
        st.markdown("### 👥 SIU Team Workload & Case Assignments")
        workload = kpis.get("team_workload", [])
        if workload:
            df_work = pd.DataFrame(workload)
            fig_work = px.bar(
                df_work,
                x="member",
                y="cases",
                color="member",
                title="Active Investigation Load per Investigator"
            )
            fig_work.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_work, use_container_width=True)
            st.dataframe(df_work, use_container_width=True, hide_index=True)
        else:
            st.info("No active investigator case assignments recorded.")

    # --------------------------------------------------------------------------
    # TAB 4: Export Executive Report
    # --------------------------------------------------------------------------
    with tab_report:
        st.markdown("### 📄 Export Executive Compliance & Operations Report")
        st.caption("Generate a formatted summary CSV of active cases, risk exposure, and management outcomes.")

        with db_transaction(db_path) as conn:
            all_cases = conn.execute(
                """
                SELECT i.*, p.total_claim_amount, p.total_claims, p.primary_state
                FROM investigations i
                LEFT JOIN providers p ON i.provider_id = p.provider_id
                ORDER BY i.ai_risk_score DESC
                """
            ).fetchall()

        if all_cases:
            df_all = pd.DataFrame([dict(r) for r in all_cases])
            csv_data = df_all.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Executive Investigation Report (CSV)",
                data=csv_data,
                file_name="Executive_Fraud_Intelligence_Report.csv",
                mime="text/csv",
                type="primary"
            )
            st.dataframe(df_all.head(20), use_container_width=True, hide_index=True)
