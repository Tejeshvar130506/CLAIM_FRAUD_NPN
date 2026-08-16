"""
Agent Execution Timeline Component
----------------------------------
Renders a structured, clean visual timeline showing the sequence of agent executions
without exposing internal confidential chain-of-thought tokens.
"""

import streamlit as st
from typing import Optional
from src.agents.contracts import AgentOrchestrationResult


def render_agent_timeline(orchestration: AgentOrchestrationResult) -> None:
    """
    Renders an agent execution status timeline with timing metrics.
    """
    st.markdown("### 🤖 Multi-Agent Execution Lifecycle")

    col1, col2, col3, col4, col5 = st.columns(5)

    # 1. Perception Agent
    with col1:
        st.success("✅ **1. Perception Agent**\n\n*Data & Keys Validated*")
        if orchestration.perception:
            st.caption(f"Score: {orchestration.perception.quality_report.overall_quality_score:.0f}/100 ({orchestration.perception.execution_time_ms}ms)")

    # 2. Fraud Analysis Agent
    with col2:
        st.success("✅ **2. Fraud Analysis**\n\n*XGBoost + EBM*")
        if orchestration.fraud_analysis:
            st.caption(f"Score: {orchestration.fraud_analysis.risk_score}/100 | Prob: {orchestration.fraud_analysis.fraud_probability*100:.1f}%")

    # 3. Negotiation Agent
    with col3:
        st.success("✅ **3. Negotiation Agent**\n\n*Evidence Challenged*")
        if orchestration.negotiation:
            st.caption(f"Action: {orchestration.negotiation.proposed_action.replace('_', ' ')}")

    # 4. Arbitrator
    with col4:
        st.success("✅ **4. Arbitrator**\n\n*AI Risk Synthesis*")
        if orchestration.arbitrator:
            st.caption(f"Priority: {orchestration.arbitrator.recommended_investigation_priority}")

    # 5. Outcome
    with col5:
        if orchestration.investigation_case_created:
            st.warning(f"🚨 **Candidate Created**\n\n`{orchestration.investigation_case_number}`")
        else:
            st.info("ℹ️ **Assessment Recorded**\n\n*Routine Monitoring*")

    st.caption(f"⏱️ Total Multi-Agent Pipeline Latency: **{orchestration.total_execution_time_ms} ms** | Run ID: `{orchestration.run_id}`")
