"""
==============================================================================
Multi-Agent Healthcare Provider Fraud Intelligence Platform
Streamlit Web Application Entry Point & Role-Based Routing
==============================================================================
"""

import os
import streamlit as st
import pandas as pd
from pathlib import Path

from src.config import APP_TITLE, APP_VERSION, DATABASE_PATH
from src.auth.security import verify_user_credentials, get_user_by_username
from src.auth.rbac import get_role_dashboard_name, Role
from src.services.audit_service import log_audit_event
from src.database.seed_data import seed_database
from src.frontend_utils import render_badge

# Import Views
from frontend.views.user_view import render_user_view
from frontend.views.investigator_view import render_investigator_view
from frontend.views.manager_view import render_manager_view
from frontend.views.admin_view import render_admin_view
from frontend.components.provider_intelligence import render_provider_intelligence

# Configure Page
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
css_path = Path(__file__).parent / "frontend" / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session() -> None:
    """Initializes Streamlit session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = None
    if "global_search_provider" not in st.session_state:
        st.session_state["global_search_provider"] = None

    # Ensure database is initialized
    seed_database(DATABASE_PATH)


def login_user(username: str, role: str, full_name: str, email: str) -> None:
    """Sets session state on successful authentication."""
    st.session_state["authenticated"] = True
    st.session_state["user"] = {
        "username": username,
        "role": role,
        "full_name": full_name,
        "email": email
    }
    st.session_state["active_view"] = role
    log_audit_event(
        username=username,
        role=role,
        action="LOGIN_SUCCESS",
        entity_type="SESSION",
        entity_id=username,
        status="SUCCESS",
        db_path=DATABASE_PATH
    )


def logout_user() -> None:
    """Logs out the active user."""
    if st.session_state.get("user"):
        uname = st.session_state["user"]["username"]
        role = st.session_state["user"]["role"]
        log_audit_event(
            username=uname,
            role=role,
            action="LOGOUT",
            entity_type="SESSION",
            entity_id=uname,
            status="SUCCESS",
            db_path=DATABASE_PATH
        )
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    st.session_state["active_view"] = None
    st.session_state["global_search_provider"] = None
    st.rerun()


def render_login_screen() -> None:
    """Renders the secure login portal with demo role quick-access."""
    st.markdown(
        """
        <div class="portal-header">
            <h1>🏥 Healthcare Provider Fraud Intelligence Platform</h1>
            <p>Multi-Agent Decision Support & Special Investigations System</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### 🔐 Secure System Authentication")
        st.caption("Sign in with your credentials to access your authorized role workspace.")

        with st.form("login_form"):
            username = st.text_input("Username", value="")
            password = st.text_input("Password", type="password", value="")
            submit = st.form_submit_button("Sign In to Portal", type="primary", use_container_width=True)

        if submit:
            user_prof = verify_user_credentials(username, password, db_path=DATABASE_PATH)
            if user_prof:
                login_user(
                    username=user_prof["username"],
                    role=user_prof["role"],
                    full_name=user_prof["full_name"],
                    email=user_prof["email"]
                )
                st.success(f"Welcome back, {user_prof['full_name']}!")
                st.rerun()
            else:
                st.error("Invalid username or password. Please verify your credentials.")
                log_audit_event(
                    username=username,
                    role="UNKNOWN",
                    action="LOGIN_FAILED",
                    entity_type="SESSION",
                    entity_id=username,
                    status="FAILURE",
                    db_path=DATABASE_PATH
                )

    with col2:
        st.markdown("### ⚡ Quick Role Access (Demo Accounts)")
        st.caption("Click any persona below to quickly authenticate as that role:")

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            if st.button("🧑‍💻 **Claims Analyst**\n\n`Role: USER`", use_container_width=True):
                login_user("user", "USER", "Claims Analyst", "analyst@healthcare-audit.gov")
                st.rerun()
            if st.button("🔍 **SIU Investigator**\n\n`Role: INVESTIGATOR`", use_container_width=True):
                login_user("investigator", "INVESTIGATOR", "Senior Investigator", "investigator@healthcare-audit.gov")
                st.rerun()

        with dcol2:
            if st.button("👔 **SIU Manager**\n\n`Role: MANAGER`", use_container_width=True):
                login_user("manager", "MANAGER", "Operations Manager", "manager@healthcare-audit.gov")
                st.rerun()
            if st.button("⚙️ **System Admin**\n\n`Role: ADMIN`", use_container_width=True):
                login_user("admin", "ADMIN", "System Administrator", "admin@healthcare-audit.gov")
                st.rerun()


def render_sidebar() -> None:
    """Renders the persistent application sidebar."""
    user = st.session_state["user"]

    with st.sidebar:
        st.markdown(
            f"""
            <div class="user-badge">
                <div style="font-weight:700; font-size:0.95rem;">{user['full_name']}</div>
                <div style="color:#64748b; font-size:0.8rem;">@{user['username']} • <strong>{user['role']}</strong></div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Global Provider Quick Search
        st.markdown("#### 🔍 Global Provider Search")
        search_input = st.text_input("Provider ID", value="", placeholder="e.g. PRV51069", key="sidebar_search")
        if st.button("Search Provider", use_container_width=True):
            if search_input.strip():
                st.session_state["global_search_provider"] = search_input.strip().upper()
                st.rerun()

        if st.session_state.get("global_search_provider"):
            if st.button("❌ Clear Provider Search", use_container_width=True):
                st.session_state["global_search_provider"] = None
                st.rerun()

        st.markdown("---")

        # Role View Switcher for Admins or Multi-Role Testers
        if user["role"] in {"ADMIN", "MANAGER"}:
            st.markdown("#### 🔄 Switch Workspace View")
            allowed_views = ["USER", "INVESTIGATOR", "MANAGER", "ADMIN"] if user["role"] == "ADMIN" else ["USER", "INVESTIGATOR", "MANAGER"]
            current_idx = allowed_views.index(st.session_state["active_view"]) if st.session_state["active_view"] in allowed_views else 0
            selected_view = st.selectbox(
                "Active Perspective",
                allowed_views,
                index=current_idx,
                key="view_selector"
            )
            st.session_state["active_view"] = selected_view

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True, type="secondary"):
            logout_user()


def main():
    init_session()

    if not st.session_state["authenticated"]:
        render_login_screen()
        return

    render_sidebar()

    user = st.session_state["user"]
    active_role = st.session_state.get("active_view", user["role"])

    # Handle Global Provider Search Override
    if st.session_state.get("global_search_provider"):
        st.markdown(f"### 🔍 Global Provider Intelligence View")
        render_provider_intelligence(
            provider_id=st.session_state["global_search_provider"],
            user_role=active_role,
            user_username=user["username"]
        )
        return

    # Role-Based Routing
    if active_role == "USER":
        render_user_view(username=user["username"], role=active_role, db_path=DATABASE_PATH)
    elif active_role == "INVESTIGATOR":
        render_investigator_view(username=user["username"], role=active_role, db_path=DATABASE_PATH)
    elif active_role == "MANAGER":
        render_manager_view(username=user["username"], role=active_role, db_path=DATABASE_PATH)
    elif active_role == "ADMIN":
        render_admin_view(username=user["username"], role=active_role, db_path=DATABASE_PATH)
    else:
        st.error(f"Unrecognized role workspace: {active_role}")


if __name__ == "__main__":
    main()
