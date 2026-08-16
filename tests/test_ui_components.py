"""
Unit Tests for Streamlit UI Components & View Routers (Phase G)
"""

import os
import unittest
from pathlib import Path

from src.frontend_utils import render_badge
from src.auth.rbac import get_role_dashboard_name
from frontend.components.agent_timeline import render_agent_timeline
from frontend.components.provider_intelligence import render_provider_intelligence
from frontend.views.user_view import render_user_view
from frontend.views.investigator_view import render_investigator_view
from frontend.views.manager_view import render_manager_view
from frontend.views.admin_view import render_admin_view


class TestUIComponents(unittest.TestCase):
    def test_css_stylesheet_exists(self):
        css_path = Path(__file__).parent.parent / "frontend" / "styles.css"
        self.assertTrue(css_path.exists(), "frontend/styles.css must exist.")

    def test_badge_renderer_outputs_valid_html(self):
        levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        for lvl in levels:
            badge_html = render_badge(lvl)
            self.assertIn(lvl, badge_html)
            self.assertIn("border-radius", badge_html)

    def test_role_dashboard_names(self):
        self.assertIn("User", get_role_dashboard_name("USER"))
        self.assertIn("Special Investigations", get_role_dashboard_name("INVESTIGATOR"))
        self.assertIn("Management", get_role_dashboard_name("MANAGER"))
        self.assertIn("Admin", get_role_dashboard_name("ADMIN"))


if __name__ == "__main__":
    unittest.main()
