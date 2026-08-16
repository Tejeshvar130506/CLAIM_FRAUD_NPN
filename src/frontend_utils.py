"""
Frontend Utility Helpers
------------------------
Provides HTML badge renderers, formatters, and UI component helpers.
"""

def render_badge(risk_level: str) -> str:
    """Returns HTML formatted risk badge string."""
    level = (risk_level or "LOW").upper()
    colors = {
        "LOW": ("#dcfce7", "#15803d", "#bbf7d0"),
        "MEDIUM": ("#fef9c3", "#854d0e", "#fde047"),
        "HIGH": ("#ffedd5", "#c2410c", "#fed7aa"),
        "CRITICAL": ("#fee2e2", "#b91c1c", "#fca5a5")
    }
    bg, text, border = colors.get(level, ("#f1f5f9", "#475569", "#cbd5e1"))
    return f'<span style="background-color:{bg}; color:{text}; border:1px solid {border}; padding:3px 8px; border-radius:12px; font-size:0.75rem; font-weight:700;">{level}</span>'
