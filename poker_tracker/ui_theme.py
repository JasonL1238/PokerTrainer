"""Design system for the Streamlit app.

Flat, professional dark theme (GTO-tool style, minimal gradients).
Base colors live in .streamlit/config.toml; this module refines component
chrome (tabs, metrics, cards, tables) and provides small branded helpers.
"""
from __future__ import annotations

import streamlit as st

# Palette tokens (keep in sync with .streamlit/config.toml).
BG = "#0E1220"
SURFACE = "#161C2E"
SURFACE_RAISED = "#1C2338"
BORDER = "#262E47"
TEXT = "#E8EAF2"
TEXT_MUTED = "#9AA3BC"
ACCENT = "#7C5CFA"
ACCENT_SOFT = "#2A2550"
POSITIVE = "#2FBF71"
NEGATIVE = "#E4586B"

_THEME_CSS = f"""
<style>
/* ---------- Typography & app frame ---------- */
html, body, [data-testid="stAppViewContainer"] {{
    font-feature-settings: "tnum" 1;
}}
[data-testid="stAppViewContainer"] .block-container {{
    padding-top: 2.2rem;
    max-width: 1240px;
}}
h1, h2, h3 {{
    letter-spacing: -0.015em;
}}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] hr {{
    border-color: {BORDER};
}}

/* ---------- Tabs: flat pill navigation ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.25rem;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.3rem;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 7px;
    padding: 0.45rem 0.95rem;
    color: {TEXT_MUTED};
    background: transparent;
    font-weight: 500;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT};
    background: {SURFACE_RAISED};
}}
.stTabs [aria-selected="true"] {{
    background: {ACCENT_SOFT} !important;
    color: {TEXT} !important;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{
    display: none;
}}

/* ---------- Metrics: stat tiles ---------- */
[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.85rem 1rem;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED};
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetricValue"] {{
    font-weight: 600;
}}

/* ---------- Buttons ---------- */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
    border-radius: 8px;
    border: 1px solid {BORDER};
    font-weight: 500;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {{
    border-color: {ACCENT};
    color: {TEXT};
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

/* ---------- Panels: forms, expanders, tables ---------- */
[data-testid="stForm"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
}}
[data-testid="stExpander"] details {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
}}

/* ---------- Alerts: flat, subtle ---------- */
[data-testid="stAlert"] {{
    border-radius: 10px;
    border: 1px solid {BORDER};
}}

/* ---------- Brand header ---------- */
.pt-brand {{
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.1rem;
}}
.pt-brand .pt-logo {{
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: {TEXT};
}}
.pt-brand .pt-logo span {{
    color: {ACCENT};
}}
.pt-brand .pt-tag {{
    color: {TEXT_MUTED};
    font-size: 0.85rem;
}}
.pt-positive {{ color: {POSITIVE}; }}
.pt-negative {{ color: {NEGATIVE}; }}
</style>
"""


def inject_theme() -> None:
    """Apply the app-wide CSS layer. Call once per rerun, right after set_page_config."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def brand_header() -> None:
    """Render the product wordmark and compliance tagline."""
    st.markdown(
        '<div class="pt-brand">'
        '<div class="pt-logo">Poker<span>Trainer</span></div>'
        '<div class="pt-tag">Post-session study &amp; coaching — no real-time assistance</div>'
        "</div>",
        unsafe_allow_html=True,
    )
