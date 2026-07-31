"""PokerTrainer's shared visual system for the Streamlit product shell."""

from __future__ import annotations

from string import Template

import streamlit as st

# Brass-rail desk: warm ink charcoal, chip brass accents, muted felt for wins.
BG = "#121418"
SURFACE = "#1A1E24"
SURFACE_RAISED = "#222831"
SURFACE_STRONG = "#2A313C"
SURFACE_SOFT = "#161A20"
BORDER = "#2E3540"
BORDER_STRONG = "#3D4654"
TEXT = "#F2EFE8"
TEXT_MUTED = "#9AA3B0"
ACCENT = "#D9A441"
ACCENT_HOVER = "#E8B85A"
ACCENT_SOFT = "#2A2314"
POSITIVE = "#4FAE7A"
WARNING = "#E0A04A"
GOLD = "#D9A441"
NEGATIVE = "#D96B5E"


_THEME_CSS = Template(
    r"""
<style>
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=Syne:wght@600;700;800&display=swap");

:root {
    --pt-bg: $BG;
    --pt-surface: $SURFACE;
    --pt-surface-raised: $SURFACE_RAISED;
    --pt-surface-strong: $SURFACE_STRONG;
    --pt-surface-soft: $SURFACE_SOFT;
    --pt-border: $BORDER;
    --pt-border-strong: $BORDER_STRONG;
    --pt-text: $TEXT;
    --pt-muted: $TEXT_MUTED;
    --pt-accent: $ACCENT;
    --pt-accent-hover: $ACCENT_HOVER;
    --pt-accent-soft: $ACCENT_SOFT;
    --pt-positive: $POSITIVE;
    --pt-warning: $WARNING;
    --pt-gold: $GOLD;
    --pt-negative: $NEGATIVE;
    --pt-space-1: 0.25rem;
    --pt-space-2: 0.5rem;
    --pt-space-3: 0.75rem;
    --pt-space-4: 1rem;
    --pt-space-6: 1.5rem;
    --pt-space-8: 2rem;
    --pt-space-12: 3rem;
    --pt-space-16: 4rem;
    --pt-radius-sm: 2px;
    --pt-radius: 5px;
    --pt-radius-lg: 8px;
    --pt-shadow: 0 14px 40px rgba(0, 0, 0, 0.28);
    --pt-font-display: "Syne", "Avenir Next Condensed", "Arial Narrow", sans-serif;
    --pt-font-sans: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", sans-serif;
    --pt-font-mono: "IBM Plex Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    --pt-ease: cubic-bezier(.2, .75, .25, 1);
}

html, body, [data-testid="stAppViewContainer"] {
    color: var(--pt-text);
    background: var(--pt-bg);
    font-family: var(--pt-font-sans);
    font-feature-settings: "tnum" 1, "ss01" 1;
}

html, body { overflow-x: clip; }

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 100% 0%, rgba(217, 164, 65, 0.07), transparent 50%),
        radial-gradient(ellipse 60% 40% at 0% 100%, rgba(79, 174, 122, 0.05), transparent 45%),
        repeating-linear-gradient(
            -18deg,
            transparent,
            transparent 11px,
            rgba(255, 255, 255, 0.015) 11px,
            rgba(255, 255, 255, 0.015) 12px
        ),
        var(--pt-bg);
}

[data-testid="stAppViewContainer"] .block-container {
    width: 100%;
    max-width: 1560px;
    padding: 1rem clamp(1.35rem, 2.8vw, 3rem) 2.25rem;
}

h1, h2, h3, h4, h5 {
    color: var(--pt-text);
    font-family: var(--pt-font-display);
    letter-spacing: -0.02em;
    font-weight: 700;
}

p, label, [data-testid="stCaptionContainer"] { color: var(--pt-muted); }
a { color: var(--pt-accent); }
code, pre, [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }

/* Application chrome */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { opacity: 0.55; }
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #161A20 0%, #121418 100%);
    border-right: 1px solid var(--pt-border);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: var(--pt-space-4) var(--pt-space-3) var(--pt-space-6);
}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    margin-top: -36px;
}
[data-testid="stSidebar"] hr { border-color: var(--pt-border); margin: var(--pt-space-4) 0; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 0.2rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    min-height: 42px;
    padding: 0.55rem 0.72rem;
    border: 1px solid transparent;
    border-radius: var(--pt-radius-sm);
    font-family: var(--pt-font-sans);
    transition: transform 140ms var(--pt-ease), background 140ms ease, border-color 140ms ease, color 140ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: var(--pt-surface-raised);
    color: var(--pt-text);
    transform: translateX(2px);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: var(--pt-accent-soft);
    border-color: #5A4A22;
    color: var(--pt-text);
    box-shadow: inset 3px 0 0 var(--pt-accent);
}
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stWidgetLabel"] { display: none; }

button:focus-visible, input:focus-visible, textarea:focus-visible,
[role="tab"]:focus-visible, [role="radio"]:focus-visible,
[role="combobox"]:focus-visible {
    outline: 2px solid var(--pt-accent) !important;
    outline-offset: 2px !important;
}

/* Brand and page rhythm */
.pt-brand { display: flex; align-items: center; gap: 0.8rem; margin-bottom: var(--pt-space-6); }
.pt-brand .pt-mark {
    display: grid; place-items: center; width: 30px; height: 30px;
    border: 1px solid #8A6B2A; border-radius: 2px; color: #1A1408;
    background: linear-gradient(145deg, #E8B85A 0%, #D9A441 55%, #B8862E 100%);
    font-family: var(--pt-font-display); font-size: 0.62rem;
    font-weight: 800; letter-spacing: -0.04em;
    transform: rotate(45deg);
    box-shadow: 2px 2px 0 #5A4A22;
}
.pt-brand .pt-mark span {
    display: block;
    transform: rotate(-45deg);
}
.pt-brand-copy { display: grid; line-height: 1.05; }
.pt-brand .pt-logo {
    color: var(--pt-text); font-family: var(--pt-font-display);
    font-size: 1.12rem; font-weight: 800; letter-spacing: -0.03em;
}
.pt-brand .pt-logo span { color: var(--pt-accent); }
.pt-brand .pt-brand-kicker {
    color: var(--pt-muted); font-family: var(--pt-font-mono);
    font-size: 0.58rem; letter-spacing: 0.14em; margin-top: 0.28rem;
}

.pt-page-header { margin: 0 0 var(--pt-space-3); max-width: 880px; animation: pt-rise 220ms var(--pt-ease) both; }
.pt-page-header .pt-eyebrow {
    color: var(--pt-accent); font-family: var(--pt-font-mono); font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.16em; margin-bottom: 0.4rem; text-transform: uppercase;
}
.pt-page-header h1 {
    font-size: clamp(1.7rem, 2.45vw, 2.4rem); line-height: 1.12; margin: 0;
    font-family: var(--pt-font-display); font-weight: 800;
}
.pt-page-description { max-width: 740px; margin: 0.45rem 0 0; line-height: 1.5; font-size: 0.9rem; }
.pt-section-header {
    display: flex; align-items: flex-end; justify-content: space-between; gap: var(--pt-space-4);
    margin: var(--pt-space-4) 0 var(--pt-space-2); padding-bottom: var(--pt-space-2);
    border-bottom: 1px solid var(--pt-border);
}
.pt-section-header-copy { min-width: 0; }
.pt-section-header h2 {
    font-size: 1.05rem; margin: 0; font-family: var(--pt-font-display); font-weight: 700;
}
.pt-section-header p { margin: 0.2rem 0 0; font-size: 0.76rem; }
.pt-section-meta { color: var(--pt-muted); font-family: var(--pt-font-mono); font-size: 0.69rem; white-space: nowrap; }

.pt-trust {
    color: var(--pt-muted); font-family: var(--pt-font-mono); font-size: 0.68rem;
    display: flex; align-items: center; gap: 0.5rem; letter-spacing: 0.04em;
}
.pt-trust span {
    width: 7px; height: 7px; border-radius: 1px; background: var(--pt-positive);
    box-shadow: 1px 1px 0 #2A5A40;
}

/* Hero */
.pt-hero {
    position: relative; overflow: hidden; min-height: 224px; padding: clamp(1.4rem, 3vw, 2.5rem);
    border: 1px solid var(--pt-border-strong); border-radius: var(--pt-radius-lg);
    background:
        linear-gradient(115deg, rgba(217, 164, 65, 0.08) 0%, transparent 42%),
        linear-gradient(160deg, #1E242C 0%, #151920 55%, #12161C 100%);
    box-shadow: var(--pt-shadow);
}
.pt-hero::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, var(--pt-accent), transparent 85%);
}
.pt-hero::after {
    content: ""; position: absolute; right: -40px; top: -40px; width: 180px; height: 180px;
    border: 1px solid rgba(217, 164, 65, 0.12); border-radius: 2px;
    transform: rotate(45deg); pointer-events: none;
}
.pt-hero-grid { position: relative; z-index: 1; display: grid; grid-template-columns: minmax(0, .85fr) minmax(380px, 1.15fr); gap: clamp(1.5rem, 4vw, 4rem); align-items: center; }
.pt-hero-copy { max-width: 620px; }
.pt-hero-kicker {
    color: var(--pt-accent); font-family: var(--pt-font-mono); font-size: 0.66rem;
    font-weight: 600; letter-spacing: .16em; text-transform: uppercase;
}
.pt-hero h1 {
    margin: .65rem 0 .75rem; max-width: 650px;
    font-family: var(--pt-font-display);
    font-size: clamp(2.1rem, 4.2vw, 4rem); line-height: .96; font-weight: 800;
}
.pt-hero p { max-width: 560px; margin: 0; color: #B8C0CC; font-size: clamp(.9rem, 1.25vw, 1.02rem); line-height: 1.6; }
.pt-hero-proof { display: flex; flex-wrap: wrap; gap: var(--pt-space-4); margin-top: var(--pt-space-6); color: var(--pt-muted); font-size: .7rem; }
.pt-hero-proof strong { color: var(--pt-text); font-family: var(--pt-font-mono); font-weight: 650; }

/* Shared panels and stats */
.pt-panel, .pt-kpi, .pt-empty {
    border: 1px solid var(--pt-border); border-radius: var(--pt-radius); background: var(--pt-surface);
}
.pt-panel { padding: var(--pt-space-4); }
.pt-panel-title { color: var(--pt-text); font-family: var(--pt-font-display); font-size: .88rem; font-weight: 700; }
.pt-panel-copy { color: var(--pt-muted); font-size: .76rem; line-height: 1.5; margin-top: .3rem; }
.pt-kpi {
    min-height: 112px; padding: 1rem 1.05rem; position: relative; overflow: hidden;
    border-left: 3px solid var(--pt-border-strong);
    transition: transform 160ms var(--pt-ease), border-color 160ms ease, background 160ms ease;
    animation: pt-rise 240ms var(--pt-ease) both;
}
.pt-kpi:hover { transform: translateY(-2px); border-color: var(--pt-border-strong); background: var(--pt-surface-raised); }
.pt-kpi-label {
    color: var(--pt-muted); font-family: var(--pt-font-mono); font-size: 0.64rem;
    letter-spacing: 0.08em; font-weight: 600; text-transform: uppercase;
}
.pt-kpi-value { color: var(--pt-text); font-family: var(--pt-font-mono); font-size: clamp(1.35rem, 2.2vw, 1.75rem); font-weight: 700; margin-top: 0.4rem; letter-spacing: -0.04em; }
.pt-kpi-detail { color: var(--pt-muted); font-size: 0.7rem; margin-top: 0.28rem; }
.pt-kpi-positive { border-left-color: var(--pt-positive); }
.pt-kpi-negative { border-left-color: var(--pt-negative); }
.pt-kpi-warning { border-left-color: var(--pt-warning); }

.pt-coverage { padding: .8rem .9rem; border: 1px solid var(--pt-border); border-radius: var(--pt-radius); background: var(--pt-surface); }
.pt-coverage-track { display: flex; height: 10px; overflow: hidden; border-radius: 1px; background: var(--pt-surface-soft); }
.pt-coverage-segment { display: block; min-width: 2px; }
.pt-coverage-positive { color: var(--pt-positive); background: var(--pt-positive); }
.pt-coverage-neutral { color: var(--pt-muted); background: var(--pt-muted); }
.pt-coverage-warning { color: var(--pt-warning); background: var(--pt-warning); }
.pt-coverage-negative { color: var(--pt-negative); background: var(--pt-negative); }
.pt-coverage-active { color: var(--pt-accent); background: var(--pt-accent); }
.pt-coverage-legend { display: flex; flex-wrap: wrap; gap: .45rem .9rem; margin-top: .7rem; color: var(--pt-muted); font-size: .65rem; }
.pt-coverage-key { display: inline-flex; align-items: center; gap: .32rem; background: transparent; }
.pt-coverage-key i { width: 7px; height: 7px; border-radius: 1px; background: currentColor; }
.pt-coverage-key strong { color: var(--pt-text); font-family: var(--pt-font-mono); }
.pt-coverage-empty { color: var(--pt-muted); font-size: .72rem; }

.pt-frequency { display: grid; gap: .5rem; padding: .8rem .9rem; border: 1px solid var(--pt-border); border-radius: var(--pt-radius); background: var(--pt-surface); }
.pt-frequency-row { display: grid; grid-template-columns: minmax(88px, 1fr) minmax(100px, 2fr) 28px; gap: .65rem; align-items: center; color: var(--pt-muted); font-size: .67rem; }
.pt-frequency-row > span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pt-frequency-track { height: 8px; overflow: hidden; border-radius: 1px; background: var(--pt-surface-soft); }
.pt-frequency-track i { display: block; height: 100%; background: var(--pt-accent); }
.pt-frequency-row strong { color: var(--pt-text); font-family: var(--pt-font-mono); text-align: right; }
.pt-frequency-empty { color: var(--pt-muted); font-size: .72rem; }

.pt-empty { text-align: left; padding: clamp(1.4rem, 4vw, 2.25rem) 1.5rem; background: var(--pt-surface-soft); border-style: solid; border-left: 3px solid var(--pt-accent); }
.pt-empty-marker {
    width: 28px; height: 28px; margin: 0 0 .75rem; border: 1px solid #8A6B2A;
    background: linear-gradient(145deg, #E8B85A, #B8862E); transform: rotate(45deg);
}
.pt-empty h3 { font-size: 1rem; margin: 0; font-family: var(--pt-font-display); }
.pt-empty p { font-size: 0.78rem; line-height: 1.5; margin: 0.4rem 0 0; max-width: 520px; }

.pt-badge {
    display: inline-flex; align-items: center; gap: .35rem; width: fit-content;
    border: 1px solid var(--pt-border); border-radius: 2px; padding: .22rem .5rem;
    color: var(--pt-muted); background: var(--pt-surface-soft);
    font-family: var(--pt-font-mono); font-size: .62rem; font-weight: 600; letter-spacing: .03em;
}
.pt-badge-dot { width: 5px; height: 5px; border-radius: 1px; background: currentColor; }
.pt-badge-positive { color: var(--pt-positive); border-color: #2F5E45; background: #15261D; }
.pt-badge-active { color: var(--pt-accent); border-color: #5A4A22; background: var(--pt-accent-soft); }
.pt-badge-warning { color: var(--pt-warning); border-color: #6A5024; background: #2B2112; }
.pt-badge-negative { color: var(--pt-negative); border-color: #6A3838; background: #2B1719; }

.pt-data-callout { display: flex; align-items: center; justify-content: space-between; gap: var(--pt-space-4); padding: .65rem .85rem; border-left: 3px solid var(--pt-accent); background: var(--pt-surface-soft); color: var(--pt-muted); font-size: .75rem; }
.pt-data-callout strong { color: var(--pt-text); }
.pt-workflow-step { display: grid; grid-template-columns: 30px 1fr; gap: .65rem; align-items: start; margin: .7rem 0 .4rem; }
.pt-workflow-step > span {
    display: grid; place-items: center; width: 28px; height: 28px;
    border: 1px solid var(--pt-border-strong); border-radius: 2px;
    color: var(--pt-accent); background: var(--pt-surface);
    font-family: var(--pt-font-mono); font-size: .58rem; font-weight: 700;
}
.pt-workflow-step strong { display: block; color: var(--pt-text); font-size: .9rem; line-height: 1.3; font-family: var(--pt-font-display); }
.pt-workflow-step p { margin: .2rem 0 0; color: var(--pt-muted); font-size: .76rem; line-height: 1.45; }
.pt-workflow-active > span { color: #1A1408; background: var(--pt-accent); border-color: var(--pt-accent); }
.pt-workflow-complete > span { color: var(--pt-positive); border-color: #2F6C4A; }

/* Native Streamlit surfaces */
[data-testid="stMetric"], [data-testid="stForm"], [data-testid="stExpander"] details {
    background: var(--pt-surface); border: 1px solid var(--pt-border); border-radius: var(--pt-radius);
}
[data-testid="stMetric"] { padding: 0.7rem .85rem; }
[data-testid="stMetricLabel"] { color: var(--pt-muted); letter-spacing: 0.06em; font-size: 0.68rem; font-family: var(--pt-font-mono); }
[data-testid="stMetricValue"] { font-family: var(--pt-font-mono); letter-spacing: -0.04em; }
[data-testid="stForm"] { padding: clamp(.85rem, 1.6vw, 1.1rem); }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border: 1px solid var(--pt-border); border-radius: var(--pt-radius); overflow: hidden; }
[data-testid="stAlert"] { border-radius: var(--pt-radius); border: 1px solid var(--pt-border); }
[data-testid="stFileUploaderDropzone"] { background: var(--pt-surface-soft); border-color: var(--pt-border-strong); border-radius: var(--pt-radius); }
[data-baseweb="input"] > div, [data-baseweb="select"] > div, textarea {
    border-radius: var(--pt-radius-sm) !important; border-color: var(--pt-border-strong) !important; background: #161A20 !important;
}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    min-height: 40px; border-radius: var(--pt-radius-sm); border: 1px solid var(--pt-border-strong); font-weight: 650;
    font-family: var(--pt-font-sans);
    transition: transform 130ms var(--pt-ease), background 130ms ease, border-color 130ms ease, color 130ms ease, box-shadow 130ms ease;
}
.stButton > button p, .stFormSubmitButton > button p, .stDownloadButton > button p {
    color: inherit; margin: 0;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--pt-accent); color: var(--pt-text); transform: translateY(-1px);
    box-shadow: 2px 2px 0 rgba(217, 164, 65, 0.25);
}
.stButton > button:active, .stFormSubmitButton > button:active, .stDownloadButton > button:active { transform: translateY(0) scale(.985); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: var(--pt-accent); border-color: var(--pt-accent); color: #1A1408;
}
.stButton > button[kind="primary"] p, .stFormSubmitButton > button[kind="primary"] p { color: #1A1408; }
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
    background: var(--pt-accent-hover); border-color: var(--pt-accent-hover); color: #1A1408;
    box-shadow: 2px 2px 0 rgba(90, 74, 34, 0.45);
}
.stTabs { min-width: 0; }
.stTabs [data-baseweb="tab-list"] {
    gap: .15rem; border-bottom: 1px solid var(--pt-border); overflow-x: auto; overflow-y: hidden;
    overscroll-behavior-x: contain; scrollbar-width: thin; scrollbar-color: var(--pt-border-strong) transparent;
}
.stTabs [data-baseweb="tab"] {
    flex: 0 0 auto; min-width: max-content; color: var(--pt-muted);
    font-family: var(--pt-font-sans); font-weight: 600; padding-inline: .85rem;
}
.stTabs [aria-selected="true"] { color: var(--pt-text) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--pt-accent); }
[data-testid="stCode"] { border: 1px solid var(--pt-border); border-radius: var(--pt-radius); }

/* Scoped layout behavior */
.st-key-overview_metrics [data-testid="stHorizontalBlock"],
.st-key-session_metrics [data-testid="stHorizontalBlock"] { gap: var(--pt-space-3); }
.st-key-study_workspace > div > [data-testid="stHorizontalBlock"] { align-items: flex-start; }
.st-key-study_workflow_guide {
    margin-bottom: var(--pt-space-3); padding: .85rem 1rem .65rem;
    border: 1px solid var(--pt-border); border-radius: var(--pt-radius);
    border-left: 3px solid var(--pt-accent);
    background: linear-gradient(135deg, rgba(217,164,65,.05), transparent 58%), var(--pt-surface-soft);
}
.st-key-study_workflow_guide h4 { margin: 0; font-family: var(--pt-font-display); }
.st-key-study_hand_navigation {
    margin-bottom: var(--pt-space-4); padding: .75rem .9rem .45rem;
    border: 1px solid var(--pt-border); border-radius: var(--pt-radius);
    background: rgba(22, 26, 32, .72);
}
.st-key-study_hand_navigation [data-testid="stCaptionContainer"] { text-align: center; }
.st-key-overview_actions { margin-top: var(--pt-space-4); }
.st-key-overview_actions [data-testid="stHorizontalBlock"] { gap: var(--pt-space-2); }
.st-key-login_shell { min-height: calc(100vh - 7rem); display: grid; align-items: center; }
.st-key-login_shell > div > [data-testid="stHorizontalBlock"] { align-items: center; }

/* Math review is an analysis console, so keep its primary path dense. */
.st-key-math_workspace [data-testid="stVerticalBlock"] { gap: .62rem; }
.st-key-math_workspace [data-testid="stMetric"] { min-height: 70px; padding: .48rem .65rem; }
.st-key-math_workspace [data-testid="stMetricValue"] { font-size: 1.35rem; }
.st-key-math_workspace [data-testid="stMetricLabel"] { line-height: 1.15; }
.st-key-math_workspace [data-testid="stExpander"] summary { min-height: 38px; }
.st-key-math_workspace [data-testid="stCode"] pre { max-height: 16rem; }
.st-key-math_workspace h5 { margin: .2rem 0 0; }

/* Import stays organized: fewer stacked blocks, normal readable type. */
.st-key-import_workspace > div > [data-testid="stVerticalBlock"] { gap: .7rem; }
.st-key-import_workspace [data-testid="stVerticalBlock"] { gap: .7rem; }
.st-key-import_workspace [data-testid="stExpander"] summary { min-height: 40px; }
.st-key-import_workspace [data-testid="stAlert"] { padding: .55rem .8rem; }
.st-key-import_workspace [data-testid="stFileUploaderDropzone"] { padding: .8rem .9rem; }
.st-key-import_workspace h4, .st-key-import_workspace h5 { margin: .35rem 0 .1rem; }
.st-key-import_session_target {
    padding: .7rem .85rem .6rem !important;
}
.st-key-import_session_target [data-testid="stVerticalBlock"] { gap: .4rem; }
.st-key-import_collect_bar [data-testid="stHorizontalBlock"] { gap: var(--pt-space-2); align-items: end; }
.pt-import-meta {
    display: flex; flex-wrap: wrap; gap: .45rem 1rem; margin: .15rem 0 .35rem;
    padding: .55rem .8rem; border: 1px solid var(--pt-border); border-radius: var(--pt-radius-sm);
    background: var(--pt-surface-soft); color: var(--pt-muted); font-size: .78rem; line-height: 1.4;
}
.pt-import-meta strong { color: var(--pt-text); font-family: var(--pt-font-mono); font-weight: 650; }

/* Frame-to-history reconstruction audit */
.pt-evidence-position {
    min-height: 42px; display: flex; align-items: center; justify-content: center;
    flex-wrap: wrap; gap: .4rem .75rem; padding: .5rem .75rem; text-align: center;
    border: 1px solid var(--pt-border); border-radius: var(--pt-radius);
    background: var(--pt-surface-soft);
    color: var(--pt-muted); font-family: var(--pt-font-mono); font-size: .72rem;
}
.pt-evidence-position > span { white-space: nowrap; }
.pt-evidence-position strong { color: var(--pt-text); }
.pt-evidence-verdict {
    padding: .18rem .48rem; border: 1px solid var(--pt-border-strong);
    border-radius: 2px; font-family: var(--pt-font-mono);
    font-size: .6rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
}
.pt-evidence-verdict.is-correct {
    border-color: rgba(79, 174, 122, .4); background: rgba(79, 174, 122, .1);
    color: var(--pt-positive);
}
.pt-evidence-verdict.is-incorrect {
    border-color: rgba(217, 107, 94, .4); background: rgba(217, 107, 94, .1);
    color: var(--pt-negative);
}
.pt-evidence-verdict.is-unreviewed { color: var(--pt-muted); }
[class*="st-key-evidence_navigation_"] [data-testid="stHorizontalBlock"] {
    gap: var(--pt-space-2);
}
[class*="st-key-evidence_navigation_"] [data-testid="stButton"] {
    margin-top: -.2rem;
}
.pt-evidence-impact {
    display: grid; gap: .28rem; margin-bottom: .55rem; padding: .72rem .8rem;
    border: 1px solid var(--pt-border); border-left: 3px solid var(--pt-accent);
    border-radius: var(--pt-radius-sm); background: var(--pt-surface-soft);
}
.pt-evidence-impact span {
    color: var(--pt-accent); font-family: var(--pt-font-mono); font-size: .58rem; font-weight: 700;
    letter-spacing: .1em; text-transform: uppercase;
}
.pt-evidence-impact strong {
    color: var(--pt-text); font-size: .78rem; font-weight: 650; line-height: 1.4;
}
.pt-evidence-impact small { color: var(--pt-muted); font-size: .66rem; }

@keyframes pt-rise {
    from { opacity: 0; transform: translateY(7px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 1100px) {
    [data-testid="stAppViewContainer"] .block-container { padding-inline: 1.5rem; }
    .pt-hero-grid { grid-template-columns: 1fr; }
    .pt-hero-copy { max-width: 760px; }
    .st-key-study_workspace > div > [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    .st-key-study_workspace > div > [data-testid="stHorizontalBlock"] > div {
        width: 100% !important; min-width: 100% !important; flex: 1 1 100% !important;
    }
    [class*="st-key-evidence_summary_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
    }
    [class*="st-key-evidence_summary_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > div {
        min-width: calc(50% - var(--pt-space-2)) !important;
        flex: 1 1 calc(50% - var(--pt-space-2)) !important;
    }
    [class*="st-key-evidence_comparison_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {
        flex-direction: column;
    }
    [class*="st-key-evidence_comparison_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > div {
        width: 100% !important; min-width: 100% !important; flex: 1 1 auto !important;
    }
}

@media (max-width: 900px) {
    .st-key-study_workflow_guide [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
    }
    .st-key-study_workflow_guide [data-testid="stHorizontalBlock"] > div {
        min-width: calc(50% - var(--pt-space-2)) !important;
        flex: 1 1 calc(50% - var(--pt-space-2)) !important;
    }
    .st-key-overview_metrics [data-testid="stHorizontalBlock"],
    .st-key-session_metrics [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    .st-key-overview_metrics [data-testid="stHorizontalBlock"] > div,
    .st-key-session_metrics [data-testid="stHorizontalBlock"] > div {
        min-width: calc(50% - var(--pt-space-2)) !important;
        flex: 1 1 calc(50% - var(--pt-space-2)) !important;
    }
    .st-key-hand_filters [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    .st-key-hand_filters [data-testid="stHorizontalBlock"] > div {
        min-width: min(100%, 210px) !important;
        flex: 1 1 30% !important;
    }
    .pt-evidence-position {
        justify-content: space-between; padding-inline: .65rem;
    }
}

@media (max-width: 720px) {
    [data-testid="stAppViewContainer"] .block-container { padding: 1rem .9rem 2.5rem; }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { margin-top: -44px; }
    .pt-page-header { margin-bottom: var(--pt-space-3); }
    .pt-page-header h1 { font-size: 1.7rem; }
    .pt-section-header { align-items: flex-start; flex-direction: column; gap: .25rem; margin-top: var(--pt-space-6); }
    .pt-section-meta { white-space: normal; }
    .pt-hero { min-height: 0; padding: 1.2rem; }
    .pt-hero-grid { grid-template-columns: minmax(0, 1fr); gap: 1.2rem; }
    .pt-hero h1 { font-size: clamp(2rem, 11vw, 3rem); }
    .pt-hero-proof { gap: .6rem 1rem; }
    .pt-kpi { min-height: 96px; }
    .st-key-study_workspace > div > [data-testid="stHorizontalBlock"],
    .st-key-hand_filters [data-testid="stHorizontalBlock"] { flex-direction: column; }
    .st-key-login_shell > div > [data-testid="stHorizontalBlock"] { flex-direction: column; }
    .st-key-study_workspace > div > [data-testid="stHorizontalBlock"] > div,
    .st-key-hand_filters [data-testid="stHorizontalBlock"] > div { width: 100% !important; min-width: 100% !important; flex: 1 1 auto !important; }
    .st-key-login_shell > div > [data-testid="stHorizontalBlock"] > div { width: 100% !important; min-width: 100% !important; flex: 1 1 auto !important; }
    .st-key-overview_actions [data-testid="stHorizontalBlock"] > div:last-child { display: none; }
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] { flex-direction: column; }
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] > div {
        width: 100% !important; min-width: 100% !important; flex: 1 1 auto !important;
    }
    [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
        width: 100%; max-width: calc(100vw - 1.8rem); overscroll-behavior-x: contain;
    }
}

@media (max-width: 520px) {
    .st-key-study_workflow_guide [data-testid="stHorizontalBlock"] {
        flex-direction: column;
    }
    .st-key-study_workflow_guide [data-testid="stHorizontalBlock"] > div {
        width: 100% !important; min-width: 100% !important; flex: 1 1 auto !important;
    }
    [data-testid="stAppViewContainer"] .block-container { padding: 0.95rem .75rem 2.25rem; }
    .pt-page-header h1 { font-size: clamp(1.55rem, 8vw, 1.85rem); overflow-wrap: anywhere; }
    .pt-page-description { font-size: .88rem; line-height: 1.5; }
    .pt-hero { padding: 1.05rem; }
    .pt-hero h1 { font-size: clamp(1.95rem, 10.5vw, 2.55rem); line-height: 1; overflow-wrap: anywhere; }
    .pt-hero p { font-size: .86rem; line-height: 1.55; }
    .pt-hero-proof { font-size: .64rem; }
    .pt-kpi { min-height: 88px; padding: .85rem .9rem; }
    .pt-kpi-value { font-size: 1.35rem; }
    .pt-data-callout { align-items: flex-start; flex-wrap: wrap; gap: .35rem .75rem; }
    .st-key-overview_metrics [data-testid="stHorizontalBlock"],
    .st-key-session_metrics [data-testid="stHorizontalBlock"],
    .st-key-overview_actions [data-testid="stHorizontalBlock"] { flex-direction: column; }
    .st-key-overview_metrics [data-testid="stHorizontalBlock"] > div,
    .st-key-session_metrics [data-testid="stHorizontalBlock"] > div,
    .st-key-overview_actions [data-testid="stHorizontalBlock"] > div {
        width: 100% !important; min-width: 100% !important; flex: 1 1 auto !important;
    }
    .st-key-overview_actions [data-testid="stHorizontalBlock"] > div:last-child { display: none; }
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button { min-height: 44px; }
    .stTabs [data-baseweb="tab"] { min-height: 44px; padding-inline: .7rem; }
    [data-testid="stFileUploaderDropzone"] { padding-inline: .75rem; }
    [data-testid="stDataFrame"], [data-testid="stDataEditor"] { max-width: calc(100vw - 1.5rem); }
    .pt-evidence-position { gap: .35rem .55rem; }
    .pt-evidence-position > span:first-child { flex: 1 0 100%; }
    [class*="st-key-evidence_navigation_"] [data-testid="stHorizontalBlock"] {
        flex-direction: row !important; flex-wrap: nowrap !important;
    }
    [class*="st-key-evidence_navigation_"] [data-testid="stHorizontalBlock"] > div {
        width: calc(50% - .25rem) !important; min-width: 0 !important;
        flex: 1 1 calc(50% - .25rem) !important;
    }
    [class*="st-key-evidence_summary_"] > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > div {
        width: 100% !important; min-width: 100% !important; flex: 1 1 100% !important;
    }
    .st-key-math_primary_inputs [data-testid="stHorizontalBlock"],
    .st-key-math_secondary_inputs [data-testid="stHorizontalBlock"],
    .st-key-math_workspace [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
        flex-direction: row; flex-wrap: wrap; gap: .55rem;
    }
    .st-key-math_primary_inputs [data-testid="stHorizontalBlock"] > div,
    .st-key-math_secondary_inputs [data-testid="stHorizontalBlock"] > div,
    .st-key-math_workspace [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div {
        width: calc(50% - .3rem) !important; min-width: calc(50% - .3rem) !important;
        flex: 1 1 calc(50% - .3rem) !important;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .001ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: .001ms !important;
        transition-delay: 0ms !important;
    }
}
</style>
"""
).substitute(
    BG=BG,
    SURFACE=SURFACE,
    SURFACE_RAISED=SURFACE_RAISED,
    SURFACE_STRONG=SURFACE_STRONG,
    SURFACE_SOFT=SURFACE_SOFT,
    BORDER=BORDER,
    BORDER_STRONG=BORDER_STRONG,
    TEXT=TEXT,
    TEXT_MUTED=TEXT_MUTED,
    ACCENT=ACCENT,
    ACCENT_HOVER=ACCENT_HOVER,
    ACCENT_SOFT=ACCENT_SOFT,
    POSITIVE=POSITIVE,
    WARNING=WARNING,
    GOLD=GOLD,
    NEGATIVE=NEGATIVE,
)


def inject_theme() -> None:
    """Inject the global product tokens and Streamlit component overrides."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def brand_header() -> None:
    """Render the compact product mark used in the navigation rail."""
    st.markdown(
        '<div class="pt-brand"><div class="pt-mark" aria-hidden="true"><span>PT</span></div>'
        '<div class="pt-brand-copy"><div class="pt-logo">Poker<span>Trainer</span></div>'
        '<div class="pt-brand-kicker">HAND LAB</div></div></div>',
        unsafe_allow_html=True,
    )
