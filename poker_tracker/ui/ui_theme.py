"""PokerTrainer's shared visual system for the Streamlit product shell."""

from __future__ import annotations

from string import Template

import streamlit as st

BG = "#080B0A"
SURFACE = "#0D1311"
SURFACE_RAISED = "#121A17"
SURFACE_STRONG = "#17221D"
SURFACE_SOFT = "#0A100E"
BORDER = "#243129"
BORDER_STRONG = "#33463B"
TEXT = "#F1F3ED"
TEXT_MUTED = "#99A69E"
ACCENT = "#35D07F"
ACCENT_HOVER = "#52E397"
ACCENT_SOFT = "#123323"
POSITIVE = "#42D88A"
WARNING = "#E5A64B"
GOLD = "#D5A84B"
NEGATIVE = "#E45F5F"


_THEME_CSS = Template(
    r"""
<style>
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
    --pt-radius-sm: 4px;
    --pt-radius: 8px;
    --pt-radius-lg: 12px;
    --pt-shadow: 0 18px 54px rgba(0, 0, 0, 0.24);
    --pt-font-sans: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --pt-font-mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
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
        radial-gradient(circle at 77% -15%, rgba(53, 208, 127, 0.075), transparent 35rem),
        linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px),
        var(--pt-bg);
    background-size: auto, 44px 44px, 44px 44px, auto;
}

[data-testid="stAppViewContainer"] .block-container {
    width: 100%;
    max-width: 1560px;
    padding: 1.15rem clamp(1.5rem, 3vw, 3.5rem) 2.5rem;
}

h1, h2, h3, h4, h5 {
    color: var(--pt-text);
    font-family: var(--pt-font-sans);
    letter-spacing: -0.027em;
}

p, label, [data-testid="stCaptionContainer"] { color: var(--pt-muted); }
a { color: var(--pt-accent); }
code, pre, [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }

/* Application chrome */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { opacity: 0.55; }
[data-testid="stSidebar"] {
    background: #090E0C;
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
    border-radius: var(--pt-radius);
    transition: transform 140ms var(--pt-ease), background 140ms ease, border-color 140ms ease, color 140ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: var(--pt-surface-raised);
    color: var(--pt-text);
    transform: translateX(2px);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: var(--pt-accent-soft);
    border-color: #28543d;
    color: var(--pt-text);
    box-shadow: inset 2px 0 0 var(--pt-accent);
}
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stWidgetLabel"] { display: none; }

button:focus-visible, input:focus-visible, textarea:focus-visible,
[role="tab"]:focus-visible, [role="radio"]:focus-visible,
[role="combobox"]:focus-visible {
    outline: 2px solid var(--pt-accent) !important;
    outline-offset: 2px !important;
}

/* Brand and page rhythm */
.pt-brand { display: flex; align-items: center; gap: 0.65rem; margin-bottom: var(--pt-space-6); }
.pt-brand .pt-mark {
    display: grid; place-items: center; width: 32px; height: 32px;
    border: 1px solid #347c55; border-radius: 7px; color: #07110b;
    background: var(--pt-accent); font-family: var(--pt-font-mono); font-size: 0.66rem;
    font-weight: 800; letter-spacing: -0.04em; box-shadow: 0 8px 24px rgba(53, 208, 127, 0.12);
}
.pt-brand-copy { display: grid; line-height: 1.05; }
.pt-brand .pt-logo { color: var(--pt-text); font-size: 1.03rem; font-weight: 780; letter-spacing: -0.025em; }
.pt-brand .pt-logo span { color: var(--pt-accent); }
.pt-brand .pt-brand-kicker { color: var(--pt-muted); font-size: 0.61rem; letter-spacing: 0.08em; margin-top: 0.23rem; }

.pt-page-header { margin: 0 0 var(--pt-space-3); max-width: 880px; animation: pt-rise 220ms var(--pt-ease) both; }
.pt-page-header .pt-eyebrow {
    color: var(--pt-accent); font-size: 0.68rem; font-weight: 760;
    letter-spacing: 0.13em; margin-bottom: var(--pt-space-2); text-transform: uppercase;
}
.pt-page-header h1 { font-size: clamp(1.75rem, 2.5vw, 2.6rem); line-height: 1.04; margin: 0; font-weight: 760; }
.pt-page-description { max-width: 740px; margin: 0.4rem 0 0; line-height: 1.45; font-size: 0.9rem; }
.pt-section-header {
    display: flex; align-items: flex-end; justify-content: space-between; gap: var(--pt-space-4);
    margin: var(--pt-space-4) 0 var(--pt-space-2); padding-bottom: var(--pt-space-2);
    border-bottom: 1px solid var(--pt-border);
}
.pt-section-header-copy { min-width: 0; }
.pt-section-header h2 { font-size: 1rem; margin: 0; font-weight: 720; }
.pt-section-header p { margin: 0.2rem 0 0; font-size: 0.76rem; }
.pt-section-meta { color: var(--pt-muted); font-family: var(--pt-font-mono); font-size: 0.69rem; white-space: nowrap; }

.pt-trust { color: var(--pt-muted); font-size: 0.7rem; display: flex; align-items: center; gap: 0.45rem; }
.pt-trust span { width: 6px; height: 6px; border-radius: 50%; background: var(--pt-positive); box-shadow: 0 0 0 3px #173226; }

/* Hero */
.pt-hero {
    position: relative; overflow: hidden; min-height: 224px; padding: clamp(1.4rem, 3vw, 2.5rem);
    border: 1px solid var(--pt-border-strong); border-radius: var(--pt-radius-lg);
    background:
        radial-gradient(circle at 78% 40%, rgba(53, 208, 127, 0.105), transparent 28rem),
        linear-gradient(130deg, #111a16 0%, #0b110e 58%, #0d1511 100%);
    box-shadow: var(--pt-shadow);
}
.pt-hero::before {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background: linear-gradient(110deg, rgba(255,255,255,0.035), transparent 26%);
}
.pt-hero-grid { position: relative; z-index: 1; display: grid; grid-template-columns: minmax(0, .85fr) minmax(380px, 1.15fr); gap: clamp(1.5rem, 4vw, 4rem); align-items: center; }
.pt-hero-copy { max-width: 620px; }
.pt-hero-kicker { color: var(--pt-accent); font-size: 0.68rem; font-weight: 760; letter-spacing: .14em; text-transform: uppercase; }
.pt-hero h1 { margin: .55rem 0 .75rem; max-width: 650px; font-size: clamp(2rem, 4vw, 4.2rem); line-height: .98; font-weight: 790; }
.pt-hero p { max-width: 560px; margin: 0; color: #B5C0B9; font-size: clamp(.9rem, 1.25vw, 1.02rem); line-height: 1.6; }
.pt-hero-proof { display: flex; flex-wrap: wrap; gap: var(--pt-space-4); margin-top: var(--pt-space-6); color: var(--pt-muted); font-size: .7rem; }
.pt-hero-proof strong { color: var(--pt-text); font-family: var(--pt-font-mono); font-weight: 650; }

/* Shared panels and stats */
.pt-panel, .pt-kpi, .pt-empty, .pt-progress-row {
    border: 1px solid var(--pt-border); border-radius: var(--pt-radius); background: var(--pt-surface);
}
.pt-panel { padding: var(--pt-space-4); }
.pt-panel-title { color: var(--pt-text); font-size: .82rem; font-weight: 700; }
.pt-panel-copy { color: var(--pt-muted); font-size: .76rem; line-height: 1.5; margin-top: .3rem; }
.pt-kpi {
    min-height: 112px; padding: 1rem 1.05rem; position: relative; overflow: hidden;
    transition: transform 160ms var(--pt-ease), border-color 160ms ease, background 160ms ease;
    animation: pt-rise 240ms var(--pt-ease) both;
}
.pt-kpi:hover { transform: translateY(-2px); border-color: var(--pt-border-strong); background: var(--pt-surface-raised); }
.pt-kpi::after { content: ""; position: absolute; inset: auto 0 0; height: 2px; background: var(--pt-border-strong); }
.pt-kpi-label { color: var(--pt-muted); font-size: 0.68rem; letter-spacing: 0.055em; font-weight: 680; }
.pt-kpi-value { color: var(--pt-text); font-family: var(--pt-font-mono); font-size: clamp(1.35rem, 2.2vw, 1.75rem); font-weight: 700; margin-top: 0.35rem; letter-spacing: -0.04em; }
.pt-kpi-detail { color: var(--pt-muted); font-size: 0.7rem; margin-top: 0.28rem; }
.pt-kpi-positive::after { background: var(--pt-positive); }
.pt-kpi-negative::after { background: var(--pt-negative); }
.pt-kpi-warning::after { background: var(--pt-warning); }

.pt-coverage { padding: .8rem .9rem; border: 1px solid var(--pt-border); border-radius: var(--pt-radius); background: var(--pt-surface); }
.pt-coverage-track { display: flex; height: 12px; overflow: hidden; border-radius: 2px; background: var(--pt-surface-soft); }
.pt-coverage-segment { display: block; min-width: 2px; }
.pt-coverage-positive { color: var(--pt-positive); background: var(--pt-positive); }
.pt-coverage-neutral { color: var(--pt-muted); background: var(--pt-muted); }
.pt-coverage-warning { color: var(--pt-warning); background: var(--pt-warning); }
.pt-coverage-negative { color: var(--pt-negative); background: var(--pt-negative); }
.pt-coverage-active { color: var(--pt-accent); background: var(--pt-accent); }
.pt-coverage-legend { display: flex; flex-wrap: wrap; gap: .45rem .9rem; margin-top: .7rem; color: var(--pt-muted); font-size: .65rem; }
.pt-coverage-key { display: inline-flex; align-items: center; gap: .32rem; background: transparent; }
.pt-coverage-key i { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.pt-coverage-key strong { color: var(--pt-text); font-family: var(--pt-font-mono); }
.pt-coverage-empty { color: var(--pt-muted); font-size: .72rem; }

.pt-frequency { display: grid; gap: .5rem; padding: .8rem .9rem; border: 1px solid var(--pt-border); border-radius: var(--pt-radius); background: var(--pt-surface); }
.pt-frequency-row { display: grid; grid-template-columns: minmax(88px, 1fr) minmax(100px, 2fr) 28px; gap: .65rem; align-items: center; color: var(--pt-muted); font-size: .67rem; }
.pt-frequency-row > span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pt-frequency-track { height: 9px; overflow: hidden; border-radius: 2px; background: var(--pt-surface-soft); }
.pt-frequency-track i { display: block; height: 100%; background: var(--pt-accent); }
.pt-frequency-row strong { color: var(--pt-text); font-family: var(--pt-font-mono); text-align: right; }
.pt-frequency-empty { color: var(--pt-muted); font-size: .72rem; }

.pt-empty { text-align: center; padding: clamp(1.4rem, 4vw, 2.25rem) 1.5rem; background: var(--pt-surface-soft); border-style: dashed; }
.pt-empty-marker { width: 24px; height: 2px; margin: 0 auto .65rem; background: var(--pt-accent); }
.pt-empty h3 { font-size: 0.95rem; margin: 0; }
.pt-empty p { font-size: 0.78rem; line-height: 1.5; margin: 0.4rem auto 0; max-width: 520px; }

.pt-badge { display: inline-flex; align-items: center; gap: .35rem; width: fit-content; border: 1px solid var(--pt-border); border-radius: 999px; padding: .2rem .45rem; color: var(--pt-muted); background: var(--pt-surface-soft); font-size: .65rem; font-weight: 680; }
.pt-badge-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
.pt-badge-positive { color: var(--pt-positive); border-color: #235E45; background: #102A1F; }
.pt-badge-active { color: var(--pt-accent); border-color: #28543D; background: var(--pt-accent-soft); }
.pt-badge-warning { color: var(--pt-warning); border-color: #674E26; background: #2B2112; }
.pt-badge-negative { color: var(--pt-negative); border-color: #69383B; background: #2B1719; }

.pt-data-callout { display: flex; align-items: center; justify-content: space-between; gap: var(--pt-space-4); padding: .65rem .85rem; border-left: 2px solid var(--pt-accent); background: var(--pt-surface-soft); color: var(--pt-muted); font-size: .75rem; }
.pt-data-callout strong { color: var(--pt-text); }
.pt-workflow-step { display: grid; grid-template-columns: 30px 1fr; gap: .65rem; align-items: start; margin: 1.15rem 0 .65rem; }
.pt-workflow-step > span { display: grid; place-items: center; width: 28px; height: 28px; border: 1px solid var(--pt-border-strong); border-radius: 50%; color: var(--pt-accent); background: var(--pt-surface); font-family: var(--pt-font-mono); font-size: .58rem; font-weight: 720; }
.pt-workflow-step strong { display: block; color: var(--pt-text); font-size: .9rem; }
.pt-workflow-step p { margin: .2rem 0 0; color: var(--pt-muted); font-size: .73rem; line-height: 1.45; }
.pt-workflow-active > span { color: #07110B; background: var(--pt-accent); border-color: var(--pt-accent); box-shadow: 0 0 0 4px rgba(53,208,127,.08); }
.pt-workflow-complete > span { color: var(--pt-positive); border-color: #2F6C4A; }

/* Native Streamlit surfaces */
[data-testid="stMetric"], [data-testid="stForm"], [data-testid="stExpander"] details {
    background: var(--pt-surface); border: 1px solid var(--pt-border); border-radius: var(--pt-radius);
}
[data-testid="stMetric"] { padding: 0.7rem .85rem; }
[data-testid="stMetricLabel"] { color: var(--pt-muted); letter-spacing: 0.045em; font-size: 0.68rem; }
[data-testid="stMetricValue"] { font-family: var(--pt-font-mono); letter-spacing: -0.04em; }
[data-testid="stForm"] { padding: clamp(.85rem, 1.6vw, 1.1rem); }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border: 1px solid var(--pt-border); border-radius: var(--pt-radius); overflow: hidden; }
[data-testid="stAlert"] { border-radius: var(--pt-radius); border: 1px solid var(--pt-border); }
[data-testid="stFileUploaderDropzone"] { background: var(--pt-surface-soft); border-color: var(--pt-border-strong); border-radius: var(--pt-radius); }
[data-baseweb="input"] > div, [data-baseweb="select"] > div, textarea {
    border-radius: var(--pt-radius-sm) !important; border-color: var(--pt-border-strong) !important; background: #0A100E !important;
}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    min-height: 40px; border-radius: var(--pt-radius-sm); border: 1px solid var(--pt-border-strong); font-weight: 690;
    transition: transform 130ms var(--pt-ease), background 130ms ease, border-color 130ms ease, color 130ms ease;
}
.stButton > button p, .stFormSubmitButton > button p, .stDownloadButton > button p {
    color: inherit; margin: 0;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--pt-accent); color: var(--pt-text); transform: translateY(-1px);
}
.stButton > button:active, .stFormSubmitButton > button:active, .stDownloadButton > button:active { transform: translateY(0) scale(.985); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background: var(--pt-accent); border-color: var(--pt-accent); color: #06110A; }
.stButton > button[kind="primary"] p, .stFormSubmitButton > button[kind="primary"] p { color: #06110A; }
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover { background: var(--pt-accent-hover); border-color: var(--pt-accent-hover); color: #06110A; }
.stTabs { min-width: 0; }
.stTabs [data-baseweb="tab-list"] {
    gap: .15rem; border-bottom: 1px solid var(--pt-border); overflow-x: auto; overflow-y: hidden;
    overscroll-behavior-x: contain; scrollbar-width: thin; scrollbar-color: var(--pt-border-strong) transparent;
}
.stTabs [data-baseweb="tab"] { flex: 0 0 auto; min-width: max-content; color: var(--pt-muted); font-weight: 640; padding-inline: .85rem; }
.stTabs [aria-selected="true"] { color: var(--pt-text) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--pt-accent); }
[data-testid="stCode"] { border: 1px solid var(--pt-border); border-radius: var(--pt-radius); }

/* Scoped layout behavior */
.st-key-overview_metrics [data-testid="stHorizontalBlock"],
.st-key-session_metrics [data-testid="stHorizontalBlock"] { gap: var(--pt-space-3); }
.st-key-study_workspace > div > [data-testid="stHorizontalBlock"] { align-items: flex-start; }
.st-key-study_hand_list, .st-key-study_inspector { position: sticky; top: 1rem; }
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

/* Frame-to-history reconstruction audit */
.pt-evidence-position {
    min-height: 40px; display: grid; place-items: center; text-align: center;
    color: var(--pt-muted); font-family: var(--pt-font-mono); font-size: .72rem;
}
.pt-evidence-position strong { color: var(--pt-text); }
.pt-evidence-impact {
    display: grid; gap: .28rem; margin-bottom: .55rem; padding: .72rem .8rem;
    border: 1px solid var(--pt-border); border-left: 2px solid var(--pt-accent);
    border-radius: var(--pt-radius-sm); background: var(--pt-surface-soft);
}
.pt-evidence-impact span {
    color: var(--pt-accent); font-size: .61rem; font-weight: 760;
    letter-spacing: .075em; text-transform: uppercase;
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
    .st-key-study_workspace > div > [data-testid="stHorizontalBlock"] > div { min-width: min(100%, 310px); flex: 1 1 42% !important; }
    .st-key-study_workspace > div > [data-testid="stHorizontalBlock"] > div:nth-child(3) { min-width: 100%; flex-basis: 100% !important; }
    .st-key-study_hand_list, .st-key-study_inspector { position: static; }
}

@media (max-width: 900px) {
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
    [class*="st-key-evidence_comparison_"] > div > [data-testid="stHorizontalBlock"] {
        flex-direction: column;
    }
    [class*="st-key-evidence_comparison_"] > div > [data-testid="stHorizontalBlock"] > div {
        width: 100% !important; min-width: 100% !important; flex: 1 1 auto !important;
    }
}

@media (max-width: 720px) {
    [data-testid="stAppViewContainer"] { background-size: auto, 32px 32px, 32px 32px, auto; }
    [data-testid="stAppViewContainer"] .block-container { padding: 1.1rem .9rem 3rem; }
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { margin-top: -44px; }
    .pt-page-header { margin-bottom: var(--pt-space-4); }
    .pt-page-header h1 { font-size: 1.72rem; }
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
    [data-testid="stAppViewContainer"] .block-container { padding: 1rem .75rem 2.5rem; }
    .pt-page-header h1 { font-size: clamp(1.58rem, 8vw, 1.85rem); overflow-wrap: anywhere; }
    .pt-page-description { font-size: .86rem; }
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
        '<div class="pt-brand"><div class="pt-mark" aria-hidden="true">PT</div>'
        '<div class="pt-brand-copy"><div class="pt-logo">Poker<span>Trainer</span></div>'
        '<div class="pt-brand-kicker">DECISION LAB</div></div></div>',
        unsafe_allow_html=True,
    )
