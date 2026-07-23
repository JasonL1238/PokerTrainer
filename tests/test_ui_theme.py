from inspect import getsource

from poker_tracker.ui.ui_theme import _THEME_CSS, brand_header


def test_theme_uses_product_tokens_and_responsive_breakpoints() -> None:
    assert "--pt-bg: #080B0A" in _THEME_CSS
    assert "--pt-accent: #35D07F" in _THEME_CSS
    assert "max-width: 1560px" in _THEME_CSS
    assert "@media (max-width: 1100px)" in _THEME_CSS
    assert "@media (max-width: 900px)" in _THEME_CSS
    assert "@media (max-width: 720px)" in _THEME_CSS
    assert "@media (max-width: 520px)" in _THEME_CSS


def test_theme_preserves_button_contrast_and_scrollable_tabs() -> None:
    assert ".stButton > button p" in _THEME_CSS
    assert 'button[kind="primary"] p' in _THEME_CSS
    assert "overflow-x: auto" in _THEME_CSS
    assert "overscroll-behavior-x: contain" in _THEME_CSS


def test_theme_respects_reduced_motion() -> None:
    assert "@media (prefers-reduced-motion: reduce)" in _THEME_CSS
    assert "animation-duration: .001ms !important" in _THEME_CSS
    assert "transition-duration: .001ms !important" in _THEME_CSS


def test_shell_uses_text_brand_mark_and_compact_spacing() -> None:
    assert 'aria-hidden="true">PT</div>' in getsource(brand_header)
    assert "padding: 1.15rem" in _THEME_CSS
    assert ".pt-empty-marker" in _THEME_CSS


def test_math_workspace_has_compact_scoped_density() -> None:
    assert ".st-key-math_workspace" in _THEME_CSS
    assert 'min-height: 70px' in _THEME_CSS
    assert 'max-height: 16rem' in _THEME_CSS


def test_sidebar_brand_aligns_with_main_content_before_scroll() -> None:
    assert '[data-testid="stSidebarUserContent"]' in _THEME_CSS
    assert "margin-top: -36px" in _THEME_CSS
    assert "margin-top: -44px" in _THEME_CSS
