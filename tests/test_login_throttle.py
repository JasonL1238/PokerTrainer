"""Phase 12: repeated sign-in attempts must be bounded.

The bound is process-wide on purpose. Streamlit offers no reliable per-client
identity, so a per-session counter is reset by opening a new tab — which makes
it decoration rather than a limit.
"""

from __future__ import annotations

import pytest

from poker_tracker.ui.login_throttle import (
    LoginThrottle,
    ThrottleSettings,
    reset_shared_throttle,
    shared_throttle,
)


@pytest.fixture
def throttle():
    return LoginThrottle(
        ThrottleSettings(max_failures=3, window_seconds=60.0, cooldown_seconds=30.0)
    )


def test_ordinary_mistyping_does_not_lock_anyone_out(throttle):
    assert throttle.record_failure(now=0.0) == 0.0
    assert throttle.record_failure(now=1.0) == 0.0
    assert throttle.is_locked(now=2.0) is False


def test_the_window_filling_starts_a_cooldown(throttle):
    throttle.record_failure(now=0.0)
    throttle.record_failure(now=1.0)
    wait = throttle.record_failure(now=2.0)
    assert wait == pytest.approx(30.0)
    assert throttle.is_locked(now=2.0) is True
    assert throttle.is_locked(now=31.0) is True
    assert throttle.is_locked(now=33.0) is False


def test_failures_age_out_of_the_window(throttle):
    """Three failures spread over an afternoon are not an attack."""
    throttle.record_failure(now=0.0)
    throttle.record_failure(now=30.0)
    # The first has aged out by now, so this is only the second in the window.
    assert throttle.record_failure(now=70.0) == 0.0
    assert throttle.is_locked(now=70.0) is False


def test_a_correct_password_clears_the_record(throttle):
    throttle.record_failure(now=0.0)
    throttle.record_failure(now=1.0)
    throttle.record_success()
    assert throttle.failure_count(now=2.0) == 0
    assert throttle.record_failure(now=3.0) == 0.0


def test_attempts_during_a_cooldown_do_not_extend_it(throttle):
    """The cooldown is served once; otherwise an attacker locks it forever."""
    for moment in (0.0, 1.0, 2.0):
        throttle.record_failure(now=moment)
    assert throttle.retry_after(now=2.0) == pytest.approx(30.0)

    # Hammering during the lock must not push the release further out.
    for moment in (3.0, 4.0, 5.0):
        throttle.record_failure(now=moment)
    assert throttle.retry_after(now=6.0) <= 30.0
    assert throttle.is_locked(now=40.0) is False


def test_the_ledger_is_shared_across_sessions():
    """A per-session counter would be reset by opening a new tab."""
    reset_shared_throttle()
    try:
        first = shared_throttle()
        second = shared_throttle()
        assert first is second
    finally:
        reset_shared_throttle()


# --- Configuration ----------------------------------------------------------


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("POKER_LOGIN_MAX_FAILURES", "5")
    monkeypatch.setenv("POKER_LOGIN_COOLDOWN_SECONDS", "120")
    settings = ThrottleSettings.from_env()
    assert settings.max_failures == 5
    assert settings.cooldown_seconds == 120.0


@pytest.mark.parametrize("bad", ["nonsense", "-1", "0", ""])
def test_an_invalid_setting_falls_back_rather_than_disabling_the_limit(
    monkeypatch, bad
):
    """A typo in configuration must not silently remove the bound."""
    monkeypatch.setenv("POKER_LOGIN_MAX_FAILURES", bad)
    settings = ThrottleSettings.from_env()
    assert settings.max_failures >= 1
    # Specifically, it must not become unlimited.
    assert settings.max_failures <= 15
