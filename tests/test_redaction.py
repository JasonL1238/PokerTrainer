"""Secret scrubbing for anything persisted or displayed.

The rule these tests defend: redact first, bound second. Bounding first hides a
secret only when it happens to sit past the cutoff, and a client library puts
the key it was using at the front of the message, not the end.
"""

from __future__ import annotations

import pytest

from poker_tracker.safety.redaction import REDACTED, redact_text, safe_error_message


@pytest.mark.parametrize(
    "text",
    [
        "auth failed for sk-ant-api03-AAAABBBBCCCCDDDD",
        "slack said xoxb-123456789012-abcdefghijkl",
        "github token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123",
        "aws key AKIAIOSFODNN7EXAMPLE denied",
        "Authorization: Bearer abcdef0123456789.token",
        "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r",
    ],
)
def test_credential_shapes_are_removed(text):
    scrubbed = redact_text(text, include_environment=False)
    assert REDACTED in scrubbed
    # Nothing that looks like the original token survives.
    for token in text.split():
        if len(token) > 20 and any(c.isdigit() for c in token):
            assert token not in scrubbed


@pytest.mark.parametrize(
    "text",
    [
        "api_key=hunter2secret",
        'config {"password": "hunter2secret"}',
        "APP_PASSWORD = hunter2secret",
        "access_key => hunter2secret",
    ],
)
def test_credential_assignments_are_removed_whatever_the_syntax(text):
    scrubbed = redact_text(text, include_environment=False)
    assert "hunter2secret" not in scrubbed
    assert REDACTED in scrubbed


def test_dsn_credentials_are_removed_but_the_host_survives():
    """A message must still identify what it failed to reach."""
    scrubbed = redact_text(
        "could not connect to postgres://admin:s3cr3tpw@db.internal:5432/study",
        include_environment=False,
    )
    assert "s3cr3tpw" not in scrubbed
    assert "admin" not in scrubbed
    assert "db.internal" in scrubbed


def test_configured_secret_values_are_removed_even_without_a_recognizable_shape(
    monkeypatch,
):
    """"correcthorse" matches no pattern, but it is still the password."""
    monkeypatch.setenv("APP_PASSWORD", "correcthorse")
    scrubbed = redact_text("login rejected: tried correcthorse")
    assert "correcthorse" not in scrubbed
    assert REDACTED in scrubbed


def test_short_environment_values_do_not_scramble_ordinary_prose(monkeypatch):
    """An env var set to "on" must not blank every occurrence of that word."""
    monkeypatch.setenv("POKER_AUTH_MODE", "on")
    scrubbed = redact_text("running on the local machine")
    assert scrubbed == "running on the local machine"


def test_non_secret_environment_variables_are_left_alone(monkeypatch):
    monkeypatch.setenv("POKER_DATA_DIR", "/srv/pokertrainer/data")
    scrubbed = redact_text("writing to /srv/pokertrainer/data")
    assert "/srv/pokertrainer/data" in scrubbed


# --- Ordering: redact before bounding ---------------------------------------


def test_leading_secret_is_redacted_not_merely_truncated():
    exc = ValueError("api_key=sk-ant-leadingsecret " + "x" * 5000)
    message = safe_error_message(exc)
    assert "sk-ant-leadingsecret" not in message
    assert len(message) <= 500


def test_trailing_secret_is_also_gone():
    exc = ValueError("x" * 200 + " api_key=sk-ant-trailingsecret")
    message = safe_error_message(exc)
    assert "sk-ant-trailingsecret" not in message


def test_message_is_single_line_and_bounded():
    exc = ValueError("first line\nsecond line\r\nthird line " + "y" * 5000)
    message = safe_error_message(exc)
    assert "\n" not in message and "\r" not in message
    assert len(message) <= 500


def test_empty_exception_falls_back_to_its_type_name():
    assert safe_error_message(ValueError("")) == "ValueError"
    assert safe_error_message(TimeoutError()) == "TimeoutError"


def test_ordinary_message_survives_unchanged():
    exc = ValueError("Reconstructed timeline failed import validation.")
    assert safe_error_message(exc) == (
        "Reconstructed timeline failed import validation."
    )
