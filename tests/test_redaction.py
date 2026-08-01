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


def test_redacting_a_json_document_leaves_valid_json():
    """Reports and bundles are redacted whole, then read back by machines."""
    import json

    document = json.dumps(
        {
            "api_key": "sk-ant-secret-value",
            "config": {"password": "hunter2secret", "host": "db.internal"},
            "note": "everything else survives",
        },
        indent=2,
    )
    scrubbed = redact_text(document, include_environment=False)
    parsed = json.loads(scrubbed)

    assert parsed["api_key"] == REDACTED
    assert parsed["config"]["password"] == REDACTED
    assert parsed["config"]["host"] == "db.internal"
    assert parsed["note"] == "everything else survives"


def test_a_numeric_json_secret_is_replaced_with_valid_json():
    """A quoted key means JSON even when the value is not quoted."""
    import json

    scrubbed = redact_text('{"api_key": 1234567, "n": 5}', include_environment=False)
    parsed = json.loads(scrubbed)
    assert parsed["api_key"] == REDACTED
    assert parsed["n"] == 5


def test_session_id_is_not_treated_as_a_credential():
    """It is a foreign key on almost every row here, not a web session token."""
    import json

    scrubbed = redact_text('{"session_id": 42}', include_environment=False)
    assert json.loads(scrubbed)["session_id"] == 42


# --- Adversarial round 2 findings -------------------------------------------


@pytest.mark.parametrize(
    "scheme", ["Bearer", "Basic", "Token", "ApiKey", "Digest", "Negotiate", "SSWS"]
)
def test_every_authorization_scheme_is_redacted_not_just_two(scheme):
    """Enumerating schemes printed "<redacted>" beside an intact credential.

    That reads as scrubbed, which is worse than no redaction at all.
    """
    secret = "4f9a2b7c1d8e6f3a0b5c9d2e0f1a2b3c"
    scrubbed = redact_text(f"Authorization: {scheme} {secret}", include_environment=False)
    assert secret not in scrubbed
    assert REDACTED in scrubbed


@pytest.mark.parametrize(
    "text",
    [
        'password="correct horse battery staple"',
        'password="p,ssw0rd"',
        'password="p;ssw0rd"',
        'password="p)ssw0rd"',
        "password='p,ssw0rd'",
        'api_key="two words here"',
    ],
)
def test_a_quoted_value_containing_a_delimiter_is_still_redacted(text):
    """Quoting a secret used to make redaction strictly worse than leaving it bare.

    The value class stopped at the first delimiter and the closing-quote
    backreference then failed, so the whole match failed and nothing was hidden.
    """
    scrubbed = redact_text(text, include_environment=False)
    assert REDACTED in scrubbed
    for fragment in ("correct horse", "p,ssw0rd", "p;ssw0rd", "p)ssw0rd", "two words"):
        assert fragment not in scrubbed


def test_redact_structure_survives_json_encoding():
    """json.dumps escapes quotes, which stops the assignment pattern matching."""
    import json

    from poker_tracker.safety.redaction import redact_structure

    payload = {
        "notes": 'pasted upstream body: {"api_key": "AIzaSyD9x1QwErTyUiOpAsDf"}',
        "description": "Authorization: Token 4f9a2b7c1d8e6f3a0b5c9d2e",
    }
    serialized = json.dumps(redact_structure(payload, include_environment=False))
    assert "AIzaSyD9x1QwErTyUiOpAsDf" not in serialized
    assert "4f9a2b7c1d8e6f3a0b5c9d2e" not in serialized
    # Still valid JSON afterwards.
    assert json.loads(serialized)


def test_redact_structure_recurses_and_redacts_by_key_name():
    from poker_tracker.safety.redaction import redact_structure

    scrubbed = redact_structure(
        {"outer": [{"APP_PASSWORD": "hunter2", "port": 8501}]},
        include_environment=False,
    )
    assert scrubbed["outer"][0]["APP_PASSWORD"] == REDACTED
    assert scrubbed["outer"][0]["port"] == 8501


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_a_non_positive_limit_bounds_rather_than_slicing_from_the_end(limit):
    """collapsed[:limit-1] counts backwards for limit <= 0."""
    message = safe_error_message(ValueError("x" * 200), limit=limit)
    assert len(message) <= max(0, limit)
