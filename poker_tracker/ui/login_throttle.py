"""Bound repeated sign-in attempts.

PLAN.md Phase 12: "Rate-limit or otherwise bound repeated login attempts when
exposed beyond localhost."

Streamlit gives no reliable per-client identity — session state lives in the
browser session, so an attacker scripting fresh sessions gets an unlimited
budget from any per-session counter. The bound here is therefore PROCESS-WIDE:
a sliding window over every failed attempt this server has seen, and a cooldown
once that window fills.

That is a blunt instrument, and deliberately so. It means a determined attacker
can lock out the legitimate operator for the cooldown period. For a
single-operator, local-first application that trade is right: the operator can
restart the process or read the password from their own environment, whereas an
unbounded guess rate against a shared password is the thing that actually loses
the data. A multi-tenant deployment would need per-client limits and is out of
scope for this application's stated posture.

Time is injected rather than read from the clock inside the checks, so the
behavior is testable without sleeping.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass

# Defaults chosen so ordinary mistyping never trips the limit: five failures in
# a minute is a person fumbling a password manager, fifteen is a script.
DEFAULT_MAX_FAILURES = 15
DEFAULT_WINDOW_SECONDS = 60.0
DEFAULT_COOLDOWN_SECONDS = 60.0
# A small constant delay on every failure. Cheap for a human, expensive for a
# loop, and it applies before the window fills rather than only after.
DEFAULT_FAILURE_DELAY_SECONDS = 0.5

ENV_MAX_FAILURES = "POKER_LOGIN_MAX_FAILURES"
ENV_WINDOW_SECONDS = "POKER_LOGIN_WINDOW_SECONDS"
ENV_COOLDOWN_SECONDS = "POKER_LOGIN_COOLDOWN_SECONDS"


@dataclass(frozen=True)
class ThrottleSettings:
    max_failures: int = DEFAULT_MAX_FAILURES
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS

    @classmethod
    def from_env(cls) -> ThrottleSettings:
        def _number(name: str, fallback: float) -> float:
            raw = os.environ.get(name, "").strip()
            if not raw:
                return fallback
            try:
                value = float(raw)
            except ValueError:
                # A typo must not silently disable the limit.
                return fallback
            return value if value > 0 else fallback

        return cls(
            max_failures=max(1, int(_number(ENV_MAX_FAILURES, DEFAULT_MAX_FAILURES))),
            window_seconds=_number(ENV_WINDOW_SECONDS, DEFAULT_WINDOW_SECONDS),
            cooldown_seconds=_number(ENV_COOLDOWN_SECONDS, DEFAULT_COOLDOWN_SECONDS),
        )


class LoginThrottle:
    """A sliding window of failed attempts, with a cooldown once it fills."""

    def __init__(self, settings: ThrottleSettings | None = None) -> None:
        self._settings = settings or ThrottleSettings()
        self._failures: deque[float] = deque()
        self._locked_until: float = 0.0
        self._lock = threading.Lock()

    @property
    def settings(self) -> ThrottleSettings:
        return self._settings

    def _prune(self, now: float) -> None:
        cutoff = now - self._settings.window_seconds
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    def retry_after(self, *, now: float | None = None) -> float:
        """Seconds until sign-in is allowed again; 0.0 when it is allowed now."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            self._prune(moment)
            return max(0.0, self._locked_until - moment)

    def is_locked(self, *, now: float | None = None) -> bool:
        return self.retry_after(now=now) > 0.0

    def record_failure(self, *, now: float | None = None) -> float:
        """Record a wrong password. Returns seconds until the next attempt."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            self._prune(moment)
            self._failures.append(moment)
            if len(self._failures) >= self._settings.max_failures:
                self._locked_until = moment + self._settings.cooldown_seconds
                # Start the next window clean, so the cooldown is served once
                # rather than re-armed by attempts that arrive while locked.
                self._failures.clear()
            return max(0.0, self._locked_until - moment)

    def record_success(self) -> None:
        """A correct password clears the record.

        The window exists to bound guessing. Once somebody proves they know the
        password, holding their earlier typos against them serves no purpose.
        """
        with self._lock:
            self._failures.clear()
            self._locked_until = 0.0

    def failure_count(self, *, now: float | None = None) -> int:
        moment = time.monotonic() if now is None else now
        with self._lock:
            self._prune(moment)
            return len(self._failures)


# One ledger per process, shared across browser sessions on purpose: a
# per-session counter is reset by opening a new tab.
_THROTTLE: LoginThrottle | None = None
_THROTTLE_LOCK = threading.Lock()


def shared_throttle() -> LoginThrottle:
    global _THROTTLE
    with _THROTTLE_LOCK:
        if _THROTTLE is None:
            _THROTTLE = LoginThrottle(ThrottleSettings.from_env())
        return _THROTTLE


def reset_shared_throttle() -> None:
    """Drop the process-wide ledger. For tests and for a deliberate restart."""
    global _THROTTLE
    with _THROTTLE_LOCK:
        _THROTTLE = None
