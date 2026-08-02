"""Tooling that makes the Phase 14 exit gate measurable rather than judged.

The gate reads "all mandatory suites pass without unexplained skips or flaky
reruns". Three of its words were unenforceable before this package existed:

``unexplained``
    ``skip_policy`` reads every skip declaration in the test tree out of the
    source and judges whether its reason names an external condition a reader
    can evaluate. An unconditional skip, a missing reason, a placeholder, or a
    condition nobody has reviewed is a violation.

``flaky``
    ``flake`` runs the suite repeatedly, shuffling collection order between
    passes via the ``random_order`` plugin, and names every test whose result
    was not the same in every pass. A verdict that cannot be reproduced is
    worth less than the name of the test that produced it.

``mandatory suites``
    ``coverage_report`` says which core modules the suite actually executes.
    It reports; it does not gate. A percentage is not the useful output --
    the name of important code nothing runs is.

Nothing here imports the application, so it can run against a tree it is also
measuring.
"""

from __future__ import annotations

__all__ = ["coverage_report", "flake", "random_order", "skip_policy"]
