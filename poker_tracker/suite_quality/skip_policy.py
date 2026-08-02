"""Judge every skip declaration in the test tree against a rule, not a reader.

PLAN Phase 14's exit gate forbids "unexplained skips". Until this module the
judgement was a human reading reason strings, which means a sixth skip could
arrive in a pull request and nothing would notice. The rule below is what that
reader was doing, written down.

A skip is EXPLAINED when both hold:

1.  It is conditional. ``skipif`` with a condition, ``importorskip``, and a
    ``pytest.skip()`` guarded by an ``if`` all leave the test running somewhere.
    A bare ``@pytest.mark.skip`` or an unguarded ``pytest.skip()`` removes the
    test on every host forever, and no reason string makes that an explanation
    of a missing external condition -- there is no condition.
2.  Its reason names the external condition that is absent. "eval7 not
    installed" and "root ignores directory permissions" say what is missing and
    let a reader decide whether the environment is at fault. "flaky", "todo",
    and "" do not.

Clause 2 is matched against ``CONDITION_TERMS`` -- a deliberately small
vocabulary of the things a test can legitimately depend on and not find. A
reason that satisfies nobody's vocabulary is not automatically a violation: it
can be entered in ``REVIEWED_SKIPS`` with a written justification. That is the
escape hatch, and it is the point. Vocabulary or registry, a new skip has to
pass through something a human wrote.

The scan reads source with ``ast`` rather than observing a run because a skip
that never fires on this host is exactly the one a runtime check cannot see.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

EXPLAINED = "explained"
REVIEWED = "reviewed"
UNEXPLAINED = "unexplained"

# The forms this module knows how to read out of source. ``mark_skip`` is the
# unconditional one and is a violation by construction; the rest are judged on
# their reason.
FORM_MARK_SKIP = "mark_skip"
FORM_MARK_SKIPIF = "mark_skipif"
FORM_CALL = "call"
FORM_IMPORTORSKIP = "importorskip"

# Word-boundary tokens that name something outside the code under test. Grouped
# by the kind of absence they describe so an addition has to argue its family.
CONDITION_TERMS: tuple[str, ...] = (
    # A dependency, artifact, or file the checkout may not carry.
    "install",
    "installed",
    "import",
    "importable",
    "module",
    "package",
    "available",
    "unavailable",
    "missing",
    "absent",
    "present",
    "requires",
    "required",
    "needs",
    "checkout",
    "fixture",
    "fixtures",
    "artifact",
    "artifacts",
    "corpus",
    "recording",
    "recordings",
    "weights",
    "database",
    "credentials",
    # The host: kernel, interpreter, or CPU the suite happens to run on.
    "posix",
    "windows",
    "darwin",
    "macos",
    "linux",
    "platform",
    "host",
    "kernel",
    "architecture",
    "interpreter",
    # Filesystem capability and process privilege.
    "filesystem",
    "hardlink",
    "hardlinks",
    "symlink",
    "symlinks",
    "case-sensitive",
    "case-insensitive",
    "mount",
    "permission",
    "permissions",
    "root",
    "euid",
    "uid",
    "disk",
    # Reachability.
    "network",
    "offline",
    "sandbox",
)

# Reasons that look like reasons and are not. Matched on the whole normalised
# string, so "flaky filesystem on this host" is still judged on its condition
# term and only a bare disposition is rejected here.
#
# The second group is the one that earns this list its place: a reason that is
# nothing but a condition word would sail through the vocabulary check while
# saying nothing about WHAT is missing, which is exactly what the exit gate
# means by unexplained.
PLACEHOLDER_REASONS: frozenset[str] = frozenset(
    {
        "",
        "-",
        "n/a",
        "na",
        "skip",
        "skipped",
        "skipping",
        "todo",
        "tbd",
        "wip",
        "fixme",
        "xxx",
        "broken",
        "flaky",
        "fails",
        "failing",
        "fails sometimes",
        "not implemented",
        "not implemented yet",
        "temporarily disabled",
        "disabled",
        "unreliable",
        "slow",
        "too slow",
        # Condition words standing alone, naming nothing.
        "missing",
        "not available",
        "unavailable",
        "absent",
        "not installed",
        "not present",
        "unsupported",
        "requires",
        "needs",
        "permissions",
        "platform",
        "wrong platform",
        "host",
        "wrong host",
    }
)

# Skips whose reason states a real precondition that no vocabulary would
# recognise. Each entry is the review the exit gate asks a human to perform,
# recorded so it happens once instead of every time somebody reads the suite.
#
# Keyed by the exact reason string, so moving the test does not invalidate the
# review and changing what the skip claims does.
REVIEWED_SKIPS: dict[str, str] = {
    "The newest version has no later migration to lack.": (
        "tests/test_migration_matrix.py -- a parametrised case, not a disabled test. The "
        "parameter set is every supported schema version; the newest one has no successor "
        "migration whose additions it could be missing, so there is nothing to assert for "
        "that member alone. The condition is the parameter, which is why it names no "
        "environment. Legitimate: every other member of the parametrisation still runs, and "
        "the case reappears the moment a version is added above it."
    ),
    "synthetic chip did not resolve to a confident digit; run not joined": (
        "tests/test_ocr_readers.py -- explained but structurally unsound, recorded here "
        "rather than silently accepted. The skip condition is computed by calling "
        "TemplateOCR.classify_digit, the code the test exists to guard, so a scoring "
        "regression that pushed the synthetic chip below 0.55 would retire the test instead "
        "of failing it. It is registered rather than failed because the file is not this "
        "module's to change; the fix is to freeze a chip crop whose classification is "
        "pinned, or to assert the score directly. Tracked as a Phase 14 defect."
    ),
}


@dataclass(frozen=True)
class SkipDeclaration:
    """One skip written in the source, wherever it was written."""

    path: Path
    line: int
    form: str
    reason: str | None
    # ``skipif`` conditions and the guard an in-body ``pytest.skip`` sits under,
    # rendered back to source so a report can show what was being tested.
    condition: str | None
    guarded: bool

    @property
    def location(self) -> str:
        return f"{self.path}:{self.line}"


@dataclass(frozen=True)
class SkipVerdict:
    declaration: SkipDeclaration
    status: str
    detail: str


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _literal_text(node: ast.AST | None) -> str | None:
    """Read a reason back out of source, tolerating f-strings.

    An f-string's constant halves are usually the part that names the condition
    ("cv_lab/models/x.npz missing -- {path}"), so keeping them is closer to what
    a reader sees than refusing to judge the reason at all.
    """
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        pieces = [
            part.value
            for part in node.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        ]
        return " ".join(pieces) if pieces else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_text(node.left)
        right = _literal_text(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _keyword(call: ast.Call, *names: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg in names:
            return keyword.value
    return None


def _condition_source(call: ast.Call) -> str | None:
    positional = [arg for arg in call.args]
    if not positional:
        return None
    try:
        return ast.unparse(positional[0])
    except Exception:  # pragma: no cover - unparse handles every node we build
        return None


class _SkipVisitor(ast.NodeVisitor):
    """Collect skip declarations, remembering whether a call sits under an ``if``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.found: list[SkipDeclaration] = []
        self._guard_depth = 0
        self._guard_source: str | None = None

    def visit_If(self, node: ast.If) -> None:
        outer_depth, outer_source = self._guard_depth, self._guard_source
        test_source = ast.unparse(node.test)

        self._guard_depth = outer_depth + 1
        self._guard_source = test_source
        for child in node.body:
            self.visit(child)

        # An ``else`` branch is just as conditional as the body it belongs to.
        self._guard_depth = outer_depth + 1
        self._guard_source = f"not ({test_source})"
        for child in node.orelse:
            self.visit(child)

        self._guard_depth, self._guard_source = outer_depth, outer_source

    def visit_Try(self, node: ast.Try) -> None:
        # A skip in an ``except`` handler is conditional on the exception.
        for child in node.body:
            self.visit(child)
        for handler in node.handlers:
            outer_depth, outer_source = self._guard_depth, self._guard_source
            self._guard_depth += 1
            self._guard_source = f"except {ast.unparse(handler.type) if handler.type else ''}"
            for child in handler.body:
                self.visit(child)
            self._guard_depth, self._guard_source = outer_depth, outer_source
        for child in node.orelse + node.finalbody:
            self.visit(child)

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted_name(node.func)
        if name.endswith("mark.skipif"):
            self.found.append(
                SkipDeclaration(
                    path=self.path,
                    line=node.lineno,
                    form=FORM_MARK_SKIPIF,
                    reason=_literal_text(_keyword(node, "reason")),
                    condition=_condition_source(node),
                    guarded=True,
                )
            )
        elif name.endswith("mark.skip"):
            self.found.append(
                SkipDeclaration(
                    path=self.path,
                    line=node.lineno,
                    form=FORM_MARK_SKIP,
                    reason=_literal_text(
                        _keyword(node, "reason") or (node.args[0] if node.args else None)
                    ),
                    condition=None,
                    guarded=False,
                )
            )
        elif name in {"pytest.skip", "skip"} or name.endswith(".pytest.skip"):
            self.found.append(
                SkipDeclaration(
                    path=self.path,
                    line=node.lineno,
                    form=FORM_CALL,
                    reason=_literal_text(
                        _keyword(node, "reason", "msg") or (node.args[0] if node.args else None)
                    ),
                    condition=self._guard_source,
                    guarded=self._guard_depth > 0,
                )
            )
        elif name.endswith("importorskip"):
            self.found.append(
                SkipDeclaration(
                    path=self.path,
                    line=node.lineno,
                    form=FORM_IMPORTORSKIP,
                    reason=_literal_text(_keyword(node, "reason")),
                    condition=_condition_source(node),
                    guarded=True,
                )
            )
        self.generic_visit(node)


def find_skip_declarations(root: Path) -> list[SkipDeclaration]:
    """Every skip written under ``root``, in path then line order."""
    declarations: list[SkipDeclaration] = []
    for path in sorted(_python_files(root)):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            # A file that does not parse is a louder failure than a skip audit,
            # and pytest collection will say so; do not mask it here.
            continue
        visitor = _SkipVisitor(path)
        visitor.visit(tree)
        found = visitor.found + _bare_skip_marks(tree, path)
        declarations.extend(sorted(found, key=lambda item: item.line))
    return declarations


def _bare_skip_marks(tree: ast.AST, path: Path) -> list[SkipDeclaration]:
    """``@pytest.mark.skip`` written without parentheses.

    It is the same unconditional skip as the called form and cannot even carry a
    reason, so the scan would be trivially evadable without this pass. The
    attribute is only a declaration when nothing calls it -- otherwise it is the
    ``func`` of a Call the visitor already recorded.
    """
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    return [
        SkipDeclaration(
            path=path,
            line=node.lineno,
            form=FORM_MARK_SKIP,
            reason=None,
            condition=None,
            guarded=False,
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and id(node) not in called
        and _dotted_name(node).endswith("mark.skip")
    ]


def _python_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _normalise(reason: str) -> str:
    return " ".join(reason.split()).strip().strip(".").lower()


def names_external_condition(reason: str) -> bool:
    """True when the reason contains a term from the reviewed vocabulary."""
    lowered = reason.lower()
    for term in CONDITION_TERMS:
        index = lowered.find(term)
        while index != -1:
            before = lowered[index - 1] if index else " "
            after_index = index + len(term)
            after = lowered[after_index] if after_index < len(lowered) else " "
            if not before.isalnum() and not after.isalnum():
                return True
            index = lowered.find(term, index + 1)
    return False


def classify(declaration: SkipDeclaration) -> SkipVerdict:
    """Apply the two clauses, in order."""
    if declaration.form == FORM_IMPORTORSKIP:
        # importorskip states the condition in the only argument it takes: the
        # module it could not import. Nothing about it can be unexplained.
        target = declaration.condition or "the requested module"
        return SkipVerdict(declaration, EXPLAINED, f"importorskip names {target}")

    if declaration.form == FORM_MARK_SKIP:
        return SkipVerdict(
            declaration,
            UNEXPLAINED,
            "unconditional skip: the test runs on no host, so no missing external "
            "condition can explain it. Use skipif with the condition, or delete the test.",
        )

    if declaration.form == FORM_CALL and not declaration.guarded:
        return SkipVerdict(
            declaration,
            UNEXPLAINED,
            "pytest.skip() with no enclosing condition: the test runs on no host.",
        )

    reason = declaration.reason
    if reason is None:
        return SkipVerdict(
            declaration,
            UNEXPLAINED,
            "no reason a reader can evaluate (absent, or not a literal string)",
        )

    normalised = _normalise(reason)
    if normalised in PLACEHOLDER_REASONS:
        return SkipVerdict(
            declaration,
            UNEXPLAINED,
            f"reason {reason!r} states a disposition, not the external condition that is missing",
        )

    if names_external_condition(reason):
        return SkipVerdict(declaration, EXPLAINED, "reason names an external condition")

    registered = REVIEWED_SKIPS.get(" ".join(reason.split()))
    if registered is not None:
        return SkipVerdict(declaration, REVIEWED, registered)

    return SkipVerdict(
        declaration,
        UNEXPLAINED,
        f"reason {reason!r} names no external condition from CONDITION_TERMS. Either say "
        "what is missing from the environment, or register the reason in "
        "skip_policy.REVIEWED_SKIPS with the review a reader would otherwise have to do.",
    )


def audit(root: Path) -> list[SkipVerdict]:
    return [classify(declaration) for declaration in find_skip_declarations(root)]


def violations(verdicts: Iterable[SkipVerdict]) -> list[SkipVerdict]:
    return [verdict for verdict in verdicts if verdict.status == UNEXPLAINED]


def stale_registrations(verdicts: Iterable[SkipVerdict]) -> list[str]:
    """Registry entries no skip in the tree claims any more.

    A review that outlived its skip is how a registry rots into an allowlist
    nobody reads, so the audit reports it rather than letting it accumulate.
    """
    claimed = {verdict.detail for verdict in verdicts if verdict.status == REVIEWED}
    return sorted(reason for reason, note in REVIEWED_SKIPS.items() if note not in claimed)


def format_report(verdicts: Iterable[SkipVerdict]) -> str:
    lines: list[str] = []
    ordered = list(verdicts)
    for verdict in ordered:
        declaration = verdict.declaration
        lines.append(
            f"{verdict.status.upper():<11} {declaration.location} [{declaration.form}] "
            f"{declaration.reason!r}"
        )
        if verdict.status != EXPLAINED:
            lines.append(f"            {verdict.detail}")
    counts = {status: 0 for status in (EXPLAINED, REVIEWED, UNEXPLAINED)}
    for verdict in ordered:
        counts[verdict.status] += 1
    lines.append(
        f"\n{len(ordered)} skip declarations: {counts[EXPLAINED]} explained, "
        f"{counts[REVIEWED]} reviewed, {counts[UNEXPLAINED]} unexplained"
    )
    return "\n".join(lines)
