"""The checks that make PLAN Phase 14's exit gate enforceable, and their own proof.

The gate says "all mandatory suites pass without unexplained skips or flaky
reruns". Three words in it were judgements rather than checks:

* an unexplained skip was whatever a reader thought was unexplained;
* a flaky rerun could not happen, because reruns did not happen;
* "mandatory suites" was not measured against the modules they must cover.

Every test below either pins one clause of the skip rule or drives the flake
aggregation over synthetic passes, so the tooling that will fail somebody's
build is itself proven to fail for the right reason. The live audit at the end
is the gate: it runs over the real test tree on every ordinary ``pytest``.

Sample sources are string literals so this file cannot audit itself into a
violation, and so a sample can hold a skip form that would be a defect if it
were real.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from poker_tracker.suite_quality import coverage_report, flake, random_order, skip_policy
from poker_tracker.suite_quality.flake import ERROR, FAILED, PASSED, SKIPPED, PassResult

REPO = Path(__file__).resolve().parents[1]


def _scan(tmp_path: Path, source: str) -> list[skip_policy.SkipVerdict]:
    module = tmp_path / "test_sample.py"
    module.write_text(textwrap.dedent(source), encoding="utf-8")
    return skip_policy.audit(module)


# ---------------------------------------------------------------------------
# Clause 1: a skip has to be conditional
# ---------------------------------------------------------------------------


def test_an_unconditional_skip_is_a_violation_however_good_its_reason(tmp_path: Path) -> None:
    """The reason here names a real external condition and it still fails.

    ``@pytest.mark.skip`` removes the test on every host, so there is no
    condition for the reason to be about. Accepting it because the string reads
    well is precisely how a permanently disabled test hides behind the word
    "explained".
    """
    verdicts = _scan(
        tmp_path,
        """
        import pytest

        @pytest.mark.skip(reason="eval7 not installed on this platform")
        def test_thing():
            pass
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.UNEXPLAINED]
    assert "unconditional" in verdicts[0].detail


def test_a_bare_skip_marker_without_parentheses_is_still_found(tmp_path: Path) -> None:
    """``@pytest.mark.skip`` with no call carries no reason at all.

    It is an ast.Attribute rather than an ast.Call, so a scan that only looks at
    calls misses the one form that cannot even pretend to explain itself.
    """
    verdicts = _scan(
        tmp_path,
        """
        import pytest

        @pytest.mark.skip
        def test_thing():
            pass
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.UNEXPLAINED]
    assert verdicts[0].declaration.form == skip_policy.FORM_MARK_SKIP


def test_an_unguarded_pytest_skip_call_is_a_violation(tmp_path: Path) -> None:
    verdicts = _scan(
        tmp_path,
        """
        import pytest

        def test_thing():
            pytest.skip("numpy not installed")
            assert False
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.UNEXPLAINED]
    assert "no enclosing condition" in verdicts[0].detail


def test_a_guarded_pytest_skip_records_the_guard_it_sits_under(tmp_path: Path) -> None:
    verdicts = _scan(
        tmp_path,
        """
        import os
        import pytest

        def test_thing():
            if os.geteuid() == 0:
                pytest.skip("root ignores directory permissions")
            assert True
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.EXPLAINED]
    assert verdicts[0].declaration.condition == "os.geteuid() == 0"
    assert verdicts[0].declaration.guarded


def test_a_skip_in_an_else_branch_is_conditional_on_that_branch(tmp_path: Path) -> None:
    """An ``else`` is as conditional as the ``if`` it belongs to.

    Recorded as its own case because the guard the report shows has to be the
    negation, not the ``if`` test the reader would otherwise be handed.
    """
    verdicts = _scan(
        tmp_path,
        """
        import shutil
        import pytest

        def test_thing():
            if shutil.which("ffmpeg"):
                assert True
            else:
                pytest.skip("ffmpeg missing from this host")
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.EXPLAINED]
    assert verdicts[0].declaration.condition == "not (shutil.which('ffmpeg'))"


def test_a_skip_inside_an_except_handler_is_conditional_on_the_exception(
    tmp_path: Path,
) -> None:
    verdicts = _scan(
        tmp_path,
        """
        import pytest

        def test_thing():
            try:
                import eval7
            except ImportError:
                pytest.skip("eval7 not importable in this checkout")
            assert eval7
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.EXPLAINED]


# ---------------------------------------------------------------------------
# Clause 2: the reason has to name the external condition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    [
        "flaky",
        "TODO",
        "",
        "broken",
        "  disabled . ",
        # These carry a vocabulary term and would otherwise be accepted, which
        # is what the placeholder list is for: naming a kind of absence is not
        # naming what is absent.
        "missing",
        "unavailable",
        "not installed",
        "wrong platform",
    ],
)
def test_a_disposition_is_not_a_reason(tmp_path: Path, reason: str) -> None:
    verdicts = _scan(
        tmp_path,
        f"""
        import pytest

        @pytest.mark.skipif(True, reason={reason!r})
        def test_thing():
            pass
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.UNEXPLAINED]


def test_a_skipif_with_no_reason_at_all_is_a_violation(tmp_path: Path) -> None:
    verdicts = _scan(
        tmp_path,
        """
        import sys
        import pytest

        @pytest.mark.skipif(sys.platform == "darwin")
        def test_thing():
            pass
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.UNEXPLAINED]
    assert "no reason" in verdicts[0].detail


def test_a_reason_computed_at_runtime_cannot_be_evaluated_by_a_reader(tmp_path: Path) -> None:
    """A reason built from a variable is unreadable in review, so it fails.

    The failure mode this closes is a reason that is technically present and
    says nothing anybody can check without executing the module.
    """
    verdicts = _scan(
        tmp_path,
        """
        import pytest

        WHY = some_lookup()

        @pytest.mark.skipif(True, reason=WHY)
        def test_thing():
            pass
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.UNEXPLAINED]


def test_an_f_string_reason_is_judged_on_its_literal_halves(tmp_path: Path) -> None:
    """The constant part of an f-string is what names the condition.

    Refusing to read f-strings at all would push honest reasons -- which
    interpolate the path that is missing -- into the violation list.
    """
    verdicts = _scan(
        tmp_path,
        """
        import pytest
        from pathlib import Path

        WEIGHTS = Path("cv_lab/models/region_spine_v1.pt")

        @pytest.mark.skipif(
            not WEIGHTS.exists(), reason=f"model weights missing at {WEIGHTS}"
        )
        def test_thing():
            pass
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.EXPLAINED]


def test_a_plausible_reason_naming_no_condition_needs_a_written_review(tmp_path: Path) -> None:
    """This is the sixth skip the exit gate must not let through silently.

    It is conditional, its reason is a whole English sentence, and it still
    fails -- because nothing in it says what the environment is missing. The
    author's options are to say what is absent or to write the review down in
    REVIEWED_SKIPS, and either one puts a human in the path.
    """
    verdicts = _scan(
        tmp_path,
        """
        import pytest

        @pytest.mark.skipif(True, reason="this case does not apply to the new engine")
        def test_thing():
            pass
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.UNEXPLAINED]
    assert "REVIEWED_SKIPS" in verdicts[0].detail


def test_a_registered_reason_carries_its_review_into_the_verdict(tmp_path: Path) -> None:
    reason = "The newest version has no later migration to lack."
    verdicts = _scan(
        tmp_path,
        f"""
        import pytest

        def test_thing(version):
            if version >= 18:
                pytest.skip({reason!r})
            assert True
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.REVIEWED]
    assert verdicts[0].detail == skip_policy.REVIEWED_SKIPS[reason]


def test_importorskip_is_explained_by_the_module_it_names(tmp_path: Path) -> None:
    verdicts = _scan(
        tmp_path,
        """
        import pytest

        def test_thing():
            np = pytest.importorskip("numpy")
            assert np
        """,
    )

    assert [verdict.status for verdict in verdicts] == [skip_policy.EXPLAINED]
    assert "numpy" in verdicts[0].detail


def test_a_condition_term_is_matched_on_word_boundaries_not_substrings() -> None:
    """"rooted" and "rootkit" are not "root", or the vocabulary matches anything."""
    assert skip_policy.names_external_condition("root ignores directory permissions")
    assert skip_policy.names_external_condition("the solver is POSIX-only")
    assert not skip_policy.names_external_condition("uprooted expectations")
    assert not skip_policy.names_external_condition("the numbers disagree")


# ---------------------------------------------------------------------------
# The live gate
# ---------------------------------------------------------------------------


def test_every_skip_in_this_repository_names_an_external_condition() -> None:
    """The gate itself. A seventh skip cannot arrive without answering to it.

    Both trees pytest collects are audited, because ``deploy/tests`` was added
    to testpaths precisely so a bare ``pytest`` would stop ignoring it.
    """
    verdicts = skip_policy.audit(REPO / "tests") + skip_policy.audit(REPO / "deploy" / "tests")
    offenders = skip_policy.violations(verdicts)

    assert not offenders, "\n" + skip_policy.format_report(offenders)


def test_no_reviewed_skip_registration_has_outlived_its_skip() -> None:
    """A registry nobody prunes becomes an allowlist nobody reads."""
    verdicts = skip_policy.audit(REPO / "tests") + skip_policy.audit(REPO / "deploy" / "tests")

    assert skip_policy.stale_registrations(verdicts) == []


# ---------------------------------------------------------------------------
# Shuffled collection order
# ---------------------------------------------------------------------------


def _items(spec: dict[str, int]) -> list[str]:
    return [f"{module}::test_{index}" for module, count in spec.items() for index in range(count)]


def _module_of(node_id: str) -> str:
    return node_id.partition("::")[0]


def test_a_seed_reproduces_its_order_exactly() -> None:
    """A flake report whose seed does not reproduce is an anecdote."""
    items = _items({"a.py": 5, "b.py": 4, "c.py": 6})

    first = random_order.shuffled(items, seed=99, key=_module_of)
    second = random_order.shuffled(items, seed=99, key=_module_of)

    assert first == second
    assert sorted(first) == sorted(items)


def test_modules_stay_contiguous_so_module_fixtures_are_not_torn_apart() -> None:
    """A flat shuffle would rebuild the module-scoped OCR bank thousands of times.

    Keeping each module's tests together is what makes a shuffled pass cost the
    same as an ordered one while still permuting both module order and the
    order inside a module.
    """
    items = _items({"a.py": 5, "b.py": 4, "c.py": 6})

    order = random_order.shuffled(items, seed=7, key=_module_of)

    seen: list[str] = []
    for node_id in order:
        module = _module_of(node_id)
        if not seen or seen[-1] != module:
            assert module not in seen, f"{module} was interrupted and resumed"
            seen.append(module)
    assert sorted(seen) == ["a.py", "b.py", "c.py"]


def test_both_module_order_and_within_module_order_actually_move() -> None:
    items = _items({"a.py": 6, "b.py": 6, "c.py": 6})

    order = random_order.shuffled(items, seed=3, key=_module_of)

    assert [_module_of(node) for node in order] != [_module_of(node) for node in items]
    a_order = [node for node in order if _module_of(node) == "a.py"]
    assert a_order != [node for node in items if _module_of(node) == "a.py"]


def test_a_run_without_a_seed_is_the_run_it_was_before() -> None:
    """Loading the plugin must not reorder anything on its own.

    Every other agent debugging a failure needs ``pytest`` to mean what it
    meant yesterday, so the shuffle is opt-in at the point of use.
    """
    module = REPO / "tests" / "test_icm.py"
    common = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--collect-only", "-q"]

    plain = subprocess.run([*common, str(module)], cwd=REPO, capture_output=True, text=True)
    loaded = subprocess.run(
        # The top-level shim, which is what flake.PLUGIN names. pytest imports a
        # `-p` plugin before any conftest, so naming the in-package module here
        # would load the application before tests/conftest.py redirects the
        # operator's database -- and conftest refuses that run outright.
        [*common, "-p", flake.PLUGIN, str(module)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    assert plain.returncode == 0 and loaded.returncode == 0, loaded.stdout + loaded.stderr
    # Node ids only. The summary line carries a wall-clock duration, and
    # comparing that would make this test the flake it exists to prevent.
    collected = [line for line in plain.stdout.splitlines() if "::" in line]
    assert collected and collected == [
        line for line in loaded.stdout.splitlines() if "::" in line
    ]


# ---------------------------------------------------------------------------
# Flake aggregation
# ---------------------------------------------------------------------------


def _pass(label: str, outcomes: dict[str, str], *, seed: int | None = None) -> PassResult:
    return PassResult(
        label=label, seed=seed, exit_code=0, duration_s=1.0, outcomes=outcomes
    )


def test_a_test_that_passed_once_and_failed_once_is_named_unstable() -> None:
    """This is the whole point: a name, not a count.

    One failure in twelve runs with nothing attached to it is unactionable; the
    same failure with a test id and the seed it appeared under is a bug report.
    """
    report = flake.build_report(
        [
            _pass("ordered", {"tests/a.py::test_x": PASSED, "tests/a.py::test_y": PASSED}),
            _pass("seed1", {"tests/a.py::test_x": FAILED, "tests/a.py::test_y": PASSED}, seed=1),
            _pass("seed2", {"tests/a.py::test_x": PASSED, "tests/a.py::test_y": PASSED}, seed=2),
        ]
    )

    assert [entry["test"] for entry in report.unstable] == ["tests/a.py::test_x"]
    assert report.unstable[0]["outcomes"]["seed1"] == FAILED
    assert not report.stable


def test_a_test_that_failed_in_every_pass_is_broken_not_flaky() -> None:
    """Reporting a real failure as flakiness is how it gets waved through."""
    report = flake.build_report(
        [
            _pass("ordered", {"tests/a.py::test_x": FAILED}),
            _pass("seed1", {"tests/a.py::test_x": ERROR}, seed=1),
        ]
    )

    assert report.unstable == []
    assert [entry["test"] for entry in report.consistently_failing] == ["tests/a.py::test_x"]


def test_a_test_that_only_ran_in_some_passes_is_order_dependent() -> None:
    report = flake.build_report(
        [
            _pass("ordered", {"tests/a.py::test_x": PASSED}),
            _pass("seed1", {}, seed=1),
        ]
    )

    assert report.order_dependent[0]["test"] == "tests/a.py::test_x"
    assert report.order_dependent[0]["absent_from"] == ["seed1"]
    assert not report.stable


def test_a_test_that_skipped_under_one_order_and_ran_under_another_is_reported() -> None:
    """A skip that moves with order is a coverage hole that moves with it."""
    report = flake.build_report(
        [
            _pass("ordered", {"tests/a.py::test_x": PASSED}),
            _pass("seed1", {"tests/a.py::test_x": SKIPPED}, seed=1),
        ]
    )

    assert report.order_dependent[0]["detail"].startswith("skipped in some passes")
    assert report.unstable == []


def test_agreement_across_every_pass_is_the_only_stable_verdict() -> None:
    report = flake.build_report(
        [
            _pass("ordered", {"tests/a.py::test_x": PASSED, "tests/a.py::test_y": SKIPPED}),
            _pass("seed1", {"tests/a.py::test_x": PASSED, "tests/a.py::test_y": SKIPPED}, seed=1),
        ]
    )

    assert report.stable
    assert report.unstable == report.order_dependent == report.consistently_failing == []


def test_junit_parsing_reads_pass_fail_error_and_skip(tmp_path: Path) -> None:
    """pytest's own report format is the record, so no plugin has to be installed."""
    report = tmp_path / "junit.xml"
    report.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <testsuites><testsuite name="pytest" tests="4">
          <testcase classname="tests.test_a" name="test_ok" time="0.1"/>
          <testcase classname="tests.test_a" name="test_bad" time="0.1">
            <failure message="assert 1 == 2">boom</failure>
          </testcase>
          <testcase classname="tests.test_a" name="test_broken" time="0.1">
            <error message="fixture blew up">boom</error>
          </testcase>
          <testcase classname="tests.test_a" name="test_off" time="0.0">
            <skipped type="pytest.skip" message="eval7 not installed"/>
          </testcase>
        </testsuite></testsuites>
        """.strip(),
        encoding="utf-8",
    )

    outcomes = flake.parse_junit(report)

    assert outcomes == {
        "tests/test_a.py::test_ok": PASSED,
        "tests/test_a.py::test_bad": FAILED,
        "tests/test_a.py::test_broken": ERROR,
        "tests/test_a.py::test_off": SKIPPED,
    }


def test_a_test_recorded_twice_keeps_its_worse_outcome(tmp_path: Path) -> None:
    """pytest emits two testcase elements when a phase fails around a green call.

    Both element orders are here on purpose. Keeping the last would lose the
    error in ``test_y`` and keeping the first would lose it in ``test_x``; only
    preferring the worse outcome survives both, and a teardown error that
    disappears is the exact leakage a shuffled pass exists to find.
    """
    report = tmp_path / "junit.xml"
    report.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
        <testsuites><testsuite name="pytest" tests="2">
          <testcase classname="tests.test_a" name="test_x" time="0.1"/>
          <testcase classname="tests.test_a" name="test_x" time="0.1">
            <error message="teardown">boom</error>
          </testcase>
          <testcase classname="tests.test_a" name="test_y" time="0.1">
            <error message="setup">boom</error>
          </testcase>
          <testcase classname="tests.test_a" name="test_y" time="0.1"/>
        </testsuite></testsuites>
        """.strip(),
        encoding="utf-8",
    )

    assert flake.parse_junit(report) == {
        "tests/test_a.py::test_x": ERROR,
        "tests/test_a.py::test_y": ERROR,
    }


def test_a_real_two_pass_hunt_over_a_stable_module_reports_stable(tmp_path: Path) -> None:
    """End to end: two real pytest subprocesses, one shuffled, compared.

    Runs against a two-test throwaway module rather than the suite so it costs a
    second, but it is the same code path the whole-suite hunt uses -- including
    the plugin being loaded by import path in a subprocess, which is the part
    that breaks silently.
    """
    module = tmp_path / "test_stable_sample.py"
    module.write_text(
        "def test_one():\n    assert True\n\n\ndef test_two():\n    assert 2 == 2\n",
        encoding="utf-8",
    )

    report = flake.hunt(
        repo=REPO,
        report_dir=tmp_path / "reports",
        passes=2,
        seeds=[5],
        pytest_args=[str(module)],
    )

    assert [item.exit_code for item in report.passes] == [0, 0]
    assert report.passes[1].seed == 5
    assert set(report.passes[0].outcomes.values()) == {PASSED}
    assert report.stable, flake.format_report(report)


def test_the_hunt_report_serialises_to_reviewable_json(tmp_path: Path) -> None:
    report = flake.build_report(
        [
            _pass("ordered", {"tests/a.py::test_x": PASSED}),
            _pass("seed1", {"tests/a.py::test_x": FAILED}, seed=1),
        ]
    )
    destination = tmp_path / "nested" / "flake_report.json"

    flake.write_report(report, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["stable"] is False
    assert payload["unstable"][0]["test"] == "tests/a.py::test_x"
    assert payload["passes"][1]["seed"] == 1


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------


_COVERAGE_PAYLOAD = {
    "files": {
        "poker_tracker/services/study_readiness.py": {
            "summary": {"num_statements": 400, "covered_lines": 380}
        },
        "poker_tracker/coaching/solver_grounding.py": {
            "summary": {"num_statements": 300, "covered_lines": 30}
        },
        "poker_tracker/coaching/prompts.py": {
            "summary": {"num_statements": 100, "covered_lines": 95}
        },
        # Lower percentage than solver_grounding, an order of magnitude less
        # unexecuted code: the pair that separates the two sort keys.
        "poker_tracker/persistence/legacy_reader.py": {
            "summary": {"num_statements": 40, "covered_lines": 0}
        },
        "poker_tracker/ui/tiny_helper.py": {"summary": {"num_statements": 5, "covered_lines": 0}},
        "cv_lab/scripts/pipeline/region_detections.py": {
            "summary": {"num_statements": 900, "covered_lines": 0}
        },
    }
}


def test_grouping_follows_the_products_own_module_boundaries() -> None:
    groups = coverage_report.group_by_module(
        coverage_report.load_measurements(_COVERAGE_PAYLOAD)
    )
    by_name = {group.name: group for group in groups}

    assert by_name["poker_tracker/coaching"].statements == 400
    assert by_name["poker_tracker/coaching"].covered == 125
    assert round(by_name["poker_tracker/services"].percent, 1) == 95.0
    # cv_lab is research tooling; pooling it in would average away the finding.
    assert "cv_lab" not in "".join(by_name)


def test_under_covered_names_the_biggest_hole_first_not_the_lowest_percentage() -> None:
    """A 300-statement module at 10% matters more than a 40-line reader at 0%.

    Ordering by percentage inverts these two and buries the finding under the
    smaller one; the 5-statement helper is dropped altogether because acting on
    it is noise, not coverage.
    """
    groups = coverage_report.group_by_module(
        coverage_report.load_measurements(_COVERAGE_PAYLOAD)
    )

    flagged = coverage_report.under_covered(groups, threshold=60.0)

    assert [item.path for item in flagged] == [
        "poker_tracker/coaching/solver_grounding.py",
        "poker_tracker/persistence/legacy_reader.py",
    ]
    assert flagged[0].missing == 270


def test_a_core_file_the_payload_never_mentions_is_still_reported(tmp_path: Path) -> None:
    """The zero that goes missing is the zero worth reading.

    Coverage lists a file it never executed only if its walk of the source tree
    reaches it, and that walk descends a directory only when the directory has
    an ``__init__.py``. Four core packages here have none, so a module no test
    imports vanishes from the report instead of appearing at 0% -- the exact
    failure mode where measurement stops measuring and says nothing.
    """
    root = tmp_path / "repo"
    (root / "poker_tracker" / "coaching").mkdir(parents=True)
    (root / "poker_tracker" / "coaching" / "seen.py").write_text("x = 1\n", encoding="utf-8")
    (root / "poker_tracker" / "coaching" / "never_imported.py").write_text(
        "a = 1\nb = 2\nc = 3\n", encoding="utf-8"
    )
    measurements = [
        coverage_report.FileCoverage(
            path="poker_tracker/coaching/seen.py", statements=1, covered=1
        )
    ]

    missing = coverage_report.undiscovered_files(
        measurements,
        repo_root=root,
        packages=("poker_tracker/coaching",),
        single_files=(),
    )

    assert missing == [("poker_tracker/coaching/never_imported.py", 3)]
    assert "never_imported.py" in coverage_report.format_undiscovered(missing)


def test_a_complete_payload_reports_nothing_missing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "poker_tracker" / "coaching").mkdir(parents=True)
    (root / "poker_tracker" / "coaching" / "seen.py").write_text("x = 1\n", encoding="utf-8")
    measurements = [
        coverage_report.FileCoverage(
            path="poker_tracker/coaching/seen.py", statements=1, covered=1
        )
    ]

    missing = coverage_report.undiscovered_files(
        measurements, repo_root=root, packages=("poker_tracker/coaching",), single_files=()
    )

    assert missing == []
    assert "Every core file" in coverage_report.format_undiscovered(missing)


def test_a_payload_that_is_not_a_coverage_report_is_refused() -> None:
    with pytest.raises(ValueError, match="coverage json"):
        coverage_report.load_measurements({"totals": {"percent_covered": 91.0}})


def test_every_core_package_named_for_coverage_still_exists() -> None:
    """The core list is a claim about this repository, so it is checked against it.

    A renamed package would otherwise silently drop out of the report, which is
    the failure mode where coverage measurement quietly stops measuring the
    thing it was added for.
    """
    for package in coverage_report.CORE_PACKAGES:
        assert (REPO / package).is_dir(), package
    for single in coverage_report.SINGLE_FILE_GROUPS:
        assert (REPO / single).is_file(), single


def test_the_report_states_a_total_without_turning_it_into_a_verdict() -> None:
    """Deliberate: this module reports and does not gate.

    A fail_under number rewards executing lines rather than asserting on them,
    and the useful output of coverage here is which module nothing runs.
    """
    groups = coverage_report.group_by_module(
        coverage_report.load_measurements(_COVERAGE_PAYLOAD)
    )

    text = coverage_report.format_report(groups, threshold=60.0)

    assert "core total" in text
    assert "poker_tracker/coaching/solver_grounding.py" in text
    assert "fail" not in text.lower()


def test_the_order_dependence_this_harness_found_stays_repaired() -> None:
    """The flake the repeat harness surfaced, as its own two-test reproduction.

    ``test_the_session_hand_list_performs_the_deletion_the_blockers_name``
    clears ``st.cache_resource`` before its AppTest run and not after, so
    ``app.py``'s cached ``PokerDatabase`` -- opened on a tmp_path file pytest
    has already deleted, which SQLite keeps readable through the open handle --
    used to survive into every later AppTest in the process. The shell smoke
    test then saw a session in that leaked database, ``create_hand_form``
    stopped returning early, and its positional ``app.radio[0]`` found the
    manual-entry form's control instead of the navigation one.

    Run as a child process because the defect is about what one test leaves in
    the interpreter for the next, and this process has already run both.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "tests/test_phase1_adversarial_round13.py::"
            "test_the_session_hand_list_performs_the_deletion_the_blockers_name",
            "tests/test_app_shell.py::test_product_shell_navigation_smoke",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
