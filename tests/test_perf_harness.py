"""The performance harness has to be honest about what it did not measure.

Every test here defends one of three rules:

* a number that was not obtained is ``None`` with a reason, never ``0``;
* a number carries the host and the conditions that produced it;
* a comparison never turns a missing or cross-machine value into a verdict.

The measurement paths that need a corpus or a container are exercised against a
synthetic repository rather than skipped, so the code that would run on a real
recording is the code these tests run.
"""

from __future__ import annotations

import json
import resource
import sys
from pathlib import Path

import pytest

from poker_tracker.perf import compare as compare_module
from poker_tracker.perf import probes as probe_module
from poker_tracker.perf.__main__ import (
    EXIT_OK,
    EXIT_REGRESSION,
    EXIT_SESSION_CHECK,
    EXIT_USAGE,
    main,
)
from poker_tracker.perf.compare import (
    IMPROVED,
    INCOMPARABLE_HOST,
    MISSING_BASELINE,
    MISSING_CURRENT,
    REGRESSED,
    UNCHANGED,
    compare_reports,
)
from poker_tracker.perf.harness import (
    HarnessOptions,
    empty_baseline,
    evaluate_session_check,
    run_harness,
    summarize,
)
from poker_tracker.perf.measurement import (
    MEASURED,
    NEVER_MEASURED,
    NOT_TAKEN,
    UNIT_SECONDS,
    MeasurementSpec,
    describe_host,
    host_fingerprint,
    measured,
    not_taken,
)
from poker_tracker.perf.probes import (
    ALL_SPECS,
    RECONSTRUCTION_FPS,
    RECONSTRUCTION_FRAMES,
    RECONSTRUCTION_SECONDS,
    SOLVER_MEDIAN,
    SOLVER_RUNS,
    ProbeContext,
    child_peak_rss_bytes,
    probe_reconstruction,
    probe_solver,
    run_child,
)
from poker_tracker.release_gate import resources as gate_resources

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "poker_tracker" / "perf" / "baselines" / "local_reference.json"

SPEC = MeasurementSpec(
    name="test.metric_seconds",
    unit=UNIT_SECONDS,
    group="imports",
    description="A metric that exists only for these tests.",
)


def _context(tmp_path: Path, **overrides) -> ProbeContext:
    context = ProbeContext(
        repo_root=overrides.pop("repo_root", REPO_ROOT),
        workspace=tmp_path / "workspace",
        db_path=overrides.pop("db_path", tmp_path / "absent.db"),
        data_root=overrides.pop("data_root", tmp_path / "data"),
        manifest_path=overrides.pop("manifest_path", tmp_path / "manifest.json"),
        **overrides,
    )
    context.prepare()
    return context


# --------------------------------------------------------------------------
# Withheld numbers
# --------------------------------------------------------------------------


def test_a_withheld_measurement_is_null_and_never_zero() -> None:
    entry = not_taken(SPEC, reason="no corpus on this machine", probe="unit").to_dict()

    assert entry["status"] == NOT_TAKEN
    assert entry["value"] is None
    assert entry["not_taken_reason"] == "no corpus on this machine"
    assert json.loads(json.dumps(entry))["value"] is None


def test_a_measured_zero_is_distinguishable_from_a_withheld_measurement() -> None:
    zero = measured(SPEC, value=0, probe="unit")
    withheld = not_taken(SPEC, reason="probe did not run", probe="unit")

    assert zero.taken and zero.value == 0
    assert not withheld.taken and withheld.value is None
    assert zero.to_dict()["status"] != withheld.to_dict()["status"]


def test_a_measured_value_may_not_be_null() -> None:
    with pytest.raises(ValueError, match="cannot be None"):
        measured(SPEC, value=None, probe="unit")  # type: ignore[arg-type]


def test_a_measured_value_may_not_be_a_flag() -> None:
    with pytest.raises(ValueError, match="not a flag"):
        measured(SPEC, value=True, probe="unit")  # type: ignore[arg-type]


def test_a_withheld_measurement_must_say_why() -> None:
    with pytest.raises(ValueError, match="needs a reason"):
        not_taken(SPEC, reason="   ", probe="unit")


# --------------------------------------------------------------------------
# Report shape
# --------------------------------------------------------------------------


def test_a_report_carries_every_declared_metric_even_when_nothing_ran(tmp_path) -> None:
    report = run_harness(
        HarnessOptions(workspace=tmp_path / "ws", groups=(), db_path=tmp_path / "none.db")
    ).to_dict()

    names = [entry["name"] for entry in report["measurements"]]
    assert names == sorted(spec.name for spec in ALL_SPECS)
    assert names == sorted(names), "measurements are emitted in a stable order"
    withheld = [e for e in report["measurements"] if e["status"] == NOT_TAKEN]
    assert all(e["value"] is None and e["not_taken_reason"] for e in withheld)
    assert report["summary"]["measurements_taken"] + report["summary"][
        "measurements_not_taken"
    ] == len(ALL_SPECS)


def test_an_unrequested_group_says_it_was_not_requested(tmp_path) -> None:
    report = run_harness(
        HarnessOptions(workspace=tmp_path / "ws", groups=(), db_path=tmp_path / "none.db")
    ).to_dict()

    reconstruction = next(
        e for e in report["measurements"] if e["name"] == RECONSTRUCTION_SECONDS.name
    )
    assert "was not requested" in reconstruction["not_taken_reason"]


def test_every_measurement_carries_its_host_and_its_conditions(tmp_path) -> None:
    report = run_harness(
        HarnessOptions(workspace=tmp_path / "ws", groups=(), db_path=tmp_path / "none.db")
    ).to_dict()

    host = report["host"]
    for field in ("system", "machine", "cpu_count", "python", "platform"):
        assert host[field] is not None, f"host is missing {field}"
    assert report["host_fingerprint"] == host_fingerprint(host)
    for entry in report["measurements"]:
        assert entry["conditions"].get("probe"), f"{entry['name']} names no probe"


def test_two_reports_of_the_same_harness_have_the_same_skeleton(tmp_path) -> None:
    first = run_harness(
        HarnessOptions(workspace=tmp_path / "a", groups=(), db_path=tmp_path / "none.db")
    ).to_dict()
    second = run_harness(
        HarnessOptions(workspace=tmp_path / "b", groups=(), db_path=tmp_path / "none.db")
    ).to_dict()

    def skeleton(report: dict) -> object:
        return (
            sorted(report.keys()),
            [(m["name"], m["unit"], m["group"]) for m in report["measurements"]],
            [c["name"] for c in report["checks"]],
        )

    assert skeleton(first) == skeleton(second)


def test_a_broken_probe_withholds_its_group_instead_of_losing_the_run(
    tmp_path, monkeypatch
) -> None:
    def explode(_context):
        raise RuntimeError("probe is broken")

    monkeypatch.setitem(probe_module.PROBES, "solver", explode)
    report = run_harness(
        HarnessOptions(
            workspace=tmp_path / "ws", groups=("solver",), db_path=tmp_path / "none.db"
        )
    ).to_dict()

    entry = next(e for e in report["measurements"] if e["name"] == SOLVER_RUNS.name)
    assert entry["status"] == NOT_TAKEN
    assert "probe is broken" in entry["not_taken_reason"]
    assert len(report["measurements"]) == len(ALL_SPECS)


def test_a_probe_that_reports_only_part_of_its_group_has_the_rest_filled_in() -> None:
    partial = [measured(probe_module.UI_FIRST_RENDER, value=0.2, probe="unit")]

    completed = probe_module.complete_group("ui", partial, probe="unit")

    by_name = {m.spec.name: m for m in completed}
    assert set(by_name) == {spec.name for spec in probe_module.GROUP_SPECS["ui"]}
    assert by_name[probe_module.UI_FIRST_RENDER.name].value == 0.2
    assert by_name[probe_module.UI_PEAK_RSS.name].value is None
    assert "no record" in by_name[probe_module.UI_PEAK_RSS.name].not_taken_reason


def test_a_metric_that_no_probe_reported_is_a_harness_error(tmp_path, monkeypatch) -> None:
    """A silently absent metric would read as "nothing to see"; it must raise."""
    monkeypatch.setattr(
        probe_module,
        "GROUP_SPECS",
        {**probe_module.GROUP_SPECS, "solver": ()},
    )
    with pytest.raises(RuntimeError, match="produced no record for"):
        run_harness(
            HarnessOptions(
                workspace=tmp_path / "ws", groups=(), db_path=tmp_path / "none.db"
            )
        )


# --------------------------------------------------------------------------
# Operator state is never written
# --------------------------------------------------------------------------


def test_probe_children_are_redirected_out_of_the_operators_state(tmp_path) -> None:
    context = _context(tmp_path)
    env = probe_module.child_env(context)

    workspace = str(context.workspace)
    assert env["POKER_DB_PATH"].startswith(workspace)
    assert env["POKER_DATA_DIR"].startswith(workspace)
    assert env["TMPDIR"].startswith(workspace)


def test_the_solver_probe_opens_the_database_read_only(tmp_path) -> None:
    absent = tmp_path / "not_here.db"
    results = probe_solver(_context(tmp_path, db_path=absent))

    assert not absent.exists(), "a read-only probe must not create a database"
    assert {m.spec.name for m in results} == {
        spec.name for spec in probe_module.GROUP_SPECS["solver"]
    }
    assert all(not m.taken for m in results)


def test_the_probes_database_handle_cannot_write(tmp_path) -> None:
    """The harness reads an operator's live library; the handle must refuse writes."""
    import sqlite3

    db_path = tmp_path / "library.db"
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE solver_runs (status TEXT, runtime_seconds REAL)")
    seed.commit()
    seed.close()

    connection = probe_module.open_readonly(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO solver_runs VALUES ('succeeded', 1.0)")
    finally:
        connection.close()


def test_the_solver_probe_counts_zero_runs_but_withholds_the_percentiles(
    tmp_path,
) -> None:
    import sqlite3

    db_path = tmp_path / "library.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE solver_runs (status TEXT, runtime_seconds REAL)")
    connection.commit()
    connection.close()

    results = {m.spec.name: m for m in probe_solver(_context(tmp_path, db_path=db_path))}

    assert results[SOLVER_RUNS.name].value == 0
    assert results[SOLVER_RUNS.name].taken
    assert results[SOLVER_MEDIAN.name].value is None
    assert "no completed solver run" in results[SOLVER_MEDIAN.name].not_taken_reason


def test_the_solver_probe_summarizes_recorded_runtimes(tmp_path) -> None:
    import sqlite3

    db_path = tmp_path / "library.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE solver_runs (status TEXT, runtime_seconds REAL)")
    connection.executemany(
        "INSERT INTO solver_runs VALUES (?, ?)",
        [("succeeded", 10.0), ("succeeded", 30.0), ("succeeded", 20.0), ("failed", 999.0)],
    )
    connection.commit()
    connection.close()

    results = {m.spec.name: m for m in probe_solver(_context(tmp_path, db_path=db_path))}

    assert results[SOLVER_RUNS.name].value == 3
    assert results[SOLVER_MEDIAN.name].value == 20.0
    assert results[SOLVER_MEDIAN.name].conditions["access"] == "sqlite mode=ro"


# --------------------------------------------------------------------------
# Child probes
# --------------------------------------------------------------------------


def test_a_child_reports_its_own_wall_time_and_peak_memory(tmp_path) -> None:
    result = run_child(
        _context(tmp_path), name="sleeper", body="_pt.sleep(0.05)\n", timeout=60
    )

    assert result.ok
    assert result.seconds is not None and result.seconds >= 0.05
    assert result.peak_rss_bytes and result.peak_rss_bytes > 1_000_000
    assert result.log_path.is_file()


def test_a_child_that_times_out_is_withheld_rather_than_recorded_as_zero(
    tmp_path,
) -> None:
    result = run_child(
        _context(tmp_path), name="hang", body="_pt.sleep(30)\n", timeout=1.0
    )

    assert not result.ok
    assert result.seconds is None
    assert "timeout" in (result.error or "")


def test_a_child_that_fails_reports_the_failure_not_a_number(tmp_path) -> None:
    result = run_child(
        _context(tmp_path),
        name="broken",
        body="raise SystemExit('weights are missing')\n",
        timeout=60,
    )

    assert not result.ok
    assert result.seconds is None
    assert "weights are missing" in (result.error or "")


def test_child_peak_memory_uses_the_same_unit_rule_as_the_release_gate() -> None:
    """One rule, two call sites: a child reports raw ru_maxrss and cannot import
    the gate's converter, so this pins the two conversions together."""
    usage = resource.getrusage(resource.RUSAGE_SELF)

    assert child_peak_rss_bytes(usage.ru_maxrss, sys.platform) == gate_resources._maxrss_bytes(
        usage
    )


# --------------------------------------------------------------------------
# Reconstruction and throughput
# --------------------------------------------------------------------------


_FAKE_PIPELINE = """
import json
import sys
import time

argv = sys.argv
out = argv[argv.index("--out") + 1]
time.sleep(0.05)
print("sampled 42 frames (3 nontable skipped), 9 named cards total")
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"states": [], "summary": {}}, handle)
"""


def _synthetic_repo(tmp_path: Path, *, pipeline_body: str = _FAKE_PIPELINE) -> Path:
    root = tmp_path / "repo"
    (root / "cv_lab" / "scripts" / "pipeline").mkdir(parents=True)
    (root / "cv_lab" / "models").mkdir(parents=True)
    (root / probe_module.PIPELINE_SCRIPT).write_text(pipeline_body, encoding="utf-8")
    (root / "cv_lab" / "models" / "region_spine_v1.pt").write_bytes(b"detector")
    (root / "cv_lab" / "models" / "card_cls_v1.pt").write_bytes(b"classifier")
    return root


def test_reconstruction_reports_wall_time_frames_and_throughput(tmp_path) -> None:
    root = _synthetic_repo(tmp_path)
    video = tmp_path / "session.mp4"
    video.write_bytes(b"not really a video, the pipeline is synthetic here")
    context = _context(tmp_path, repo_root=root, video=video)

    results = {m.spec.name: m for m in probe_reconstruction(context)}

    wall = results[RECONSTRUCTION_SECONDS.name]
    frames = results[RECONSTRUCTION_FRAMES.name]
    fps = results[RECONSTRUCTION_FPS.name]
    assert wall.taken and wall.value >= 0.05
    assert frames.taken and frames.value == 42
    assert fps.taken and fps.value == pytest.approx(42 / wall.value, rel=0.05)
    assert results["reconstruction.timeline_bytes"].value > 0
    assert wall.conditions["case"]["case_id"] == "operator_supplied"
    assert wall.conditions["sample_interval_s"] == 1.0


def test_throughput_is_withheld_when_the_pipeline_states_no_frame_count(
    tmp_path,
) -> None:
    root = _synthetic_repo(
        tmp_path,
        pipeline_body=(
            "import json, sys\n"
            "argv = sys.argv\n"
            "open(argv[argv.index('--out') + 1], 'w').write('{}')\n"
            "print('finished quietly')\n"
        ),
    )
    video = tmp_path / "session.mp4"
    video.write_bytes(b"synthetic")
    context = _context(tmp_path, repo_root=root, video=video)

    results = {m.spec.name: m for m in probe_reconstruction(context)}

    assert results[RECONSTRUCTION_SECONDS.name].taken, "wall time is still measurable"
    assert results[RECONSTRUCTION_FPS.name].value is None
    assert "frame count" in results[RECONSTRUCTION_FPS.name].not_taken_reason


def test_reconstruction_is_withheld_when_the_weights_are_not_installed(
    tmp_path,
) -> None:
    root = _synthetic_repo(tmp_path)
    (root / "cv_lab" / "models" / "card_cls_v1.pt").unlink()
    video = tmp_path / "session.mp4"
    video.write_bytes(b"synthetic")

    results = probe_reconstruction(_context(tmp_path, repo_root=root, video=video))

    assert all(not m.taken for m in results)
    assert "card_classifier" in results[0].not_taken_reason


def test_reconstruction_names_the_variable_that_would_make_it_runnable(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv(probe_module.VALIDATION_ROOT_ENV, raising=False)
    root = _synthetic_repo(tmp_path)

    results = probe_reconstruction(_context(tmp_path, repo_root=root))

    assert all(not m.taken for m in results)
    assert probe_module.VALIDATION_ROOT_ENV in results[0].not_taken_reason


# --------------------------------------------------------------------------
# The one-hour representative-session requirement
# --------------------------------------------------------------------------


def _host(*, designated: bool = False) -> dict:
    return {
        "label": "reference" if designated else "some-laptop",
        "designated_reference_label": "reference" if designated else None,
        "is_designated_reference": designated,
    }


def test_the_session_check_reports_never_run_when_nothing_was_reconstructed() -> None:
    measurements = [
        not_taken(
            RECONSTRUCTION_SECONDS, reason="no corpus is installed", probe="unit"
        )
    ]

    check = evaluate_session_check(measurements, _host())

    assert check["status"] == "never_run"
    assert check["observed_seconds"] is None
    assert check["certifies_release_gate"] is False
    assert "no corpus is installed" in check["reason"]


def test_the_session_check_fails_when_a_session_exceeds_the_hour() -> None:
    measurements = [measured(RECONSTRUCTION_SECONDS, value=3601.0, probe="unit")]

    check = evaluate_session_check(measurements, _host(designated=True))

    assert check["status"] == "exceeded_limit"
    assert check["observed_seconds"] == 3601.0
    assert check["certifies_release_gate"] is False


def test_the_session_check_certifies_only_on_the_designated_machine() -> None:
    measurements = [measured(RECONSTRUCTION_SECONDS, value=120.0, probe="unit")]

    off_reference = evaluate_session_check(measurements, _host())
    on_reference = evaluate_session_check(measurements, _host(designated=True))

    assert off_reference["status"] == "within_limit"
    assert off_reference["certifies_release_gate"] is False
    assert "not the designated reference machine" in off_reference["reason"]
    assert on_reference["certifies_release_gate"] is True


def test_the_session_check_counts_the_upload_alongside_the_reconstruction() -> None:
    measurements = [
        measured(RECONSTRUCTION_SECONDS, value=100.0, probe="unit"),
        measured(probe_module.UPLOAD_SECONDS, value=5.0, probe="unit"),
    ]

    check = evaluate_session_check(measurements, _host(designated=True))

    assert check["observed_seconds"] == 105.0
    assert check["components"][RECONSTRUCTION_SECONDS.name] == 100.0


# --------------------------------------------------------------------------
# The shipped baseline
# --------------------------------------------------------------------------


def test_the_shipped_baseline_measured_nothing_and_says_so() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert baseline["measurements"], "the baseline must still declare every metric"
    for entry in baseline["measurements"]:
        assert entry["value"] is None, f"{entry['name']} carries an invented number"
        assert entry["status"] == NEVER_MEASURED
    assert baseline["checks"][0]["status"] == "never_run"
    assert baseline["checks"][0]["certifies_release_gate"] is False
    assert all(value is None for value in baseline["host_fingerprint"].values())


def test_the_shipped_baseline_declares_exactly_the_current_metrics() -> None:
    """Adding a metric without regenerating the baseline leaves a silent hole."""
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert [entry["name"] for entry in baseline["measurements"]] == sorted(
        spec.name for spec in ALL_SPECS
    )
    assert baseline == empty_baseline()


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _report(values: dict[str, float | None], *, host: str = "a") -> dict:
    fingerprint = {"system": "Linux", "machine": "arm64", "cpu_count": 4, "python": "3.13.5", "label": host}
    measurements = []
    for name, value in values.items():
        spec = next(spec for spec in ALL_SPECS if spec.name == name)
        entry = {
            "name": name,
            "unit": spec.unit,
            "group": spec.group,
            "description": spec.description,
            "lower_is_better": spec.lower_is_better,
            "conditions": {"probe": "unit"},
        }
        if value is None:
            entry.update(
                {"status": NOT_TAKEN, "value": None, "not_taken_reason": "not run here"}
            )
        else:
            entry.update({"status": MEASURED, "value": value, "not_taken_reason": None})
        measurements.append(entry)
    return {"host_fingerprint": fingerprint, "measurements": measurements}


def _entry(comparison: dict, name: str) -> dict:
    return next(e for e in comparison["entries"] if e["name"] == name)


def test_a_slower_run_on_the_same_host_is_a_regression() -> None:
    comparison = compare_reports(
        _report({RECONSTRUCTION_SECONDS.name: 100.0}),
        _report({RECONSTRUCTION_SECONDS.name: 200.0}),
    )

    assert _entry(comparison, RECONSTRUCTION_SECONDS.name)["status"] == REGRESSED
    assert comparison["regressions"] == [RECONSTRUCTION_SECONDS.name]


def test_a_faster_run_is_an_improvement_and_small_drift_is_not_a_change() -> None:
    improved = compare_reports(
        _report({RECONSTRUCTION_SECONDS.name: 100.0}),
        _report({RECONSTRUCTION_SECONDS.name: 50.0}),
    )
    drift = compare_reports(
        _report({RECONSTRUCTION_SECONDS.name: 100.0}),
        _report({RECONSTRUCTION_SECONDS.name: 110.0}),
    )

    assert _entry(improved, RECONSTRUCTION_SECONDS.name)["status"] == IMPROVED
    assert _entry(drift, RECONSTRUCTION_SECONDS.name)["status"] == UNCHANGED
    assert improved["regressions"] == [] and drift["regressions"] == []


def test_throughput_regresses_when_it_falls_not_when_it_rises() -> None:
    fell = compare_reports(
        _report({RECONSTRUCTION_FPS.name: 10.0}), _report({RECONSTRUCTION_FPS.name: 2.0})
    )
    rose = compare_reports(
        _report({RECONSTRUCTION_FPS.name: 10.0}), _report({RECONSTRUCTION_FPS.name: 20.0})
    )

    assert _entry(fell, RECONSTRUCTION_FPS.name)["status"] == REGRESSED
    assert _entry(rose, RECONSTRUCTION_FPS.name)["status"] == IMPROVED


def test_a_missing_number_on_either_side_is_never_a_verdict() -> None:
    no_current = compare_reports(
        _report({RECONSTRUCTION_SECONDS.name: 100.0}),
        _report({RECONSTRUCTION_SECONDS.name: None}),
    )
    no_baseline = compare_reports(
        _report({RECONSTRUCTION_SECONDS.name: None}),
        _report({RECONSTRUCTION_SECONDS.name: 100.0}),
    )

    assert _entry(no_current, RECONSTRUCTION_SECONDS.name)["status"] == MISSING_CURRENT
    assert _entry(no_baseline, RECONSTRUCTION_SECONDS.name)["status"] == MISSING_BASELINE
    assert no_current["regressions"] == [] and no_baseline["regressions"] == []
    assert no_current["compared"] == 0


def test_numbers_from_two_machines_are_reported_but_not_judged() -> None:
    comparison = compare_reports(
        _report({RECONSTRUCTION_SECONDS.name: 100.0}, host="laptop"),
        _report({RECONSTRUCTION_SECONDS.name: 900.0}, host="cloud-arm"),
    )

    entry = _entry(comparison, RECONSTRUCTION_SECONDS.name)
    assert entry["status"] == INCOMPARABLE_HOST
    assert entry["current_value"] == 900.0
    assert comparison["comparable"] is False
    assert comparison["regressions"] == []
    assert "different machines" in entry["note"]


def test_the_shipped_empty_baseline_produces_no_verdicts(tmp_path) -> None:
    current = run_harness(
        HarnessOptions(workspace=tmp_path / "ws", groups=(), db_path=tmp_path / "none.db")
    ).to_dict()

    comparison = compare_reports(empty_baseline(), current)

    assert comparison["regressions"] == []
    assert comparison["compared"] == 0
    assert all(
        entry["status"] in {MISSING_BASELINE, MISSING_CURRENT}
        for entry in comparison["entries"]
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_the_cli_writes_a_machine_readable_report(tmp_path, capsys) -> None:
    out = tmp_path / "report.json"
    code = main(
        [
            "run",
            "--groups",
            "",
            "--workspace",
            str(tmp_path / "ws"),
            "--out",
            str(out),
        ]
    )
    capsys.readouterr()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert code == EXIT_OK
    assert payload["kind"] == "pokertrainer_perf_report"
    assert len(payload["measurements"]) == len(ALL_SPECS)
    assert summarize(payload).startswith("host:")


def test_the_cli_refuses_an_unknown_probe_group(tmp_path, capsys) -> None:
    code = main(["run", "--groups", "teleportation", "--out", str(tmp_path / "r.json")])
    captured = capsys.readouterr()

    assert code == EXIT_USAGE
    assert "unknown probe group" in captured.err


def test_the_cli_fails_the_session_check_when_it_has_never_run(tmp_path, capsys) -> None:
    code = main(
        [
            "run",
            "--groups",
            "",
            "--workspace",
            str(tmp_path / "ws"),
            "--out",
            str(tmp_path / "r.json"),
            "--require-session-check",
        ]
    )
    captured = capsys.readouterr()

    assert code == EXIT_SESSION_CHECK
    assert "never_run" in captured.err


def test_the_cli_reports_a_regression_against_a_baseline(tmp_path, capsys) -> None:
    report_path = tmp_path / "current.json"
    main(
        [
            "run",
            "--groups",
            "",
            "--workspace",
            str(tmp_path / "ws"),
            "--out",
            str(report_path),
        ]
    )
    current = json.loads(report_path.read_text(encoding="utf-8"))
    # A baseline recorded on this machine, with one metric twice as good.
    baseline = dict(current)
    baseline["measurements"] = [
        {**entry, "value": entry["value"] / 2}
        if entry["name"] == "memory.harness_peak_rss_bytes" and entry["value"]
        else entry
        for entry in current["measurements"]
    ]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    code = main(
        [
            "compare",
            "--baseline",
            str(baseline_path),
            "--report",
            str(report_path),
            "--fail-on-regression",
        ]
    )
    captured = capsys.readouterr()

    assert code == EXIT_REGRESSION
    assert "memory.harness_peak_rss_bytes" in captured.out


def test_the_cli_writes_an_empty_baseline_on_request(tmp_path, capsys) -> None:
    out = tmp_path / "baseline.json"
    code = main(["new-baseline", "--out", str(out)])
    capsys.readouterr()

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert code == EXIT_OK
    assert all(entry["value"] is None for entry in payload["measurements"])


def test_the_host_block_redacts_configured_secrets(monkeypatch) -> None:
    monkeypatch.setenv("POKER_TEST_API_KEY", "sk-not-a-real-key-000000000")

    host = describe_host(REPO_ROOT)

    assert host["env"].get("POKER_TEST_API_KEY") == "<redacted>"


def test_the_comparison_module_declares_a_finite_tolerance() -> None:
    assert 0 < compare_module.DEFAULT_TOLERANCE < 1
