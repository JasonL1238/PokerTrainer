"""Bounds on frame sampling: what the sampler may emit, and what the release
gate may assume about how long a recording is.

The two are one defect. ``_sample_times`` ran ``while t <= end`` with no
end-of-file test, so past the last frame the decoder kept handing back the final
image and the sampler emitted it under every remaining timestamp. The release
gate then supplied ``end`` = 86 400 for any manifest case without a
``duration_s``, which turns a short recording into ~86 401 sampled timestamps of
which all but a few hundred are the same picture. Neither the timeline nor the
report says so: the states look like a table that stopped changing, which is
exactly what a real idle table looks like.

A wrong prediction that is visibly rejected is a coverage limitation. Repeated
frames presented as distinct observations are not rejected by anything, so they
are the other kind.
"""

from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

av = pytest.importorskip("av")
np = pytest.importorskip("numpy")

from cv_lab.scripts.pipeline.run_two_model_pipeline import _sample_times  # noqa: E402
from poker_tracker.release_gate import runner  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic clips. Every frame is visibly different from every other, so two
# equal images can only mean the decoder returned the same frame twice.
# --------------------------------------------------------------------------- #
def _frame_image(index: int, size: int = 64):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :, 0] = (index * 37) % 256
    img[(index * 7) % size, :, 1] = 255
    return img


def _write_clip(path: Path, *, frames: int, gap_pts: int, rate) -> Path:
    """A clip of ``frames`` pictures, one every ``gap_pts`` of a 1s time base."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=rate)
    stream.width = stream.height = 64
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, 1)
    for index in range(frames):
        frame = av.VideoFrame.from_ndarray(_frame_image(index), format="rgb24")
        frame.pts = index * gap_pts
        frame.time_base = Fraction(1, 1)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture
def clip_3s(tmp_path: Path) -> Path:
    """3 seconds at 10 fps: the last frame is at t=2.9."""
    path = tmp_path / "clip_3s.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=10)
    stream.width = stream.height = 64
    stream.pix_fmt = "yuv420p"
    for index in range(30):
        frame = av.VideoFrame.from_ndarray(_frame_image(index), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture
def clip_sparse(tmp_path: Path) -> Path:
    """5 frames, one every 2 seconds -- a screen recording of a static screen."""
    return _write_clip(tmp_path / "sparse.mp4", frames=5, gap_pts=2, rate=Fraction(1, 2))


def _sample(path: Path, *, start: float, end: float, interval: float):
    stats: dict = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        samples = list(
            _sample_times(container, stream, start, end, interval, stats=stats)
        )
    digests = [hashlib.sha256(img.tobytes()).hexdigest() for _t, img in samples]
    return [t for t, _img in samples], digests, stats


# --------------------------------------------------------------------------- #
# The sampler
# --------------------------------------------------------------------------- #
def test_sampling_stops_at_the_end_of_the_stream_instead_of_repeating_it(clip_3s):
    """The measured defect: a 3-second clip asked for 10 seconds produced 11
    timestamped states from 4 distinct pictures, the last 8 of them the same
    final frame emitted under 8 different times."""
    times, digests, stats = _sample(clip_3s, start=0.0, end=10.0, interval=1.0)

    assert times == [0.0, 1.0, 2.0]
    assert len(set(digests)) == len(digests), "no picture may be emitted twice"
    assert stats["ended_at"] == pytest.approx(3.0)
    assert stats["requested"] == 4
    assert stats["emitted"] == 3


def test_a_window_inside_the_stream_is_sampled_exactly_as_before(clip_3s):
    """The bound must not truncate a legitimate request. Every sample time the
    stream can answer is still answered, at the same timestamp as before."""
    times, digests, stats = _sample(clip_3s, start=0.0, end=2.0, interval=0.5)

    assert times == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert len(set(digests)) == 5
    assert stats["ended_at"] is None
    assert stats["duplicate_times"] == 0


def test_one_decoded_frame_is_never_emitted_under_two_timestamps(clip_sparse):
    """A recording with a 2s gap between frames, sampled every 1s. Half the
    sample times resolve to a frame that was already emitted; emitting it again
    would report one observation as two, and the duplicates would then debounce
    each other into a confirmed reading."""
    times, digests, stats = _sample(clip_sparse, start=0.0, end=9.0, interval=1.0)

    assert len(set(digests)) == len(digests) == 5
    assert stats["duplicate_times"] == 4
    assert stats["emitted"] == 5
    assert times == [0.0, 1.0, 3.0, 5.0, 7.0]


def test_a_sample_time_before_the_first_frame_does_not_stop_the_run(clip_3s):
    """Starting the window before the stream's own start is a normal request,
    not an exhausted stream: the first frame answers it."""
    times, digests, _stats = _sample(clip_3s, start=0.0, end=0.2, interval=0.1)

    assert times == [0.0, 0.1, 0.2]
    assert len(set(digests)) == 3


# --------------------------------------------------------------------------- #
# The release gate's sampling window
# --------------------------------------------------------------------------- #
MODELS = {
    "region_detector": {"present": True, "path": "cv_lab/models/region_spine_v1.pt"},
    "card_classifier": {"present": True, "path": "cv_lab/models/card_cls_v1.pt"},
}


def _manifest(tmp_path: Path, *, logical_name: str, duration_s) -> Path:
    """One release-scored case pointing at ``logical_name`` in the vault."""
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir(exist_ok=True)
    (truth_dir / "case.json").write_text(json.dumps({"hands": []}), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "corpus_id": "sampling_bounds",
        "cases": [
            {
                "case_id": "case_00",
                "split": "development",
                "runtime_class": "video",
                "counts_toward_release": True,
                "recording": {"logical_name": logical_name, "duration_s": duration_s},
                "truth_relpath": "truth/case.json",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def gate_ready(monkeypatch, tmp_path: Path):
    """Every full-mode precondition except the recording itself."""
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("POKER_VALIDATION_ROOT", str(vault))
    monkeypatch.setattr(runner, "ffmpeg_version", lambda: "ffmpeg version 6.0")
    calls: list[dict] = []

    def _record(**kwargs):
        calls.append(kwargs)
        return {"ok": False, "error": "not run: this test only inspects the window"}

    monkeypatch.setattr(runner, "_run_pipeline_for_case", _record)
    return vault, calls


def test_a_case_without_a_duration_is_probed_not_defaulted_to_a_day(
    gate_ready, tmp_path: Path
):
    """``duration_s: null`` used to become ``--end 86400``: ~86 401 sample times
    from a 3-second recording. The recording itself knows how long it is."""
    vault, calls = gate_ready
    _write_clip(vault / "case.mp4", frames=30, gap_pts=1, rate=1)

    detail = runner._run_full_reconstruction(
        _manifest(tmp_path, logical_name="case.mp4", duration_s=None),
        report_dir=tmp_path / "reports",
        resolved_models=MODELS,
    )

    assert len(calls) == 1
    assert calls[0]["duration_s"] == pytest.approx(30.0, abs=1.0)
    assert calls[0]["duration_s"] < 86_400.0
    case = detail["cases"][0]
    assert case["duration_source"] == "probed_from_recording"


def test_a_declared_duration_is_used_as_declared(gate_ready, tmp_path: Path):
    vault, calls = gate_ready
    _write_clip(vault / "case.mp4", frames=30, gap_pts=1, rate=1)

    detail = runner._run_full_reconstruction(
        _manifest(tmp_path, logical_name="case.mp4", duration_s=12.5),
        report_dir=tmp_path / "reports",
        resolved_models=MODELS,
    )

    assert calls[0]["duration_s"] == pytest.approx(12.5)
    assert detail["cases"][0]["duration_source"] == "manifest"


def test_an_unmeasurable_duration_fails_the_case_instead_of_guessing_one(
    gate_ready, tmp_path: Path
):
    """No declared duration and a file that cannot be probed is missing
    information. The gate must say so and not run: a guessed window produces
    either a timeout reported as a reconstruction failure, or a timeline
    assembled from repeats of one frame."""
    vault, calls = gate_ready
    (vault / "case.mp4").write_bytes(b"not a video")

    detail = runner._run_full_reconstruction(
        _manifest(tmp_path, logical_name="case.mp4", duration_s=None),
        report_dir=tmp_path / "reports",
        resolved_models=MODELS,
    )

    assert calls == [], "nothing may be reconstructed over an unknown window"
    case = detail["cases"][0]
    assert case["ok"] is False
    assert "duration is unknown" in case["fail_closed"]
    assert case["duration_source"] == "unmeasurable"
    # Not a bad score: the run could not be performed.
    assert detail["setup_invalid"] is True
    assert detail["cases_scored"] == 0


class _Completed:
    """The shape ``subprocess.run`` returns, for tests that stub it out."""

    returncode = 0
    stdout = ""
    stderr = ""


@pytest.mark.parametrize("bad", [None, "600", True, float("inf"), 0.0, -5.0])
def test_the_pipeline_is_never_launched_over_an_unusable_duration(
    monkeypatch, tmp_path: Path, bad
):
    """Defence in depth at the layer that builds the command line: no fallback
    lives here, because a case report cannot describe a window chosen here."""
    launched: list[list[str]] = []

    def _fake_run(command, **kwargs):
        launched.append(command)
        return _Completed()

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)

    result = runner._run_pipeline_for_case(
        video_path=tmp_path / "case.mp4",
        timeline_path=tmp_path / "case.timeline.json",
        duration_s=bad,
        detector=None,
        classifier=None,
    )

    assert launched == []
    assert result["ok"] is False
    assert "duration is unknown" in result["error"]


def test_the_resolved_duration_is_what_the_pipeline_is_asked_to_sample(
    monkeypatch, tmp_path: Path
):
    commands: list[list[str]] = []

    def _fake_run(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--out") + 1]).write_text("{}", encoding="utf-8")
        return _Completed()

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)

    result = runner._run_pipeline_for_case(
        video_path=tmp_path / "case.mp4",
        timeline_path=tmp_path / "case.timeline.json",
        duration_s=42.5,
        detector=None,
        classifier=None,
    )

    assert result["ok"] is True
    command = commands[0]
    assert command[command.index("--end") + 1] == "42.5"
