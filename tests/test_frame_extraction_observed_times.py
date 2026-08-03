"""What an extracted frame is allowed to say about itself.

Two claims the extractor used to make without checking them. It stored each
frame under the time the seek asked for rather than the time the decoder said
the frame came from, so on a variable-rate recording -- a screen capture encodes
nothing while the screen is still -- a picture taken at 10s was filed at 4s and
the seven unwatched seconds in between disappeared from the record. And it
counted a frame as extracted without checking that the JPEG was written, so a
failed write left a row pointing at a file that is not there.

A wrong prediction that is visibly rejected is a coverage limitation. A frame
filed under a time nobody observed, or a database row for a picture that was
never saved, is study-ready material that nothing downstream questions.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from poker_tracker.persistence.db import PokerDatabase
from poker_tracker.persistence.models import VideoRecord
from poker_tracker.ui import frame_extraction


class _FakeCapture:
    """A capture whose frames sit at ``frame_times``, seek-to-next like OpenCV.

    ``reports_positions=False`` is the backend that cannot say where a frame
    came from: real OpenCV returns 0 for every frame when the stream carries no
    presentation timestamps.
    """

    frame_times: list[float] = []
    reports_positions = True

    def __init__(self, _path: str) -> None:
        self.next_index = 0
        self.current_index = 0

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_MSEC:
            target = value / 1000.0
            self.next_index = next(
                (i for i, t in enumerate(self.frame_times) if t >= target),
                len(self.frame_times),
            )
        return True

    def read(self):
        if self.next_index >= len(self.frame_times):
            return False, None
        self.current_index = self.next_index
        self.next_index += 1
        image = np.full((8, 8, 3), self.current_index * 7 % 256, dtype=np.uint8)
        return True, image

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.next_index)
        if prop == cv2.CAP_PROP_POS_MSEC:
            if not self.reports_positions:
                return 0.0
            return self.frame_times[self.current_index] * 1000.0
        if prop == cv2.CAP_PROP_FPS:
            return 1.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frame_times))
        return 0.0

    def release(self) -> None:
        return None


def _capture_class(frame_times: list[float], *, reports_positions: bool = True):
    return type(
        "_Capture",
        (_FakeCapture,),
        {"frame_times": frame_times, "reports_positions": reports_positions},
    )


@pytest.fixture
def db_and_video(tmp_path: Path):
    db = PokerDatabase(":memory:")
    db.init_db()
    stored = tmp_path / "recording.mp4"
    stored.write_bytes(b"stand-in for the recording the fake capture serves")
    video = db.create_video(
        VideoRecord(
            original_filename=stored.name,
            stored_path=str(stored),
            file_size_bytes=stored.stat().st_size,
        )
    )
    yield db, video
    db.close()


def test_a_frame_is_stored_under_the_time_it_was_taken_not_the_time_requested(
    db_and_video, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing was recorded between 3s and 10s. Seeking to 4s, 5s ... returns
    the picture taken at 10s; filing it as 4s reports seven unobserved seconds
    as covered and leaves the next frames a second early as well."""
    db, video = db_and_video
    monkeypatch.setattr(
        frame_extraction.cv2,
        "VideoCapture",
        _capture_class([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]),
    )

    summary = frame_extraction.extract_frames_for_video(
        db,
        video.id,
        interval_seconds=1.0,
        end_time_seconds=12.0,
        frames_dir=tmp_path / "frames",
    )

    stored = [frame.timestamp_seconds for frame in db.fetch_frames_by_video(video.id)]
    assert stored == [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
    assert summary.frames_extracted == 7
    # The stretch nobody watched has to stay visible as a gap between records.
    gaps = [b - a for a, b in zip(stored, stored[1:], strict=False)]
    assert [gap for gap in gaps if gap > 1.0] == [7.0]


def test_a_backend_that_cannot_report_positions_does_not_file_everything_at_zero(
    db_and_video, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenCV answers 0ms for every frame of a stream with no timestamps. Taking
    that literally would stack the whole extraction on one second; the requested
    time is the honest remaining answer, so the fallback is used."""
    db, video = db_and_video
    monkeypatch.setattr(
        frame_extraction.cv2,
        "VideoCapture",
        _capture_class([0.0, 1.0, 2.0], reports_positions=False),
    )

    frame_extraction.extract_frames_for_video(
        db,
        video.id,
        interval_seconds=1.0,
        end_time_seconds=2.0,
        frames_dir=tmp_path / "frames",
    )

    stored = [frame.timestamp_seconds for frame in db.fetch_frames_by_video(video.id)]
    assert stored == [0.0, 1.0, 2.0]


def test_a_frame_whose_jpeg_was_not_written_is_not_recorded_as_extracted(
    db_and_video, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cv2.imwrite reports failure by returning False. Ignoring it produced a
    row, a count and a completed job for a picture that is not on disk."""
    db, video = db_and_video
    monkeypatch.setattr(
        frame_extraction.cv2, "VideoCapture", _capture_class([0.0, 1.0, 2.0])
    )
    monkeypatch.setattr(frame_extraction.cv2, "imwrite", lambda *args, **kwargs: False)

    summary = frame_extraction.extract_frames_for_video(
        db,
        video.id,
        interval_seconds=1.0,
        end_time_seconds=2.0,
        frames_dir=tmp_path / "frames",
    )

    assert summary.frames_extracted == 0
    assert db.fetch_frames_by_video(video.id) == []
    assert len(summary.errors) == 3
    assert all("Could not write frame" in message for message in summary.errors)
    assert "Extracted 0 frames" in db.fetch_jobs_by_video(video.id)[0].message


def test_a_written_frame_still_lands_on_disk_at_the_path_recorded(
    db_and_video, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write check must not turn successful writes away."""
    db, video = db_and_video
    monkeypatch.setattr(
        frame_extraction.cv2, "VideoCapture", _capture_class([0.0, 1.0, 2.0])
    )

    summary = frame_extraction.extract_frames_for_video(
        db,
        video.id,
        interval_seconds=1.0,
        end_time_seconds=2.0,
        frames_dir=tmp_path / "frames",
    )

    frames = db.fetch_frames_by_video(video.id)
    assert summary.frames_extracted == 3
    assert summary.errors == []
    assert all(Path(frame.image_path).exists() for frame in frames)
