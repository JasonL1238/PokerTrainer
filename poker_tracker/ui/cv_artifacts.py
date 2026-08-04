"""The files a CV reconstruction job leaves behind, and which of them it owns.

A job's artifacts are addressed by **job id**, not by video id, and they are
spread across three data roots: the timeline, its progress snapshot and the
pipeline pid file under ``cv_timelines``, the validated-session export under
``exports``, and the detached worker's log under ``job_logs``. Nothing names any
of them in a database column, so deleting a video's rows cannot cascade to them
and :func:`delete_extracted_frames` -- which only knows the ``video_<id>``
directory of the *simple* extraction job -- never sees them either. Without this
module, removing one 90-minute reconstruction left its timeline, export, log and
progress file on disk with no row left that could ever name them again.

What this module deliberately does **not** remove is the reconstruction's frame
directory, ``frames/cv_job_<job id>/``. Those frames are the images
``actions.source_image`` points at, and that column is in
``PokerDatabase.ARTIFACT_PATH_COLUMNS``. Hands imported from a recording outlive
the recording on purpose, so deleting the frame directory would delete evidence
that surviving hands still reference -- the one thing
:mod:`poker_tracker.services.retention` forbids at any age: "a file the product
still expects is never deletable." Retention already separates referenced frames
from unreferenced ones and is the only correct owner of that directory.

The naming convention duplicated here is the one
:mod:`poker_tracker.persistence.backup_inventory` owns for the timeline. It is
repeated rather than imported for the other four because the inventory has no
reason to know about a progress file, a pid file or a log; the timeline path
itself is taken from that module so the two cannot disagree about the file whose
loss hard-blocks a later import.
"""

from __future__ import annotations

from pathlib import Path

from poker_tracker.persistence.backup_inventory import timeline_dir_for
from poker_tracker.ui import video_storage
from poker_tracker.ui.jobs import job_log_path


def cv_job_artifact_paths(job_id: int) -> list[Path]:
    """Every job-keyed file a reconstruction owns, whether or not it exists.

    Read through the ``video_storage`` module rather than from imported
    constants so a redirected data root -- which is how the test suite and a
    restored machine both work -- reaches this reader too.
    """
    timelines = timeline_dir_for(video_storage.DATA_DIR)
    return [
        timelines / f"job_{job_id}_timeline.json",
        timelines / f"job_{job_id}_progress.json",
        timelines / f"job_{job_id}_pipeline.pid",
        video_storage.EXPORTS_DIR / f"job_{job_id}_session.json",
        job_log_path(job_id),
    ]


def remove_cv_job_artifacts(job_id: int) -> list[Path]:
    """Delete this job's own files, returning the ones that were there.

    Removed eagerly rather than left for retention to expire, because deleting
    the ``processing_jobs`` row is what makes them undiscoverable: retention
    finds a timeline by querying completed job ids, so once the row is gone the
    file is an orphan no audit can report and no operator can name. An artifact
    the product removes on purpose is strictly better than one it silently
    abandons.

    A file that cannot be unlinked is skipped rather than raised on. The caller
    is midway through an irreversible delete whose database snapshot is already
    written; aborting on an unremovable log would leave the recording half
    deleted for no gain, and the leftover is inert either way.
    """
    removed: list[Path] = []
    for path in cv_job_artifact_paths(job_id):
        try:
            if path.is_file():
                path.unlink()
                removed.append(path)
        except OSError:
            continue
    return removed
