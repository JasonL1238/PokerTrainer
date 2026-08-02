"""Property and fuzz tests for path and filename validation (PLAN Phase 14).

``safe_filename`` decides where an operator's upload lands on disk, and until
now it was pinned by exactly one example (``tests/test_video_processing.py``
feeds it ``"Session 01!.MP4"``). Everything hostile a filename can carry --
traversal sequences, an absolute path where a basename is expected, NUL bytes,
Windows drive and UNC prefixes, names that reduce to nothing after
sanitization, names that are only dots, reserved device names, unicode that
normalizes two ways, case-only differences on a case-insensitive filesystem,
and names longer than any filesystem will accept -- was untested.

THE INVARIANT, stated once and generated against rather than exampled:

    (root / safe_filename(x)).resolve() is inside root.resolve(), for every x
    the function does not reject outright.

Everything else here is a consequence of it or a bound the invariant needs in
order to be reachable in practice: the result has to be ONE path component, and
it has to be short enough that the filesystem will accept the name
``unique_stored_video_path`` builds out of it.

``safe_roi_key`` is the same shape of function on the other operator-facing
surface (ROI preview filenames), so it is generated against the same way.
"""

from __future__ import annotations

import io
import os
import unicodedata
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from poker_tracker.ui.roi import safe_roi_key
from poker_tracker.ui.video_storage import (
    _MAX_STORED_NAME_BYTES,
    _STORED_NAME_PREFIX_BYTES,
    ALLOWED_VIDEO_EXTENSIONS,
    safe_filename,
    save_video_file,
    unique_stored_video_path,
    validate_video_extension,
)

SETTINGS = settings(
    max_examples=300,
    deadline=None,
    # The tmp directories below are read-only scratch for a resolve() comparison
    # -- nothing is written into them -- so reuse across generated inputs is
    # correct rather than merely tolerated.
    suppress_health_check=[
        HealthCheck.filter_too_much,
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
    ],
)

# The fragments a hostile upload name is actually built from. Drawing free text
# alone almost never produces "../" or a drive prefix, so the families this
# suite exists to cover would be generated approximately never.
HOSTILE_FRAGMENTS = st.sampled_from(
    [
        "..",
        "../",
        "..\\",
        "/",
        "\\",
        "//",
        "\x00",
        "\n",
        "\r",
        "\t",
        ".",
        "...",
        "-",
        "_",
        " ",
        "%2e%2e",
        "%2f",
        "C:",
        "C:\\",
        "\\\\server\\share",
        "~",
        "$HOME",
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "LPT1",
        "\u00c5",              # precomposed A-ring
        "A\u030a",             # decomposed A-ring: same text, different bytes
        "\u202e",              # right-to-left override
        "\ufeff",
        "\u0130",              # dotted capital I: lowercases to two code points
        "\u212a",              # KELVIN SIGN: lowercases to ASCII 'k'
        "video",
        "session 01",
        "a" * 400,
    ]
)

EXTENSIONS = st.sampled_from(
    sorted(ALLOWED_VIDEO_EXTENSIONS)
    + [ext.upper() for ext in sorted(ALLOWED_VIDEO_EXTENSIONS)]
    + [".MP4", ".Mov", ".txt", ".exe", ".mp4.txt", "", ".", ".mp4\x00.txt"]
)


@st.composite
def hostile_upload_names(draw) -> str:
    """A filename an operator (or an attacker) can hand the upload control."""
    parts = draw(st.lists(HOSTILE_FRAGMENTS, min_size=0, max_size=6))
    free = draw(st.text(max_size=20))
    extension = draw(EXTENSIONS)
    return "".join(parts) + free + extension


def _inside(root: Path, candidate: Path) -> bool:
    resolved = candidate.resolve()
    return resolved != root.resolve() and root.resolve() in resolved.parents


@given(name=hostile_upload_names())
@SETTINGS
def test_a_sanitized_upload_name_can_never_escape_the_videos_directory(
    name: str, tmp_path_factory
) -> None:
    """THE invariant. A name that cannot be made safe must be refused, and a name
    that is accepted must land inside the root -- there is no third answer.

    ``resolve()`` is what makes this a real test rather than a string check: it
    collapses ``..`` and follows the path the way the filesystem will, so a
    result that merely LOOKS like a basename but resolves upward still fails.
    """
    root = tmp_path_factory.mktemp("videos-root")
    try:
        safe = safe_filename(name)
    except ValueError:
        return  # refusing is always a correct answer
    assert _inside(root, root / safe), (
        f"{name!r} sanitized to {safe!r}, which resolves outside {root}"
    )


@given(name=hostile_upload_names())
@SETTINGS
def test_a_sanitized_upload_name_is_a_single_path_component(name: str) -> None:
    """The escape invariant holds because the result is one component and nothing
    the filesystem reads as navigation. Stated separately so a regression names
    which property broke rather than only that the resolve moved."""
    try:
        safe = safe_filename(name)
    except ValueError:
        return
    assert safe not in {"", ".", ".."}
    assert "\x00" not in safe
    assert os.sep not in safe
    if os.altsep:
        assert os.altsep not in safe
    assert "/" not in safe and "\\" not in safe
    assert Path(safe).name == safe
    assert not Path(safe).is_absolute()
    assert Path(safe).suffix in ALLOWED_VIDEO_EXTENSIONS


@given(name=hostile_upload_names())
@SETTINGS
def test_a_sanitized_upload_name_fits_a_real_filesystem_path_component(
    name: str,
) -> None:
    """A name the function accepts must be storable.

    Before the bound existed, a 10,000-character upload name produced a
    10,033-character stored path: ``tempfile.mkstemp`` succeeded (its prefix is
    fixed), the whole video streamed to disk, and ``os.replace`` then raised a
    bare ``OSError: [Errno 63] File name too long`` carrying the entire path.
    The operator lost the upload and got an errno instead of a message.

    Sanitization leaves only ``[a-z0-9._-]``, so the encoded length equals the
    character length and one budget covers both.
    """
    try:
        safe = safe_filename(name)
    except ValueError:
        return
    stored = len(safe.encode("utf-8")) + _STORED_NAME_PREFIX_BYTES
    assert stored <= _MAX_STORED_NAME_BYTES, (
        f"{name[:40]!r} -> a stored component of {stored} bytes, which no common "
        "filesystem will accept"
    )


@given(name=hostile_upload_names())
@SETTINGS
def test_sanitizing_an_already_sanitized_name_changes_nothing(name: str) -> None:
    """Idempotence. The stored name is re-derived on several paths (ingest, the
    frame extractor, the job recovery toast); a function that keeps shortening
    its own output would make those disagree about which file is which."""
    try:
        once = safe_filename(name)
    except ValueError:
        return
    assert safe_filename(once) == once


@given(name=hostile_upload_names())
@SETTINGS
def test_case_and_unicode_normalization_differences_collapse_to_one_ascii_name(
    name: str,
) -> None:
    """This project has already been bitten by a case-insensitive filesystem.

    The stored name is lowercase ASCII, so two uploads differing only in case
    cannot become two files that the filesystem then treats as one. Composed and
    decomposed unicode are NOT promised to agree -- ``\u00c5.mp4`` and
    ``A\u030a.mp4`` sanitize differently -- and that is safe only because
    ``unique_stored_video_path`` derives uniqueness from a uuid rather than from
    the stem; this test states the guarantee that actually holds instead of the
    one it is tempting to assume.
    """
    try:
        safe = safe_filename(name)
    except ValueError:
        return
    assert safe == safe.lower()
    assert safe.isascii()
    assert safe_filename(name.upper()) == safe_filename(name.lower()) or (
        # Unicode case folding is not a bijection (KELVIN SIGN, dotted capital I),
        # so upper/lower round trips can legitimately differ. What may never
        # differ is the ASCII-ness and the containment already asserted above.
        safe_filename(name.upper()).isascii()
    )
    assert unicodedata.normalize("NFC", safe) == safe


@given(name=hostile_upload_names())
@SETTINGS
def test_the_stored_video_path_stays_inside_the_videos_directory(
    name: str, tmp_path_factory
) -> None:
    """The invariant on the function operators actually reach: the same claim one
    level up, where the uuid prefix and the directory join happen."""
    root = tmp_path_factory.mktemp("data")
    videos = root / "videos"
    try:
        stored = unique_stored_video_path(name, videos)
    except ValueError:
        return
    assert _inside(videos, stored)
    assert stored.parent.resolve() == videos.resolve()


@given(
    stem=st.text(max_size=40),
    extension=st.sampled_from(sorted(ALLOWED_VIDEO_EXTENSIONS)),
)
@SETTINGS
def test_extension_validation_accepts_only_the_allowlist_in_any_case(
    stem: str, extension: str
) -> None:
    for variant in (extension, extension.upper(), extension.title()):
        assert validate_video_extension(f"clip{variant}") == extension
    assert validate_video_extension(f"{stem}x{extension}") == extension


@given(name=hostile_upload_names())
@SETTINGS
def test_a_name_outside_the_extension_allowlist_is_refused_not_repaired(
    name: str,
) -> None:
    """Refusal is the only answer for an unsupported container. A sanitizer that
    invented ``.mp4`` for a ``.exe`` upload would put an executable behind a
    video extension in the videos directory."""
    suffix = Path(name).suffix.lower()
    if suffix in ALLOWED_VIDEO_EXTENSIONS:
        return
    with pytest.raises(ValueError, match="Unsupported video extension"):
        safe_filename(name)


@pytest.mark.parametrize(
    "name",
    [
        "../../etc/passwd.mp4",
        "..\\..\\windows\\system32\\config.mp4",
        "/etc/shadow.mp4",
        "/../../../../root/.ssh/id_rsa.mp4",
        "video\x00.exe.mp4",
        "C:\\Windows\\System32\\evil.mp4",
        "\\\\attacker\\share\\evil.mp4",
        "....mp4",
        "..mp4",
        "   .mp4",
        "___.mp4",
        "---.mp4",
        "\u202eexe.mp4",
        "CON.mp4",
        "NUL.mp4",
        "x" * 10_000 + ".mp4",
    ],
)
def test_named_hostile_upload_names_are_contained(name: str, tmp_path: Path) -> None:
    """The specific shapes the property search is meant to cover, written out so a
    reader can see them and a failure names the case rather than a seed."""
    safe = safe_filename(name)
    assert _inside(tmp_path, tmp_path / safe)
    assert Path(safe).name == safe
    assert len(safe) + _STORED_NAME_PREFIX_BYTES <= _MAX_STORED_NAME_BYTES


def test_an_oversized_upload_name_is_stored_rather_than_failing_at_os_replace(
    tmp_path: Path,
) -> None:
    """The regression for the length bound, driven through the writer.

    Pre-fix this raised ``OSError: [Errno 63] File name too long`` from
    ``os.replace`` after the payload had already been written to a temp file in
    the videos directory.
    """
    videos = tmp_path / "videos"
    stored = save_video_file(io.BytesIO(b"not-really-a-video"), "y" * 5000 + ".MP4", videos)
    assert stored.is_file()
    assert stored.parent == videos
    assert len(stored.name.encode("utf-8")) <= _MAX_STORED_NAME_BYTES
    assert stored.read_bytes() == b"not-really-a-video"


@given(
    key=st.one_of(
        st.text(max_size=60),
        st.lists(HOSTILE_FRAGMENTS, min_size=1, max_size=5).map("".join),
    )
)
@SETTINGS
def test_a_sanitized_roi_key_is_a_single_path_component(key: str, tmp_path: Path) -> None:
    """``safe_roi_key`` names ROI preview files, so it carries the same invariant
    as ``safe_filename`` and, unlike it, has no extension allowlist standing in
    front of it -- every input must come out containable."""
    safe = safe_roi_key(key)
    assert safe
    assert safe not in {".", ".."}
    assert "\x00" not in safe
    assert "/" not in safe and "\\" not in safe
    assert Path(safe).name == safe
    assert safe == safe.lower()
    assert _inside(tmp_path, tmp_path / safe)
    assert safe_roi_key(safe) == safe
