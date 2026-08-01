"""Install and verify the CV model artifacts the application loads at runtime.

Two of the weights the reconstruction pipeline resolves are tens of megabytes
and excluded from the repository, so a clean checkout does not have them. Until
now the container build papered over that by copying them out of one developer's
working tree, which meant the image could only ever be built on that machine and
nowhere else -- and, because the copy was silent, the dependency was invisible
until someone tried a clean build.

This script makes the dependency explicit. ``deploy/model_manifest.json`` names
every artifact with the digest that identifies it; this script either verifies
what is installed against that manifest or installs artifacts from a source the
operator names, refusing anything whose digest does not match. The repository
therefore states exactly what a working install requires, while the artifacts
themselves stay where the project keeps large operator state: on a mount, not in
the image.

    python deploy/provision_models.py --verify
    python deploy/provision_models.py --source /media/backup/pokertrainer-models
    python deploy/provision_models.py --source https://models.example.internal/pokertrainer/v1

``poker_tracker/release_gate/models.py`` pins the two YOLO roles for a different
purpose -- making a release verdict reproducible against the weights that
produced it -- and reads whatever is installed. This manifest is the
provisioning superset: it also covers the OCR and template banks, which the
release gate does not hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "deploy" / "model_manifest.json"
DEFAULT_MODELS_DIR = REPO_ROOT / "cv_lab" / "models"
CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Artifact:
    role: str
    filename: str
    sha256: str
    bytes: int
    required: bool
    purpose: str
    # Small enough to track in git, so it arrives with the checkout and with the
    # image. Nothing provisions it; it is still verified, because a corrupted
    # copy would silently change every OCR read.
    shipped_in_repository: bool

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> Artifact:
        return cls(
            role=str(entry["role"]),
            filename=str(entry["filename"]),
            sha256=str(entry["sha256"]).lower(),
            bytes=int(entry["bytes"]),
            required=bool(entry.get("required", False)),
            purpose=str(entry.get("purpose", "")),
            shipped_in_repository=bool(entry.get("shipped_in_repository", False)),
        )


class ProvisioningError(RuntimeError):
    """A failure an operator has to act on, phrased as what to do about it."""


def load_manifest(path: Path = MANIFEST_PATH) -> list[Artifact]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProvisioningError(f"Model manifest is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProvisioningError(f"Model manifest is not valid JSON: {path}: {exc}") from exc
    entries = payload.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise ProvisioningError(f"Model manifest lists no artifacts: {path}")
    return [Artifact.from_entry(entry) for entry in entries]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_target(models_dir: Path, filename: str) -> Path:
    """Where a write for ``filename`` actually has to land.

    In the container ``cv_lab/models`` holds symlinks into the data mount, so
    writing the entry itself would replace the symlink with a regular file in
    the read-only application layer. Resolving first keeps the mount as the one
    place the weights live.
    """
    entry = models_dir / filename
    return entry.resolve() if entry.is_symlink() else entry


def inspect(models_dir: Path, artifacts: list[Artifact]) -> list[dict[str, Any]]:
    """State of every manifest artifact in ``models_dir``, one row each."""
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        path = models_dir / artifact.filename
        row: dict[str, Any] = {
            "role": artifact.role,
            "filename": artifact.filename,
            "required": artifact.required,
            "path": str(path),
            "resolved_path": str(_install_target(models_dir, artifact.filename)),
        }
        if not path.is_file():
            row["state"] = "missing"
            rows.append(row)
            continue
        actual = sha256_file(path)
        row["sha256"] = actual
        row["state"] = "ok" if actual == artifact.sha256 else "digest_mismatch"
        if row["state"] == "digest_mismatch":
            row["expected_sha256"] = artifact.sha256
        rows.append(row)
    return rows


def _fetch_to(source: str, filename: str, destination: Path) -> None:
    """Copy one artifact from a directory or an https base URL.

    Written to a sibling temporary file and renamed only after the digest is
    checked, so an interrupted transfer can never be mistaken for an installed
    weight.
    """
    parsed = urllib.parse.urlparse(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{filename}.", suffix=".part"
    )
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        if parsed.scheme in {"http", "https"}:
            if parsed.scheme == "http":
                raise ProvisioningError(
                    f"Refusing to fetch model weights over plain http: {source}. "
                    "Use https, or copy the files to a local directory and pass that."
                )
            url = source.rstrip("/") + "/" + filename
            try:
                with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
                    with temp_path.open("wb") as out:
                        shutil.copyfileobj(response, out, CHUNK_BYTES)
            except urllib.error.URLError as exc:
                raise ProvisioningError(f"Could not download {url}: {exc}") from exc
        else:
            origin = Path(source).expanduser() / filename
            if not origin.is_file():
                raise ProvisioningError(f"Source does not contain {filename}: {origin}")
            shutil.copyfile(origin, temp_path)
        # mkstemp is owner-only by default; the application user has to be able
        # to read a weight an administrator installed.
        temp_path.chmod(0o644)
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)


def install(
    models_dir: Path,
    artifacts: list[Artifact],
    source: str,
    *,
    include_optional: bool,
) -> list[str]:
    """Install every artifact that is not already present and correct."""
    installed: list[str] = []
    for artifact in artifacts:
        if artifact.shipped_in_repository:
            continue
        if not artifact.required and not include_optional:
            continue
        entry = models_dir / artifact.filename
        if entry.is_file() and sha256_file(entry) == artifact.sha256:
            continue
        target = _install_target(models_dir, artifact.filename)
        _fetch_to(source, artifact.filename, target)
        actual = sha256_file(target)
        if actual != artifact.sha256:
            target.unlink(missing_ok=True)
            raise ProvisioningError(
                f"{artifact.filename} from {source} hashes to {actual}, but the manifest "
                f"pins {artifact.sha256}. The copy was discarded rather than installed; "
                f"{artifact.filename} is now absent and reconstruction will refuse to run."
            )
        installed.append(artifact.filename)
    return installed


def _report(rows: list[dict[str, Any]], models_dir: Path) -> bool:
    """Print the state of each artifact. True when every required one is usable."""
    ok = True
    print(f"Model artifacts in {models_dir}")
    for row in rows:
        tag = "required" if row["required"] else "optional"
        state = row["state"]
        if state == "ok":
            print(f"  [ ok       ] {row['filename']} ({tag})")
            continue
        if state == "missing":
            print(f"  [ MISSING  ] {row['filename']} ({tag}) -> {row['resolved_path']}")
        else:
            print(
                f"  [ MISMATCH ] {row['filename']} ({tag}) "
                f"has {row.get('sha256')}, manifest pins {row.get('expected_sha256')}"
            )
        if row["required"]:
            ok = False
    if not ok:
        print()
        print("Video reconstruction cannot run until the required artifacts above are in place.")
        print("Install them from wherever this deployment keeps them:")
        print("  python deploy/provision_models.py --source <directory-or-https-base>")
        print(f"Files must land in {models_dir} (a symlink there is followed to its target).")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory the application resolves model files from (default: cv_lab/models).",
    )
    parser.add_argument(
        "--source",
        help="Directory or https base URL holding the artifacts to install.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only check what is installed against the manifest.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also install artifacts the product reconstruction path does not need.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    if not args.verify and not args.source:
        parser.error("pass --source to install, or --verify to check what is installed")

    try:
        artifacts = load_manifest()
        models_dir = args.models_dir.expanduser()
        if args.source:
            installed = install(
                models_dir, artifacts, args.source, include_optional=args.include_optional
            )
            for name in installed:
                print(f"installed {name}")
            # Install mode reports what it provisions. The artifacts that ship
            # with the checkout are deliberately absent from a bare weights
            # directory, and reporting them as missing there would be noise.
            artifacts = [a for a in artifacts if not a.shipped_in_repository]
        rows = inspect(models_dir, artifacts)
    except ProvisioningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"models_dir": str(models_dir), "artifacts": rows}, indent=2))
        return 0 if all(r["state"] == "ok" for r in rows if r["required"]) else 1
    return 0 if _report(rows, models_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
