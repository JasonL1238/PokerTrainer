"""Behaviour of the model provisioning step the container build now depends on.

The image ships without the CV weights, so this script is the only thing
standing between a deployment and a model file that is truncated, substituted or
silently written into the wrong place. Each case here is a way that could go
wrong quietly.

Run: pytest deploy/tests/test_provision_models.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# Loaded by path because deploy/ is a scripts directory, not an importable
# package; the module has to be in sys.modules before execution so its
# dataclasses can resolve their own annotations.
_spec = importlib.util.spec_from_file_location(
    "provision_models", REPO_ROOT / "deploy" / "provision_models.py"
)
assert _spec and _spec.loader
provision_models = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = provision_models
_spec.loader.exec_module(provision_models)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def corpus(tmp_path: Path):
    """A source directory, a manifest describing it, and an empty models dir."""
    source = tmp_path / "source"
    source.mkdir()
    weight = b"weights" * 100
    (source / "weight_v1.pt").write_bytes(weight)
    shipped = b"templates"
    (source / "shipped.npz").write_bytes(shipped)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "role": "detector",
                        "filename": "weight_v1.pt",
                        "sha256": _digest(weight),
                        "bytes": len(weight),
                        "required": True,
                        "shipped_in_repository": False,
                    },
                    {
                        "role": "templates",
                        "filename": "shipped.npz",
                        "sha256": _digest(shipped),
                        "bytes": len(shipped),
                        "required": True,
                        "shipped_in_repository": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    models = tmp_path / "models"
    models.mkdir()
    return source, provision_models.load_manifest(manifest_path), models


def test_a_file_whose_digest_does_not_match_is_never_installed(corpus, tmp_path):
    """A substituted or truncated weight has to be refused, not installed.

    A wrong weight does not announce itself: it produces card reads that look
    like ordinary output, which is the failure this whole phase exists to make
    loud.
    """
    source, artifacts, models = corpus
    (source / "weight_v1.pt").write_bytes(b"a different model entirely")

    with pytest.raises(provision_models.ProvisioningError) as excinfo:
        provision_models.install(models, artifacts, str(source), include_optional=False)

    assert "weight_v1.pt" in str(excinfo.value)
    assert not (models / "weight_v1.pt").exists()


def test_a_partial_transfer_leaves_no_installed_file(corpus, monkeypatch):
    """An interrupted copy must not leave something a later run treats as done."""
    source, artifacts, models = corpus

    def explode(*_args, **_kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr(provision_models.shutil, "copyfile", explode)
    with pytest.raises(OSError):
        provision_models.install(models, artifacts, str(source), include_optional=False)

    assert list(models.iterdir()) == []


def test_installing_writes_through_a_symlink_instead_of_replacing_it(corpus, tmp_path):
    """The container resolves cv_lab/models through symlinks into the data mount.

    Replacing the symlink with a regular file would put the weight in the
    read-only application layer, where it is lost on the next container
    replacement.
    """
    source, artifacts, models = corpus
    mount = tmp_path / "mount"
    mount.mkdir()
    (models / "weight_v1.pt").symlink_to(mount / "weight_v1.pt")

    provision_models.install(models, artifacts, str(source), include_optional=False)

    assert (models / "weight_v1.pt").is_symlink()
    assert (mount / "weight_v1.pt").is_file()


def test_artifacts_that_ship_with_the_repository_are_not_provisioned(corpus):
    source, artifacts, models = corpus
    installed = provision_models.install(
        models, artifacts, str(source), include_optional=False
    )
    assert installed == ["weight_v1.pt"]
    assert not (models / "shipped.npz").exists()


def test_inspect_names_a_missing_required_artifact(corpus):
    source, artifacts, models = corpus
    rows = {row["filename"]: row for row in provision_models.inspect(models, artifacts)}
    assert rows["weight_v1.pt"]["state"] == "missing"
    assert rows["weight_v1.pt"]["required"] is True


def test_inspect_reports_a_corrupted_installed_artifact_as_a_mismatch(corpus):
    source, artifacts, models = corpus
    provision_models.install(models, artifacts, str(source), include_optional=False)
    (models / "weight_v1.pt").write_bytes(b"corrupted on disk")

    rows = {row["filename"]: row for row in provision_models.inspect(models, artifacts)}
    assert rows["weight_v1.pt"]["state"] == "digest_mismatch"


def test_plain_http_sources_are_refused(corpus):
    """An unauthenticated transport for a file that decides what the cards say."""
    source, artifacts, models = corpus
    with pytest.raises(provision_models.ProvisioningError) as excinfo:
        provision_models.install(
            models, artifacts, "http://models.example.com/v1", include_optional=False
        )
    assert "https" in str(excinfo.value)


def test_the_shipped_manifest_describes_the_files_this_checkout_has():
    """The manifest is only useful if its digests are the real ones.

    Skipped rather than failed when the untracked weights are absent: that is a
    clean checkout, which is a supported state.
    """
    artifacts = provision_models.load_manifest()
    models_dir = REPO_ROOT / "cv_lab" / "models"
    checked = 0
    for artifact in artifacts:
        path = models_dir / artifact.filename
        if not path.is_file():
            continue
        assert provision_models.sha256_file(path) == artifact.sha256, artifact.filename
        assert path.stat().st_size == artifact.bytes, artifact.filename
        checked += 1
    if checked == 0:
        pytest.skip("no model artifacts present in this checkout")
