"""Phase 12/8: the build must be able to say what it ships and under what terms.

The inventory is read from installed distributions rather than requirements.txt.
A requirements file records what was asked for; an SBOM has to record what is
actually present, including transitive dependencies no requirements file names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poker_tracker.maintenance.sbom import (
    Component,
    build_inventory,
    collect_components,
    collect_models,
    main,
    to_cyclonedx,
    to_notices,
)

REPO = Path(__file__).resolve().parents[1]


def test_inventory_includes_transitive_dependencies():
    """Packages nothing in requirements.txt names must still appear."""
    names = {component.name.lower() for component in collect_components()}
    assert "streamlit" in names
    assert "numpy" in names
    # A transitive dependency of streamlit, named in no requirements file.
    assert names & {"tornado", "click", "packaging"}


def test_every_component_reports_a_version_and_a_license():
    for component in collect_components():
        assert component.version, f"{component.name} has no version"
        assert component.license_text, f"{component.name} has no license string"


def test_unknown_licenses_are_reported_as_unknown_not_guessed():
    """"UNKNOWN" tells a reviewer exactly which packages need a human."""
    component = Component(name="x", version="1", license_text="UNKNOWN")
    assert component.needs_license_review is False
    assert "UNKNOWN" in to_notices(
        type(build_inventory(REPO))(components=[component], models=[])
    )


# --- The licensing gate -----------------------------------------------------


@pytest.mark.parametrize(
    ("license_text", "expected"),
    [
        ("AGPL-3.0", True),
        ("GPL-3.0-only", True),
        ("SSPL-1.0", True),
        ("MIT", False),
        ("Apache-2.0", False),
        ("BSD-3-Clause", False),
        # Dynamic linking against LGPL does not carry the same distribution
        # obligation, so flagging it identically would bury the real cases.
        ("LGPL-2.1", False),
    ],
)
def test_copyleft_detection(license_text, expected):
    assert Component("x", "1", license_text).needs_license_review is expected


def test_the_agpl_dependency_the_cv_pipeline_needs_is_flagged():
    """ultralytics is AGPL-3.0 and the reconstruction pipeline requires it.

    This is a release blocker for any DISTRIBUTED image, in the same class as
    TexasSolver's license, and it was previously unrecorded anywhere.
    """
    flagged = {c.name.lower() for c in build_inventory(REPO).needs_review}
    assert "ultralytics" in flagged


def test_fail_on_review_exits_nonzero(capsys):
    assert main(["--format", "notices", "--fail-on-review"]) == 1
    assert "license review" in capsys.readouterr().err


def test_notices_name_the_solver_gate_even_though_it_is_not_a_python_package():
    notices = to_notices(build_inventory(REPO))
    assert "TexasSolver" in notices
    assert "release blocker" in notices


# --- Models -----------------------------------------------------------------


def test_models_are_inventoried_with_hashes():
    models = collect_models(REPO)
    assert models, "no model weights found to inventory"
    for model in models:
        assert len(model["sha256"]) == 64
        assert model["bytes"] > 0
        assert model["origin"]


def test_absent_models_are_simply_absent(tmp_path: Path):
    assert collect_models(tmp_path) == []


# --- Output formats ---------------------------------------------------------


def test_cyclonedx_is_valid_json_with_the_expected_shape():
    document = to_cyclonedx(build_inventory(REPO))
    round_tripped = json.loads(json.dumps(document))
    assert round_tripped["bomFormat"] == "CycloneDX"
    assert round_tripped["components"]
    types = {component["type"] for component in round_tripped["components"]}
    assert "library" in types
    assert "machine-learning-model" in types


def test_cli_emits_both_formats(capsys):
    assert main(["--format", "cyclonedx"]) == 0
    assert json.loads(capsys.readouterr().out)["bomFormat"] == "CycloneDX"

    assert main(["--format", "notices"]) == 0
    assert "THIRD-PARTY NOTICES" in capsys.readouterr().out
