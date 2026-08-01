"""What the container definition has to keep true, checked without a daemon.

The image had never been built from a clean checkout, and two independent
defects meant it could not have been: four of its COPY sources were gitignored
files that only existed on one developer's disk, and the sole build stage had no
compiler for the one dependency that publishes no aarch64 wheel. Both were
invisible to the test suite because nothing in it had ever read the Dockerfile.

These checks read it. They cannot prove a build succeeds -- that needs a Docker
daemon and lives in deploy/verify_container.sh -- but they do prove the two
conditions under which it certainly fails, and they fail the moment either is
reintroduced.

Run: pytest deploy/tests/test_container_build_contract.py
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
ROOT_COMPOSE = REPO_ROOT / "compose.yaml"
OCI_COMPOSE = REPO_ROOT / "deploy" / "oci" / "compose.yaml"
MANIFEST = REPO_ROOT / "deploy" / "model_manifest.json"
BUILD_ENV_SCRIPT = REPO_ROOT / "deploy" / "docker" / "build_python_env.sh"


def _logical_lines(text: str) -> list[str]:
    """Dockerfile instructions with line continuations folded and comments cut."""
    lines: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not buffer and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        buffer += stripped
        lines.append(re.sub(r"\s+", " ", buffer).strip())
        buffer = ""
    if buffer:
        lines.append(re.sub(r"\s+", " ", buffer).strip())
    return lines


@pytest.fixture(scope="module")
def instructions() -> list[str]:
    return _logical_lines(DOCKERFILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tracked_paths() -> set[str]:
    """Paths a fresh clone would have: tracked, plus untracked-and-not-ignored.

    Untracked files are included because a working-tree change is not yet a
    commit; what disqualifies a path is being ignored, which is what made the
    model weights unavailable to everyone but the machine that trained them.
    """
    output = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {line for line in output.splitlines() if line}


def _copy_sources(instructions: list[str]) -> list[str]:
    """Every host path a COPY reads, excluding cross-stage copies."""
    sources: list[str] = []
    for line in instructions:
        if not line.upper().startswith("COPY "):
            continue
        parts = shlex.split(line)[1:]
        if any(part.startswith("--from=") for part in parts):
            continue
        parts = [part for part in parts if not part.startswith("--")]
        sources.extend(parts[:-1])
    return sources


def _stages(instructions: list[str]) -> list[list[str]]:
    stages: list[list[str]] = []
    for line in instructions:
        if line.upper().startswith("FROM "):
            stages.append([])
        if stages:
            stages[-1].append(line)
    return stages


def _final_stage(instructions: list[str]) -> list[str]:
    stages = _stages(instructions)
    return stages[-1] if stages else []


def test_every_copy_source_exists_in_a_clean_checkout(instructions, tracked_paths):
    """A COPY of an untracked path makes the image unbuildable by anyone else.

    This is the defect that kept `git clone && docker build .` from ever
    working: the weights lived only on the machine that trained them.
    """
    missing = []
    for source in _copy_sources(instructions):
        normalised = (source[2:] if source.startswith("./") else source).rstrip("/")
        if not normalised or normalised == ".":
            continue
        if normalised in tracked_paths:
            continue
        prefix = normalised + "/"
        if any(path.startswith(prefix) for path in tracked_paths):
            continue
        missing.append(source)
    assert missing == [], (
        "Dockerfile COPYs paths that a clean checkout does not have: "
        f"{missing}. Ship them from a tracked location, or resolve them at "
        "runtime through deploy/provision_models.py."
    )


def test_dockerignore_keeps_untracked_weights_out_of_the_build_context():
    """The build context must contain what a clean checkout contains.

    Without this, a COPY of an untracked weight keeps succeeding on the one
    machine that holds it, which is exactly how the dependency stayed hidden.
    """
    rules = [
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "cv_lab/models/*" in rules
    assert "!cv_lab/models/ocr_templates.npz" in rules
    assert rules.index("cv_lab/models/*") < rules.index("!cv_lab/models/ocr_templates.npz")


def test_the_stage_that_installs_python_packages_has_a_compiler(instructions):
    """eval7 publishes no aarch64 wheel, so arm64 must compile it.

    The previous layout had a compiler only in the TexasSolver stage, which does
    not help pip: the stage that installs requirements had none, so linux/arm64
    died in the eval7 sdist build. Asserting on "some stage somewhere" would
    have passed then, so this asserts on the stage that actually runs pip.
    """
    compiler_markers = ("build-essential", "gcc", "g++")
    python_stages = [
        stage
        for stage in _stages(instructions)
        if any("build_python_env.sh" in line for line in stage)
    ]
    assert python_stages, "no stage runs the dependency build script"
    for stage in python_stages:
        assert any(marker in line for line in stage for marker in compiler_markers), (
            "the stage that installs Python requirements has no compiler, "
            "so linux/arm64 cannot build eval7"
        )

    final = _final_stage(instructions)
    assert not any(
        marker in line
        for line in final
        if line.upper().startswith("RUN ")
        for marker in compiler_markers
    ), "the runtime stage installs a compiler; keep the toolchain in a build stage"


def test_runtime_python_environment_comes_from_a_build_stage(instructions):
    final = _final_stage(instructions)
    assert any(
        line.upper().startswith("COPY --FROM=") and "/opt/venv" in line for line in final
    ), "the runtime stage must copy a prebuilt virtualenv rather than pip install"
    assert not any(
        "pip install" in line for line in final
    ), "the runtime stage must not install Python packages; it has no compiler"


def test_eval7_source_build_has_cython_and_no_build_isolation():
    """The failure mode that a compiler alone does not fix.

    eval7's sdist imports Cython from setup.py but declares no build backend, so
    pip's isolated build environment holds only setuptools and raises
    ModuleNotFoundError before the compiler is ever used.
    """
    script = BUILD_ENV_SCRIPT.read_text(encoding="utf-8")
    assert "Cython" in script
    assert "--no-build-isolation" in script
    isolation_line = next(
        line for line in script.splitlines() if "--no-build-isolation" in line and "install" in line
    )
    assert "eval7" in isolation_line, (
        "build isolation should be disabled for eval7 specifically, not for every requirement"
    )


def test_application_layer_is_not_writable_by_the_runtime_user(instructions):
    """A runtime write into /app succeeds silently and is lost on replacement."""
    offenders = [
        line
        for line in instructions
        if "chown" in line and re.search(r"chown[^&|]*\s/app(\s|$)", line)
    ]
    assert offenders == [], (
        f"the image hands /app to the runtime user: {offenders}. Keep /app "
        "root-owned so a stray runtime write fails where it happens."
    )


def test_container_runs_unprivileged_with_a_healthcheck(instructions):
    user_index = next(
        (i for i, line in enumerate(instructions) if line.upper().startswith("USER ")), None
    )
    assert user_index is not None, "the image must declare a non-root USER"
    assert "root" not in instructions[user_index].lower()
    entry_index = next(
        (
            i
            for i, line in enumerate(instructions)
            if line.upper().startswith(("CMD ", "ENTRYPOINT "))
        ),
        None,
    )
    assert entry_index is not None and user_index < entry_index
    assert any(line.upper().startswith("HEALTHCHECK") for line in instructions)


def test_pid_one_forwards_signals(instructions):
    """Shutdown has to reach Streamlit and reap the detached heavy workers."""
    entrypoint = next(
        (line for line in instructions if line.upper().startswith("ENTRYPOINT ")), ""
    )
    assert "tini" in entrypoint, (
        "PID 1 must be an init that forwards SIGTERM; a bare shell forwards nothing"
    )


def test_solver_can_be_left_out_of_the_image(instructions):
    """The licensing gate forbids publishing a solver image, so one must exist
    that carries no TexasSolver at all."""
    assert any(
        line.upper().startswith("ARG ") and "SOLVER_VARIANT" in line for line in instructions
    )
    assert any("solver-payload-none" in line for line in instructions)
    assert any(
        line.upper().startswith("FROM SOLVER-PAYLOAD-${SOLVER_VARIANT}".upper())
        for line in instructions
    )


def test_manifest_covers_every_model_file_the_code_resolves():
    """A new weight added to the pipeline without a manifest entry is a weight
    nobody can provision."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    listed = {entry["filename"] for entry in manifest["artifacts"]}
    referenced: set[str] = set()
    # Scoped to the runtime pipeline: training scripts name checkpoints that are
    # inputs to training, not artifacts a deployment has to be given.
    for path in (REPO_ROOT / "cv_lab" / "scripts" / "pipeline").rglob("*.py"):
        for match in re.finditer(
            r"cv_lab[/\"'\s,]+models[/\"'\s,]+([A-Za-z0-9_.() ]+\.(?:pt|npz))",
            path.read_text(encoding="utf-8", errors="ignore"),
        ):
            referenced.add(match.group(1).strip())
    # A bank the repository no longer carries is not the manifest's problem;
    # only files that actually exist are required to be listed.
    resolvable = {name for name in referenced if (REPO_ROOT / "cv_lab" / "models" / name).exists()}
    assert resolvable <= listed, (
        f"model files resolved by cv_lab/scripts but absent from the manifest: "
        f"{sorted(resolvable - listed)}"
    )


def test_manifest_digests_are_well_formed():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry["filename"]
        assert entry["bytes"] > 0


def test_a_provider_neutral_compose_exists_and_stays_neutral():
    compose = yaml.safe_load(ROOT_COMPOSE.read_text(encoding="utf-8"))
    app = compose["services"]["app"]
    assert app["build"]["context"] == "."
    assert app["restart"] == "unless-stopped"
    assert "healthcheck" in app
    mounts = app["volumes"]
    assert any(str(mount).startswith("./data:/data") for mount in mounts), mounts
    for mount in mounts:
        assert not str(mount).startswith("/"), f"host-absolute path in a neutral compose: {mount}"
    assert set(compose["services"]) == {"app"}, (
        "the neutral compose must not require a provider-specific companion service"
    )


def test_compose_files_do_not_pin_port_over_the_env_file():
    """Compose merges env_file first and `environment` second, so a literal
    there silently beats what the operator set."""
    for path in (ROOT_COMPOSE, OCI_COMPOSE):
        compose = yaml.safe_load(path.read_text(encoding="utf-8"))
        port = compose["services"]["app"].get("environment", {}).get("PORT")
        assert port is None or "${" in str(port), f"{path} pins PORT to {port!r}"
