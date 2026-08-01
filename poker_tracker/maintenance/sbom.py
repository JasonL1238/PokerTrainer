"""Generate a dependency inventory and third-party notices.

PLAN.md Phase 12 requires a dependency inventory/SBOM for distributable
containers and a record of model origins, hashes and licenses. Phase 8 requires
third-party notices for any distributable artifact, and treats TexasSolver's
license as a release blocker for a bundled solver image.

    python -m poker_tracker.maintenance.sbom --format cyclonedx > sbom.json
    python -m poker_tracker.maintenance.sbom --format notices > NOTICES.txt

The inventory is read from the installed distributions, not from
requirements.txt: a requirements file records what was asked for, and an SBOM
has to record what is actually present, including transitive dependencies that
no requirements file names.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from poker_tracker.release_gate.models import MODEL_CANDIDATES
from poker_tracker.validation.hashing import sha256_file

SBOM_SPEC_VERSION = "1.5"

# Licenses that make a distributed image a licensing question rather than a
# formality. Presence here is not legal advice; it marks what needs review.
COPYLEFT_MARKERS: tuple[str, ...] = (
    "AGPL",
    "GPL",
    "SSPL",
    "OSL",
    "EUPL",
)


@dataclass
class Component:
    name: str
    version: str
    license_text: str
    author: str = ""
    homepage: str = ""

    @property
    def needs_license_review(self) -> bool:
        upper = self.license_text.upper()
        # LGPL is separated out: dynamic linking against it does not carry the
        # same distribution obligation as GPL/AGPL, so flagging it identically
        # would bury the cases that matter.
        if "LGPL" in upper:
            return False
        return any(marker in upper for marker in COPYLEFT_MARKERS)


@dataclass
class Inventory:
    components: list[Component] = field(default_factory=list)
    models: list[dict[str, Any]] = field(default_factory=list)

    @property
    def needs_review(self) -> list[Component]:
        return [c for c in self.components if c.needs_license_review]


def _license_of(dist: metadata.Distribution) -> str:
    """Best available license string.

    Packaging metadata is inconsistent here: some distributions fill
    ``License``, some leave it blank and use a Trove classifier, and some do
    neither. Reporting "UNKNOWN" is more useful than guessing, because it tells
    a reviewer exactly which packages still need a human to look.
    """
    meta = dist.metadata
    declared = (meta.get("License") or "").strip()
    if declared and declared.lower() not in {"unknown", "none"}:
        # Some projects paste the whole license text into this field.
        return declared.splitlines()[0][:200]
    classifiers = meta.get_all("Classifier") or []
    for classifier in classifiers:
        if classifier.startswith("License ::"):
            return classifier.split("::")[-1].strip()
    for key in ("License-Expression", "License-File"):
        value = (meta.get(key) or "").strip()
        if value:
            return value
    return "UNKNOWN"


def collect_components() -> list[Component]:
    components: list[Component] = []
    seen: set[str] = set()
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        components.append(
            Component(
                name=name,
                version=(dist.version or "").strip(),
                license_text=_license_of(dist),
                author=(dist.metadata.get("Author") or "").strip(),
                homepage=(dist.metadata.get("Home-page") or "").strip(),
            )
        )
    return sorted(components, key=lambda c: c.name.lower())


def collect_models(repo_root: Path) -> list[dict[str, Any]]:
    """Model weights with their hashes, so an image identifies what it ships."""
    entries: list[dict[str, Any]] = []
    for role, candidates in MODEL_CANDIDATES.items():
        for relative in candidates:
            path = repo_root / relative
            if not path.is_file():
                continue
            entries.append(
                {
                    "role": role,
                    "path": relative,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    # Trained in this repository from the private ClubWPT
                    # corpus; no third-party weights are redistributed.
                    "origin": "trained in-repo (cv_lab)",
                    "license": "same as this repository",
                }
            )
            break
    return entries


def build_inventory(repo_root: Path) -> Inventory:
    return Inventory(components=collect_components(), models=collect_models(repo_root))


def to_cyclonedx(inventory: Inventory) -> dict[str, Any]:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": SBOM_SPEC_VERSION,
        "version": 1,
        "metadata": {
            "component": {"type": "application", "name": "pokertrainer"},
            "properties": [
                {"name": "python", "value": sys.version.split()[0]},
            ],
        },
        "components": [
            {
                "type": "library",
                "name": component.name,
                "version": component.version,
                "licenses": [{"license": {"name": component.license_text}}],
                "purl": f"pkg:pypi/{component.name.lower()}@{component.version}",
            }
            for component in inventory.components
        ]
        + [
            {
                "type": "machine-learning-model",
                "name": model["role"],
                "version": model["sha256"][:12],
                "licenses": [{"license": {"name": model["license"]}}],
                "properties": [
                    {"name": "sha256", "value": model["sha256"]},
                    {"name": "origin", "value": model["origin"]},
                ],
            }
            for model in inventory.models
        ],
    }


def to_notices(inventory: Inventory) -> str:
    lines = [
        "THIRD-PARTY NOTICES",
        "",
        "This file lists the third-party components present in this build and",
        "the license each one declares. Declarations come from package metadata",
        "and are not a substitute for reading the licenses themselves.",
        "",
    ]
    for component in inventory.components:
        entry = f"{component.name} {component.version} — {component.license_text}"
        if component.homepage:
            entry += f" ({component.homepage})"
        lines.append(entry)

    review = inventory.needs_review
    lines.extend(["", "COMPONENTS REQUIRING LICENSE REVIEW BEFORE DISTRIBUTION", ""])
    if not review:
        lines.append("None detected by keyword. This is not a legal review.")
    else:
        for component in review:
            lines.append(f"  {component.name} {component.version} — {component.license_text}")
    lines.extend(
        [
            "",
            "TexasSolver is NOT included in this inventory unless it is installed",
            "as a Python distribution. It is an optional, separately-obtained",
            "native binary. PLAN.md treats its license as a release blocker for",
            "any distributed or hosted solver-enabled image: obtain written",
            "maintainer permission or a reviewed compliance approach before",
            "publishing such an image.",
            "",
            "MODELS",
            "",
        ]
    )
    for model in inventory.models:
        lines.append(
            f"  {model['role']}: {model['path']} sha256={model['sha256']} "
            f"({model['origin']})"
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("cyclonedx", "notices"),
        default="cyclonedx",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Exit 1 when any component's license needs review before distribution.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    inventory = build_inventory(args.repo_root)
    if args.format == "notices":
        sys.stdout.write(to_notices(inventory))
    else:
        json.dump(to_cyclonedx(inventory), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    if args.fail_on_review and inventory.needs_review:
        names = ", ".join(c.name for c in inventory.needs_review)
        print(f"Components needing license review: {names}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
