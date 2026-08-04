# cv_lab/ — agent instructions

Canonical rules: [../docs/agent-guidelines.md](../docs/agent-guidelines.md).
This file adds only what is different here.

CV research tooling: detector, card classifier, template OCR, timeline builders.
57 Python files, 17.8k lines — plus **3.3 GB of data you must not scan.**

## Read / never read

| Path | |
|---|---|
| `scripts/` | The code. Read this |
| `notes/` | Chronological research record. **Later findings supersede earlier ones** — read newest first, and never cite an early experiment as current |
| `labeling_poker/` | Local labeling tool |
| `datasets/ frames/ crops/ runs/ results/ models/` | **Data. Never glob, grep, or list** |

`*.pt` weights are gitignored and exist in **no checkout**. Provision with
`deploy/provision_models.py --source …`. Code that assumes weights are present
will fail in CI and in a clean clone; the one exception committed on purpose is
`models/ocr_templates.npz`.

## Boundaries

- `cv_lab` must **never** import `poker_tracker`. That direction is clean; keep it.
- The reverse is **not** clean: product code imports `cv_lab` at 6 sites
  (`ui/run_cv_job.py`, `maintenance/diagnostics.py`,
  `services/validated_hand_import.py` ×2, `perf/probes.py`, `app.py:535`), several
  as function-local imports so the CV stack stays optional at startup. **A
  signature change here can break the product**, and nothing enforces the
  boundary — grep for your symbol across `poker_tracker/` before changing it.
- Keep new function-local imports function-local. Hoisting one to module scope
  makes torch a startup dependency of the Streamlit app.

## Licensing

`ultralytics` is **AGPL-3.0** and is pinned in `deploy/docker/build_python_env.sh`
(`ULTRALYTICS_PIN`), not in any requirements file. Only `cv_lab` imports it. The
container has it; base installs and CI do not, which is why the SBOM AGPL tests
fail outside a provisioned machine — see
[../docs/testing.md](../docs/testing.md#known-failures). Do not add an AGPL
dependency to `poker_tracker/`.

## Production posture

The production read path must stay **deterministic CV**, not a vision model. A
VLM is acceptable for validation and labeling assistance only, never as the
recorded read.
