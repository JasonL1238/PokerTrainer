# Container build, provisioning and verification

The canonical operator entrypoint is [README.md](../README.md); the operational
procedures are in [RUNBOOKS.md](RUNBOOKS.md). This document covers the one thing
neither of them can state briefly: what a container build needs that a `git
clone` does not contain, and how to prove an image actually works before
trusting it.

## What a clean checkout can and cannot build

`git clone && docker build .` builds a complete image. It has to, and until
recently it did not: the Dockerfile copied four model artifacts that `.gitignore`
excludes, so the build only ever succeeded on the machine that trained them.

The image therefore ships **without** the large CV model weights. They are
resolved at runtime through symlinks in `/app/cv_lab/models` that point into
`/data/models` on the persistent mount — the same place the project keeps every
other large operator artifact (SQLite, videos, frames, timelines, backups).

| Artifact | Where it lives | Consequence when missing |
| --- | --- | --- |
| `region_spine_v1.pt` | `/data/models` (mount) | Reconstruction refuses to start |
| `card_cls_v1.pt` | `/data/models` (mount) | Reconstruction refuses to start |
| `ocr_templates.npz` | inside the image (tracked in git) | n/a |
| `card_templates.npz`, `pot_digits.npz` | `/data/models` (mount, optional) | `cv_lab` evaluation scripts only |

`deploy/model_manifest.json` names each artifact with the SHA-256 that
identifies it. Nothing is fetched implicitly; the operator says where the files
come from.

```bash
# from a directory (USB drive, restored backup, the training machine)
python deploy/provision_models.py --source /media/backup/pokertrainer-models

# or from any https location this deployment controls
python deploy/provision_models.py --source https://models.example.internal/pokertrainer/v1

# check what is installed at any time
python deploy/provision_models.py --verify
```

Every file is written to a temporary name and only renamed into place after its
digest matches the manifest, so an interrupted or substituted download can never
be mistaken for an installed weight.

Inside a running container the same command writes through the symlinks onto the
mount:

```bash
docker compose exec app python /app/deploy/provision_models.py --source /data/incoming
```

A container that starts without the required weights says so at startup and runs
everything except video reconstruction. Set `POKERTRAINER_REQUIRE_MODELS=true`
to make that a refusal to start instead.

## Both architectures

`linux/amd64` and `linux/arm64` are both supported paths. arm64 is the harder
one: `eval7` publishes no aarch64 wheel and is a Cython C extension, so arm64
has to compile it. Two things make that work, and both are easy to undo by
accident.

1. A dedicated `python-build` stage carries `build-essential` and produces a
   finished virtualenv that the runtime stage copies. The runtime image has no
   compiler.
2. `eval7`'s sdist imports Cython from `setup.py` but declares no build backend,
   so pip's isolated build environment — setuptools and wheel only — fails with
   `ModuleNotFoundError: No module named 'Cython'` before any compiler is
   reached. `deploy/docker/build_python_env.sh` installs Cython into the build
   stage and disables build isolation for that single requirement.

To confirm no other dependency has acquired the same problem:

```bash
python deploy/check_wheel_availability.py
```

It resolves both architectures with wheels only and names anything that would
have to be compiled. `eval7` is expected; anything else means
`build_python_env.sh` needs extending before the arm64 image will build.

## Running it

```bash
cp deploy/.env.example .env      # set APP_PASSWORD
mkdir -p data && sudo chown -R 10001:10001 data
docker compose up -d
docker compose ps                # wait for "healthy"
```

`compose.yaml` at the repository root is the provider-neutral definition:
loopback-bound port, `./data` bind mount, `restart: unless-stopped`, the image's
own healthcheck, and `POKERTRAINER_REQUIRE_DATA_MOUNT=true` so a missing mount
is refused rather than silently filling a disposable container layer.
`deploy/oci/` remains an optional provider-specific reference layered on the
same image.

### Environment variables the container reads

| Variable | Default in the image | Purpose |
| --- | --- | --- |
| `PORT` | `8501` | Listen port, also used by the healthcheck |
| `APP_PASSWORD` | unset | Required, because `POKERTRAINER_REQUIRE_AUTH` is true |
| `POKER_DATA_DIR` | `/data` | Root of every durable artifact |
| `POKER_DB_PATH` | `/data/poker_tracker.db` | SQLite database |
| `POKERTRAINER_SOLVER_THREADS` | `2` | TexasSolver thread ceiling |
| `POKERTRAINER_SOLVER_MEMORY_GB` | `8` | TexasSolver address-space ceiling |
| `POKERTRAINER_REQUIRE_DATA_MOUNT` | unset | `true` refuses to start without a mount at `/data` |
| `POKERTRAINER_REQUIRE_MODELS` | unset | `true` refuses to start without verified CV weights |
| `HOME`, `MPLCONFIGDIR`, `YOLO_CONFIG_DIR` | under `/data` | Every runtime write goes to the mount, never `/app` |

### Build arguments

| Argument | Default | Effect |
| --- | --- | --- |
| `SOLVER_VARIANT` | `bundled` | `none` builds an image with no TexasSolver source, binary or license in it. Under BuildKit the solver stage is never built at all. |
| `TEXASSOLVER_COMMIT` | pinned | The console-solver revision compiled into a bundled image. |

The licensing gate (`python -m poker_tracker.maintenance.sbom --fail-on-review`)
blocks publishing a solver-enabled image. `SOLVER_VARIANT=none` is the image
that gate does not block.

## Verifying an image

```bash
deploy/verify_container.sh --all-architectures --report container-verification.json
```

The script needs a Docker daemon, buildx and — for the foreign architecture —
QEMU. It builds from a `git archive` of `HEAD` in a temporary directory rather
than the working tree, because a working tree contains untracked files and that
is precisely what hid the clean-context defect. For each architecture it records
image size and checks:

- the build succeeds from a clean context;
- the image reports the architecture that was requested;
- `eval7` imports, proving the compiled extension matches the architecture;
- the container runs as uid 10001;
- the runtime image contains no compiler;
- ffmpeg is usable;
- the container becomes healthy **with `--read-only`**, which is the only
  mechanical proof that no runtime write depends on the application layer;
- the SQLite database is created on the bind mount, not in the image;
- a restart drill: `docker restart` returns to healthy with the database intact;
- container memory usage;
- `SOLVER_VARIANT=none` builds, ships no solver binary, and reports an
  actionable configuration error instead of failing.

`deploy/tests/` is the daemon-free half of the same contract:

```bash
pytest deploy/tests
```

`test_container_build_contract.py` reads the Dockerfile and both compose files
and fails if a COPY source stops being obtainable from a clean checkout, if the
compiler leaks into the runtime stage, if the eval7 source build loses its
Cython handling, if `/app` becomes writable by the runtime user, if PID 1 stops
forwarding signals, or if a compose file pins `PORT` over the operator's `.env`.
`test_provision_models.py` covers the provisioning behaviour that keeps a wrong
weight from being installed quietly.

These do **not** run under a bare `pytest`: `pyproject.toml` sets
`testpaths = ["tests"]`. Until `deploy/tests` is added there, they have to be
named explicitly, in CI as well as locally.
