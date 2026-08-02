# Build arguments used in a FROM must be declared before the first stage.
ARG TEXASSOLVER_COMMIT=42313c9cce96130d2341a8fc265160f580956054
# bundled | none. "none" produces the base image the licensing gate does not
# block: no TexasSolver source, binary or license is fetched or shipped.
ARG SOLVER_VARIANT=bundled

FROM debian:bookworm-slim AS texassolver-build
ARG TEXASSOLVER_COMMIT

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates cmake git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch console https://github.com/bupticybee/TexasSolver.git /src/TexasSolver \
    && cd /src/TexasSolver \
    && git checkout "${TEXASSOLVER_COMMIT}" \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target install --parallel 2

# Two interchangeable payload stages so SOLVER_VARIANT can select one below.
# BuildKit builds only the stage the final image reaches, so SOLVER_VARIANT=none
# never clones or compiles TexasSolver at all -- which is what "do not publish a
# solver-enabled image until its licensing gate is satisfied" needs from a build.
FROM texassolver-build AS solver-payload-bundled
RUN mkdir -p /payload \
    && cp -a /src/TexasSolver/install/console_solver /payload/console_solver \
    && cp -a /src/TexasSolver/install/resources /payload/resources \
    && cp -a /src/TexasSolver/LICENSE /payload/LICENSE \
    && echo bundled > /payload/SOLVER_VARIANT

FROM debian:bookworm-slim AS solver-payload-none
RUN mkdir -p /payload && echo none > /payload/SOLVER_VARIANT

FROM solver-payload-${SOLVER_VARIANT} AS solver-payload

# Python dependencies are built here and copied as a finished virtualenv, so the
# compiler that linux/arm64 needs for eval7 never reaches the runtime image.
FROM python:3.11-slim AS python-build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv /opt/venv

WORKDIR /build
COPY requirements.txt requirements-cv.txt ./
COPY deploy/docker/build_python_env.sh ./
RUN sh ./build_python_env.sh requirements.txt requirements-cv.txt

FROM python:3.11-slim AS runtime

# HOME, MPLCONFIGDIR and YOLO_CONFIG_DIR are pointed at the data mount because
# every one of them is written to at runtime. Leaving them under the image's own
# filesystem is what would make `--read-only` fail, and `--read-only` is the only
# mechanical proof that no runtime write depends on the application layer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    PORT=8501 \
    HOME=/data/home \
    POKER_DB_PATH=/data/poker_tracker.db \
    POKER_DATA_DIR=/data \
    POKERTRAINER_REQUIRE_AUTH=true \
    POKERTRAINER_SOLVER_THREADS=2 \
    POKERTRAINER_SOLVER_MEMORY_GB=8 \
    POKERTRAINER_CV_TIMEOUT_SECONDS=3600 \
    TEXAS_SOLVER_PATH=/opt/texassolver/console_solver \
    MPLCONFIGDIR=/data/.matplotlib \
    YOLO_CONFIG_DIR=/data/.ultralytics

# tini is PID 1 so SIGTERM reaches Streamlit and the detached CV/solver workers
# are reaped; without it the shell that used to be PID 1 forwarded nothing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libgomp1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 pokertrainer \
    && useradd --system --uid 10001 --gid pokertrainer --no-create-home --home-dir /data/home pokertrainer

COPY --from=python-build /opt/venv /opt/venv
COPY --from=solver-payload /payload/ /opt/texassolver/

WORKDIR /app

COPY app.py ./
COPY poker_tracker ./poker_tracker
COPY cv_lab/scripts ./cv_lab/scripts
# The only model artifact small enough to track in git; .gitignore negates it
# out of cv_lab/models/* for exactly this reason.
COPY cv_lab/models/ocr_templates.npz ./cv_lab/models/ocr_templates.npz
COPY .streamlit ./.streamlit
COPY deploy/model_manifest.json deploy/provision_models.py ./deploy/
COPY deploy/docker/entrypoint.sh /usr/local/bin/pokertrainer-entrypoint

# The YOLO weights and the template banks are tens of megabytes of trained
# artifacts the repository does not carry, so the image cannot contain them and
# a clean checkout must still build. They are resolved through symlinks into the
# data mount, which is also where the project keeps every other large operator
# artifact. An unprovisioned deployment leaves these dangling: the pipeline then
# fails to load a model instead of silently reading something else, and the
# entrypoint says which files are missing and how to install them.
RUN chmod +x /usr/local/bin/pokertrainer-entrypoint \
    && for weight in region_spine_v1.pt card_cls_v1.pt card_templates.npz pot_digits.npz; do \
        ln -s "/data/models/${weight}" "/app/cv_lab/models/${weight}"; \
    done \
    && mkdir -p /data \
    && chown pokertrainer:pokertrainer /data

# /app is deliberately left root-owned and not writable by uid 10001. A runtime
# write into the application layer is a bug -- the file would be invisible to the
# data mount and lost on the next container replacement -- so it should fail
# where it happens rather than succeed and disappear later.
USER pokertrainer
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8501') + '/_stcore/health', timeout=3)"

# No CMD: the entrypoint owns the default command because ${PORT} has to be
# expanded at runtime, and it runs the startup preflight either way. Passing a
# command (`docker run <image> pytest ...`) still works and is still preflighted.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/pokertrainer-entrypoint"]
