#!/bin/sh
# Container startup preflight.
#
# Everything this script reports is a condition that would otherwise surface
# much later as a confusing failure or, worse, not at all: a data directory that
# is really the container's own filesystem loses a study database on the next
# `docker rm`, and missing CV weights turn into an unexplained reconstruction
# crash an hour into a session. Each check names what is wrong and what to do.
#
# The default posture is loud rather than fatal, because the base application --
# hand review, analytics, import, export -- is fully usable without CV weights
# or a solver. Set POKERTRAINER_REQUIRE_DATA_MOUNT=true or
# POKERTRAINER_REQUIRE_MODELS=true to turn the corresponding warning into a
# refusal to start; the container verification drill sets both.
set -eu

DATA_DIR=${POKER_DATA_DIR:-/data}
APP_DIR=/app

say() { printf '%s\n' "$*" >&2; }
banner() { say ""; say "=== $* ==="; }

fatal() {
    say ""
    say "pokertrainer: refusing to start."
    for line in "$@"; do say "  $line"; done
    exit 1
}

# 1. The data directory has to be a real, writable mount.
if ! mkdir -p "$DATA_DIR" 2>/dev/null || ! [ -w "$DATA_DIR" ]; then
    fatal \
        "$DATA_DIR is not writable by the container user (uid $(id -u), gid $(id -g))." \
        "Every durable artifact -- the SQLite database, uploaded videos, frames," \
        "timelines, job logs and backups -- lives there." \
        "On the host: sudo mkdir -p <hostdir> && sudo chown -R 10001:10001 <hostdir>" \
        "then run with -v <hostdir>:$DATA_DIR"
fi

data_is_mounted() {
    # /proc/self/mountinfo lists a line per mount point; a bind mount or volume
    # at the data directory appears as its own entry, the container layer does
    # not.
    [ -r /proc/self/mountinfo ] || return 0
    awk -v target="$DATA_DIR" '$5 == target { found = 1 } END { exit found ? 0 : 1 }' \
        /proc/self/mountinfo
}

if ! data_is_mounted; then
    message_1="$DATA_DIR is inside the container filesystem, not on a mount."
    message_2="Everything written there -- database, videos, timelines, backups --"
    message_3="is destroyed when this container is removed or replaced."
    message_4="Run with -v <hostdir>:$DATA_DIR (or the volume declared in compose.yaml)."
    if [ "${POKERTRAINER_REQUIRE_DATA_MOUNT:-false}" = "true" ]; then
        fatal "$message_1" "$message_2" "$message_3" "$message_4"
    fi
    banner "WARNING: no persistent mount"
    say "  $message_1"
    say "  $message_2"
    say "  $message_3"
    say "  $message_4"
fi

# 2. Runtime state directories live on the mount, never in the application
#    layer, so the image can run with --read-only.
mkdir -p \
    "${HOME:-$DATA_DIR/home}" \
    "${MPLCONFIGDIR:-$DATA_DIR/.matplotlib}" \
    "${YOLO_CONFIG_DIR:-$DATA_DIR/.ultralytics}" \
    "$DATA_DIR/models"

# 3. CV model weights are not redistributable through the repository, so the
#    image ships without them and resolves them from the mount.
if ! python "$APP_DIR/deploy/provision_models.py" --verify; then
    if [ "${POKERTRAINER_REQUIRE_MODELS:-false}" = "true" ]; then
        fatal "Required CV model weights are missing or do not match their pinned digests." \
              "See the report above and docs/CONTAINER.md."
    fi
    banner "WARNING: video reconstruction is unavailable"
    say "  The checks above list what is missing. Every other feature works;"
    say "  a reconstruction job started now will fail rather than produce a"
    say "  partial or wrong timeline."
fi

# 4. A solver-free image must look unconfigured rather than misconfigured. The
#    application already degrades correctly on an unset TEXAS_SOLVER_PATH; a set
#    path that points at nothing would report a missing file instead of a build
#    that was never meant to have one.
if [ -n "${TEXAS_SOLVER_PATH:-}" ] && [ ! -x "${TEXAS_SOLVER_PATH}" ]; then
    if [ -f /opt/texassolver/SOLVER_VARIANT ] && [ "$(cat /opt/texassolver/SOLVER_VARIANT)" = "none" ]; then
        say "pokertrainer: this image was built without TexasSolver (SOLVER_VARIANT=none)."
        say "  Solver study is disabled; every other feature is unaffected."
    else
        say "pokertrainer: TEXAS_SOLVER_PATH=${TEXAS_SOLVER_PATH} is not an executable file."
        say "  Solver study is disabled until it points at console_solver."
    fi
    unset TEXAS_SOLVER_PATH
fi

if [ "$#" -eq 0 ]; then
    set -- streamlit run "$APP_DIR/app.py" \
        --server.port="${PORT:-8501}" \
        --server.address=0.0.0.0 \
        --server.headless=true
fi

exec "$@"
