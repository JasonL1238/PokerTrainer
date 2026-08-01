#!/usr/bin/env bash
# Prove the container claims this repository makes, on a machine that has Docker.
#
# Every one of these is currently a written assertion with nothing behind it:
# that the image builds from a clean checkout, that it builds for linux/amd64
# and linux/arm64, that it runs unprivileged, that it becomes healthy, that its
# durable writes land on the mount and not in the image, that it survives a
# restart with its data, and that it works with no TexasSolver in it. This
# script turns each into a pass or a fail with a recorded number beside it.
#
# It is deliberately a shell script rather than a pytest case: it needs a Docker
# daemon, buildx and (for the foreign architecture) QEMU, none of which belong in
# the unit suite. Run it on any machine that has them, or from CI.
#
#   deploy/verify_container.sh                      # native architecture only
#   deploy/verify_container.sh --all-architectures  # amd64 and arm64
#   deploy/verify_container.sh --report out.json
#
# The clean-checkout build is verified against a `git archive` of HEAD in a
# temporary directory, not against the working tree, because a working tree
# holds untracked files and is exactly the thing that hid the previous defect.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_BASE="${POKERTRAINER_VERIFY_IMAGE:-pokertrainer-verify}"
REPORT_PATH=""
PLATFORMS=("")
FAILURES=0
RESULTS=()

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --all-architectures) PLATFORMS=("linux/amd64" "linux/arm64"); shift ;;
        --platform) PLATFORMS=("$2"); shift 2 ;;
        --report) REPORT_PATH="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

log()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
pass() { printf '  PASS  %s\n' "$*"; RESULTS+=("{\"check\":\"$1\",\"status\":\"pass\",\"detail\":\"${2:-}\"}"); }
fail() { printf '  FAIL  %s\n' "$*"; RESULTS+=("{\"check\":\"$1\",\"status\":\"fail\",\"detail\":\"${2:-}\"}"); FAILURES=$((FAILURES + 1)); }
info() { printf '        %s\n' "$*"; }

require_tooling() {
    log "tooling"
    if ! command -v docker >/dev/null 2>&1; then
        echo "docker is not installed; nothing in this script can run." >&2
        exit 3
    fi
    if ! docker version >/dev/null 2>&1; then
        echo "the Docker daemon is unreachable; start it and re-run." >&2
        exit 3
    fi
    if ! docker buildx version >/dev/null 2>&1; then
        echo "docker buildx is required to build a foreign architecture." >&2
        exit 3
    fi
    info "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null)"
    info "buildx $(docker buildx version | head -n1)"
}

# A tarball of HEAD, so the build sees exactly what someone cloning would see.
clean_context() {
    local dest="$1"
    git -C "$REPO_ROOT" archive --format=tar HEAD | tar -x -C "$dest"
}

verify_platform() {
    local platform="$1"
    local slug tag context build_log
    slug="$(printf '%s' "${platform:-native}" | tr '/' '-')"
    tag="${IMAGE_BASE}:${slug}"
    context="$(mktemp -d)"
    build_log="$(mktemp)"
    trap 'rm -rf "$context" "$build_log"' RETURN

    log "clean-checkout build (${platform:-native architecture})"
    clean_context "$context"
    if [ -e "$context/cv_lab/models/region_spine_v1.pt" ]; then
        fail "clean-context-${slug}" "HEAD tracks a model weight it is not supposed to"
        return
    fi

    local -a build_args=(buildx build --load --tag "$tag" --progress plain)
    [ -n "$platform" ] && build_args+=(--platform "$platform")
    if docker "${build_args[@]}" "$context" >"$build_log" 2>&1; then
        pass "build-${slug}" "built ${tag} from a git archive of HEAD"
    else
        fail "build-${slug}" "see $(cp "$build_log" "${TMPDIR:-/tmp}/pokertrainer-build-${slug}.log"; echo "${TMPDIR:-/tmp}/pokertrainer-build-${slug}.log")"
        return
    fi

    local size
    size="$(docker image inspect --format '{{.Size}}' "$tag")"
    info "image size: $((size / 1024 / 1024)) MiB"
    RESULTS+=("{\"check\":\"image-size-${slug}\",\"status\":\"info\",\"detail\":\"${size}\"}")

    local arch
    arch="$(docker image inspect --format '{{.Architecture}}' "$tag")"
    if [ -z "$platform" ] || [ "linux/$arch" = "$platform" ]; then
        pass "architecture-${slug}" "$arch"
    else
        fail "architecture-${slug}" "asked for $platform, image reports $arch"
    fi

    # eval7 is the dependency with no aarch64 wheel; importing it proves the
    # extension was compiled for this architecture and that the compiler-free
    # runtime stage still has a working module.
    if docker run --rm --platform "${platform:-linux/$arch}" "$tag" python -c "import eval7; print(eval7.Card('As'))" >/dev/null 2>&1; then
        pass "eval7-import-${slug}"
    else
        fail "eval7-import-${slug}" "the compiled extension does not import on $arch"
    fi

    if docker run --rm --platform "${platform:-linux/$arch}" "$tag" id -u | grep -qx 10001; then
        pass "non-root-${slug}" "uid 10001"
    else
        fail "non-root-${slug}" "the container did not run as uid 10001"
    fi

    if docker run --rm --platform "${platform:-linux/$arch}" "$tag" sh -c 'ffmpeg -version >/dev/null 2>&1' ; then
        pass "ffmpeg-${slug}"
    else
        fail "ffmpeg-${slug}" "ffmpeg is not usable in the image"
    fi

    # A compiler in the runtime image means the multi-stage split leaked.
    if docker run --rm --platform "${platform:-linux/$arch}" "$tag" sh -c 'command -v gcc cc g++ >/dev/null 2>&1'; then
        fail "no-compiler-${slug}" "the runtime image ships a compiler"
    else
        pass "no-compiler-${slug}"
    fi

    verify_runtime "$tag" "${platform:-linux/$arch}" "$slug"
    verify_solver_free "$platform" "$slug" "$context"
}

verify_runtime() {
    local tag="$1" platform="$2" slug="$3"
    local data name port
    data="$(mktemp -d)"
    chmod 777 "$data"
    name="pokertrainer-verify-${slug}-$$"
    port=18501

    log "runtime behaviour (${slug})"

    # --read-only is the mechanical proof that no runtime write depends on the
    # application layer. If the app needs to write into /app it fails here.
    if ! docker run -d --name "$name" --platform "$platform" \
        --read-only --tmpfs /tmp \
        -e APP_PASSWORD=verification-only-not-a-secret \
        -e POKERTRAINER_REQUIRE_DATA_MOUNT=true \
        -e PORT=8501 \
        -v "$data:/data" -p "127.0.0.1:${port}:8501" "$tag" >/dev/null; then
        fail "start-readonly-${slug}" "container would not start with a read-only root filesystem"
        return
    fi

    local status="" waited=0
    while [ "$waited" -lt 180 ]; do
        status="$(docker inspect --format '{{.State.Health.Status}}' "$name" 2>/dev/null)"
        [ "$status" = "healthy" ] && break
        [ "$status" = "unhealthy" ] && break
        sleep 3
        waited=$((waited + 3))
    done
    if [ "$status" = "healthy" ]; then
        pass "healthcheck-${slug}" "healthy after ${waited}s"
    else
        fail "healthcheck-${slug}" "status=${status:-none} after ${waited}s"
        docker logs "$name" 2>&1 | tail -n 40
    fi

    if [ -f "$data/poker_tracker.db" ]; then
        pass "durable-write-on-mount-${slug}" "database created under the bind mount"
    else
        fail "durable-write-on-mount-${slug}" "no database appeared on the mount"
    fi

    # Restart drill: the same data must still be there and the container must
    # come back healthy on its own.
    local before after
    before="$(stat -c %s "$data/poker_tracker.db" 2>/dev/null || stat -f %z "$data/poker_tracker.db" 2>/dev/null || echo 0)"
    docker restart "$name" >/dev/null
    status=""; waited=0
    while [ "$waited" -lt 180 ]; do
        status="$(docker inspect --format '{{.State.Health.Status}}' "$name" 2>/dev/null)"
        [ "$status" = "healthy" ] && break
        sleep 3
        waited=$((waited + 3))
    done
    after="$(stat -c %s "$data/poker_tracker.db" 2>/dev/null || stat -f %z "$data/poker_tracker.db" 2>/dev/null || echo 0)"
    if [ "$status" = "healthy" ] && [ "$after" -ge "$before" ] && [ "$after" -gt 0 ]; then
        pass "restart-drill-${slug}" "healthy again, database retained (${after} bytes)"
    else
        fail "restart-drill-${slug}" "status=${status:-none}, database ${before} -> ${after} bytes"
    fi

    # Peak memory of the running container, which nothing in this repository has
    # ever recorded.
    local mem
    mem="$(docker stats --no-stream --format '{{.MemUsage}}' "$name" 2>/dev/null)"
    info "container memory: ${mem:-unavailable}"
    RESULTS+=("{\"check\":\"container-memory-${slug}\",\"status\":\"info\",\"detail\":\"${mem:-unavailable}\"}")

    docker rm -f "$name" >/dev/null 2>&1
    rm -rf "$data"
}

verify_solver_free() {
    local platform="$1" slug="$2" context="$3"
    log "solver-free image (${slug})"
    local tag="${IMAGE_BASE}:${slug}-nosolver"
    local -a build_args=(buildx build --load --build-arg SOLVER_VARIANT=none --tag "$tag" --progress plain)
    [ -n "$platform" ] && build_args+=(--platform "$platform")
    if ! docker "${build_args[@]}" "$context" >/dev/null 2>&1; then
        fail "solver-free-build-${slug}" "SOLVER_VARIANT=none does not build"
        return
    fi
    if docker run --rm "$tag" sh -c 'test ! -e /opt/texassolver/console_solver'; then
        pass "solver-free-build-${slug}" "no solver binary in the image"
    else
        fail "solver-free-build-${slug}" "SOLVER_VARIANT=none still shipped a solver binary"
    fi
    # The application has to remain usable, which means importing and rendering
    # without TEXAS_SOLVER_PATH rather than raising out of the Study page.
    if docker run --rm "$tag" python -c "
import poker_tracker.solver.texassolver as t
try:
    t.configured_binary()
except FileNotFoundError as exc:
    print(exc)
else:
    raise SystemExit('configured_binary() succeeded in an image with no solver')
"; then
        pass "solver-absent-degrades-${slug}" "reports an actionable configuration error"
    else
        fail "solver-absent-degrades-${slug}" "solver absence is not reported as a configuration error"
    fi
}

require_tooling
for platform in "${PLATFORMS[@]}"; do
    verify_platform "$platform"
done

log "summary"
printf '  %d check(s) failed\n' "$FAILURES"
if [ -n "$REPORT_PATH" ]; then
    {
        printf '{\n  "repo": "%s",\n  "failures": %d,\n  "checks": [\n    ' "$REPO_ROOT" "$FAILURES"
        printf '%s' "$(IFS=$'\n'; echo "${RESULTS[*]}" | paste -sd, - )"
        printf '\n  ]\n}\n'
    } >"$REPORT_PATH"
    info "report written to $REPORT_PATH"
fi
exit $(( FAILURES > 0 ? 1 : 0 ))
