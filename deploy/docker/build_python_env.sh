#!/bin/sh
# Install the application's Python dependencies into the active virtualenv.
#
# This runs in a build stage that has a C compiler and is never shipped. eval7
# is the reason the stage has to exist: it publishes no aarch64 wheel, so a
# linux/arm64 build compiles it from the sdist, and the runtime image
# deliberately carries no compiler.
#
# A compiler alone is not enough. eval7's sdist imports Cython at the top of
# setup.py but declares no build backend, so pip's isolated build environment --
# which holds only setuptools and wheel -- fails with "ModuleNotFoundError: No
# module named 'Cython'" before any compiler is reached. Installing Cython into
# this stage and disabling isolation for that single requirement is what makes
# the source build succeed. Isolation stays on for every other requirement, so a
# future dependency that needs its own build backend still gets one.
#
# Versions are read out of the requirements files rather than repeated here, so
# a pin only ever moves in one place.
set -eu

REQUIREMENTS=${1:-requirements.txt}
CV_REQUIREMENTS=${2:-requirements-cv.txt}
# Installed without its dependency closure because resolving it would replace
# the CPU torch wheels pinned in requirements-cv.txt with the CUDA build.
ULTRALYTICS_PIN=${ULTRALYTICS_PIN:-ultralytics==8.3.203}

spec_for() {
    name=$1
    file=$2
    spec=$(sed 's/#.*//' "$file" | grep -iE "^${name}([<>=!~ ]|\$)" | head -n 1 | tr -d '[:space:]')
    if [ -z "$spec" ]; then
        echo "build_python_env.sh: ${file} no longer pins ${name}." >&2
        echo "  The container build compiles ${name} from source and needs its version spec." >&2
        echo "  Restore the pin or update this script to match." >&2
        exit 1
    fi
    printf '%s\n' "$spec"
}

python -m pip install --upgrade pip setuptools wheel
python -m pip install "$(spec_for Cython "$REQUIREMENTS")"
python -m pip install --no-build-isolation "$(spec_for eval7 "$REQUIREMENTS")"
python -m pip install -r "$REQUIREMENTS" -r "$CV_REQUIREMENTS"
python -m pip install --no-deps "$ULTRALYTICS_PIN"

# Importing the extension here fails the build on the architecture that built
# it, instead of at the first equity calculation on a running host.
python -c "import eval7; eval7.Card('As')"
