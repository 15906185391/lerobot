#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${CONDA_DEFAULT_ENV:-xr}"
PYTHON_VERSION="3.12"
CREATE_ENV=0
SKIP_EDITABLE=0
SKIP_PICO_SDK=0
SKIP_VERIFY=0

usage() {
    cat <<EOF
Usage: bash scripts/setup_wheeled_arm_pico_conda.bash [options]

Installs the Python packages needed for local editable LeRobot recording with
wheeled_arm + wheeled_arm_pico.

Options:
  --env NAME          Conda environment to use. Default: current env, or xr.
  --python VERSION    Python version used with --create-env. Default: 3.12.
  --create-env        Create the conda env if it does not already exist.
  --skip-editable     Do not run pip install -e ".[core_scripts]".
  --skip-pico-sdk     Do not clone/install XRoboToolkit PICO SDK.
  --skip-verify       Do not run import verification at the end.
  -h, --help          Show this help.

Examples:
  bash scripts/setup_wheeled_arm_pico_conda.bash --env xr
  bash scripts/setup_wheeled_arm_pico_conda.bash --create-env --env xr --python 3.12
EOF
}

log() {
    echo "[setup-wheeled-arm-pico] $*"
}

die() {
    echo "[setup-wheeled-arm-pico][ERROR] $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)
            [[ $# -ge 2 ]] || die "--env requires a value"
            ENV_NAME="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || die "--python requires a value"
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --create-env)
            CREATE_ENV=1
            shift
            ;;
        --skip-editable)
            SKIP_EDITABLE=1
            shift
            ;;
        --skip-pico-sdk)
            SKIP_PICO_SDK=1
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
    die "This installer is intended for Linux/Ubuntu conda environments."
fi

if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    if [[ "${ID:-}" != "ubuntu" ]]; then
        log "Warning: this script has only been tested on Ubuntu. Detected: ${PRETTY_NAME:-unknown}"
    elif [[ "${VERSION_ID:-}" != "22.04" && "${VERSION_ID:-}" != "24.04" ]]; then
        log "Warning: this script has only been tested on Ubuntu 22.04/24.04. Detected: ${VERSION_ID:-unknown}"
    fi
fi

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    . "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    . "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    die "Could not find conda.sh. Install Miniconda/Anaconda first."
fi

if [[ "${CREATE_ENV}" -eq 1 ]]; then
    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        log "Conda env '${ENV_NAME}' already exists; reusing it."
    else
        log "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}."
        conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
    fi
fi

conda activate "${ENV_NAME}" || die "Could not activate conda env '${ENV_NAME}'. Use --create-env or create it first."

log "Using Python: $(command -v python)"
python --version

log "Installing conda-forge runtime packages."
conda install -c conda-forge -y \
    libstdcxx-ng \
    pinocchio \
    hpp-fcl \
    qpsolvers \
    daqp \
    viser \
    yourdfpy

log "Upgrading pip."
python -m pip install --upgrade pip

if [[ "${SKIP_EDITABLE}" -eq 0 ]]; then
    log "Installing LeRobot editable package with core_scripts extra."
    python -m pip install -e "${REPO_ROOT}[core_scripts]"
fi

log "Installing pip-only runtime packages."
python -m pip install lcm

if [[ "${SKIP_PICO_SDK}" -eq 0 ]]; then
    DEP_DIR="${REPO_ROOT}/dependencies"
    SDK_DIR="${DEP_DIR}/XRoboToolkit-PC-Service-Pybind"
    mkdir -p "${DEP_DIR}"

    if [[ -d "${SDK_DIR}/.git" ]]; then
        log "Updating existing XRoboToolkit SDK checkout."
        git -C "${SDK_DIR}" pull --ff-only || log "Warning: git pull failed; continuing with existing checkout."
    else
        log "Cloning XRoboToolkit SDK checkout."
        git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git "${SDK_DIR}"
    fi

    log "Installing XRoboToolkit SDK."
    bash "${SDK_DIR}/setup_ubuntu.sh"
fi

if [[ "${SKIP_VERIFY}" -eq 0 ]]; then
    log "Verifying Python imports."
    python - <<'PY'
import importlib
import importlib.util
import sys

required = [
    "pinocchio",
    "qpsolvers",
    "daqp",
    "viser",
    "yourdfpy",
    "lcm",
    "rerun",
]

missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

if not any(importlib.util.find_spec(name) is not None for name in ("hppfcl", "coal")):
    missing.append("hppfcl/coal: no collision backend importable")

try:
    importlib.import_module("xrobotoolkit_sdk")
except Exception as exc:
    missing.append(f"xrobotoolkit_sdk: {exc}")

if missing:
    print("Missing imports after install:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    raise SystemExit(1)

print("All wheeled_arm_pico runtime imports are available.")
PY
fi

log "Done. Activate with: conda activate ${ENV_NAME}"
