#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${CONDA_DEFAULT_ENV:-xr}"
PYTHON_VERSION=""
PYTHON_VERSION_SET=0
CREATE_ENV=0
SKIP_EDITABLE=0
SKIP_PICO_SDK=0
SKIP_SYSTEM_PACKAGES=0
SKIP_VERIFY=0
WITH_CONVERSION_DEPS=0

usage() {
    cat <<EOF
Usage: bash scripts/setup_wheeled_arm_pico_miniforge3.bash [options]

Installs the Python packages needed for local editable LeRobot recording with
wheeled_arm + wheeled_arm_pico in a Miniforge3 conda environment.

Options:
  --env NAME          Conda environment to use. Default: current env, or xr.
  --python VERSION    Python version used with --create-env.
                       Default: Ubuntu 22.04 -> 3.10, Ubuntu 24.04 -> 3.12.
  --create-env        Create the conda env if it does not already exist.
  --skip-editable     Do not run pip install -e ".[core_scripts,gui]".
  --skip-pico-sdk     Do not clone/install XRoboToolkit PICO SDK.
  --skip-system-packages
                       Do not install Ubuntu Qt/xcb packages needed by PySide6.
  --skip-verify       Do not run import verification at the end.
  --with-conversion-deps
                       Also install optional Any4LeRobot conversion dependencies
                       such as tensorflow-datasets, h5py, ray, datatrove and apache-beam.
  -h, --help          Show this help.

Examples:
  bash scripts/setup_wheeled_arm_pico_miniforge3.bash --env xr
  bash scripts/setup_wheeled_arm_pico_miniforge3.bash --create-env --env xr
  bash scripts/setup_wheeled_arm_pico_miniforge3.bash --create-env --env xr --python 3.12
EOF
}

log() {
    echo "[setup-wheeled-arm-pico-miniforge3] $*"
}

die() {
    echo "[setup-wheeled-arm-pico-miniforge3][ERROR] $*" >&2
    exit 1
}

python_module_available() {
    local module="$1"
    python - "${module}" <<'PY' >/dev/null 2>&1
import importlib.util
import sys

raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)
PY
}

python_any_module_available() {
    local module
    for module in "$@"; do
        if python_module_available "${module}"; then
            return 0
        fi
    done
    return 1
}

conda_package_installed() {
    local package="$1"
    conda list "${package}" 2>/dev/null | awk -v package="${package}" '$1 == package { found = 1 } END { exit !found }'
}

conda_spec_satisfied() {
    local spec="$1"
    local package="${spec%%:*}"
    local modules=""

    if [[ "${spec}" == *":"* ]]; then
        modules="${spec#*:}"
    fi

    if conda_package_installed "${package}"; then
        return 0
    fi

    if [[ -n "${modules}" ]]; then
        IFS=',' read -r -a module_names <<< "${modules}"
        if python_any_module_available "${module_names[@]}"; then
            return 0
        fi
    fi

    return 1
}

install_missing_conda_packages() {
    local specs=("$@")
    local missing=()
    local spec
    local package

    for spec in "${specs[@]}"; do
        package="${spec%%:*}"
        if conda_spec_satisfied "${spec}"; then
            log "Conda package/library '${package}' is already available; skipping."
        else
            missing+=("${package}")
        fi
    done

    if [[ "${#missing[@]}" -eq 0 ]]; then
        log "All conda-forge runtime packages are already available."
        return
    fi

    log "Installing missing conda-forge runtime packages: ${missing[*]}"
    conda install --override-channels -c conda-forge -y "${missing[@]}"
}

pip_spec_satisfied() {
    local spec="$1"
    local modules="${spec#*:}"

    IFS=',' read -r -a module_names <<< "${modules}"
    python_any_module_available "${module_names[@]}"
}

install_missing_pip_specs() {
    local specs=("$@")
    local missing=()
    local spec
    local package

    for spec in "${specs[@]}"; do
        package="${spec%%:*}"
        if pip_spec_satisfied "${spec}"; then
            log "Python library for '${package}' is already available; skipping."
        else
            missing+=("${package}")
        fi
    done

    if [[ "${#missing[@]}" -eq 0 ]]; then
        log "All requested pip-only packages are already available."
        return
    fi

    log "Installing missing pip-only packages: ${missing[*]}"
    python -m pip install "${missing[@]}"
}

lerobot_editable_with_extras_available() {
    python - "${REPO_ROOT}" <<'PY' >/dev/null 2>&1
import importlib.util
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
spec = importlib.util.find_spec("lerobot")
if spec is None or spec.submodule_search_locations is None:
    raise SystemExit(1)

locations = [pathlib.Path(path).resolve() for path in spec.submodule_search_locations]
if not any(str(path).startswith(str(repo_root)) for path in locations):
    raise SystemExit(1)

for module in ("datasets", "rerun", "PySide6"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(1)
PY
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
            PYTHON_VERSION_SET=1
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
        --skip-system-packages)
            SKIP_SYSTEM_PACKAGES=1
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=1
            shift
            ;;
        --with-conversion-deps)
            WITH_CONVERSION_DEPS=1
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
    die "This installer is intended for Linux/Ubuntu Miniforge3 environments."
fi

UBUNTU_VERSION_ID=""
if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    if [[ "${ID:-}" != "ubuntu" ]]; then
        log "Warning: this script has only been tested on Ubuntu. Detected: ${PRETTY_NAME:-unknown}"
    elif [[ "${VERSION_ID:-}" != "22.04" && "${VERSION_ID:-}" != "24.04" ]]; then
        log "Warning: this script has only been tested on Ubuntu 22.04/24.04. Detected: ${VERSION_ID:-unknown}"
    else
        UBUNTU_VERSION_ID="${VERSION_ID}"
    fi
fi

if [[ "${PYTHON_VERSION_SET}" -eq 0 ]]; then
    case "${UBUNTU_VERSION_ID}" in
        24.04)
            PYTHON_VERSION="3.12"
            ;;
        22.04|*)
            PYTHON_VERSION="3.10"
            ;;
    esac
fi

install_qt_system_packages() {
    if [[ "${SKIP_SYSTEM_PACKAGES}" -eq 1 ]]; then
        return
    fi
    if ! command -v apt-get >/dev/null 2>&1 || ! command -v dpkg-query >/dev/null 2>&1; then
        log "Warning: apt/dpkg not found; skipping Qt/xcb system package installation."
        return
    fi

    local packages=(
        libxcb-cursor0
        libxcb-icccm4
        libxcb-image0
        libxcb-keysyms1
        libxcb-render-util0
        libxcb-xinerama0
        libxcb-xkb1
        libxkbcommon-x11-0
        libegl1
        libgl1
    )
    local missing=()
    local pkg
    for pkg in "${packages[@]}"; do
        if ! dpkg-query -W -f='${Status}' "${pkg}" 2>/dev/null | grep -q "install ok installed"; then
            missing+=("${pkg}")
        fi
    done

    if [[ "${#missing[@]}" -eq 0 ]]; then
        log "Qt/xcb system packages are already installed."
        return
    fi

    log "Installing Qt/xcb system packages for PySide6: ${missing[*]}"
    sudo apt-get update
    sudo apt-get install -y "${missing[@]}"
}

install_qt_system_packages

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    . "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -n "${CONDA_EXE:-}" ]]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base)"
    # shellcheck disable=SC1091
    . "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    die "Could not find Miniforge3 conda.sh. Install Miniforge3 first, or run this script from an initialized Miniforge3 shell."
fi

conda_activate_env() {
    local env_name="$1"
    local nounset_was_enabled=0

    case "$-" in
        *u*)
            nounset_was_enabled=1
            set +u
            ;;
    esac

    conda activate "${env_name}"
    local status=$?

    if [[ "${nounset_was_enabled}" -eq 1 ]]; then
        set -u
    fi

    return "${status}"
}

if [[ "${CREATE_ENV}" -eq 1 ]]; then
    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        log "Conda env '${ENV_NAME}' already exists; reusing it."
    else
        log "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION} from conda-forge."
        conda create --override-channels -c conda-forge -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
    fi
fi

conda_activate_env "${ENV_NAME}" || die "Could not activate conda env ${ENV_NAME}. Use --create-env or create it first."

log "Using Python: $(command -v python)"
python --version

install_missing_conda_packages \
    "libstdcxx-ng:" \
    "ffmpeg:ffmpeg" \
    "pinocchio:pinocchio" \
    "hpp-fcl:hppfcl,coal" \
    "qpsolvers:qpsolvers" \
    "daqp:daqp" \
    "viser:viser" \
    "yourdfpy:yourdfpy" \
    "xcb-util-cursor:"

if python -m pip --version >/dev/null 2>&1; then
    log "pip is already available; skipping pip bootstrap."
else
    log "Bootstrapping pip."
    python -m ensurepip --upgrade
fi

if [[ "${SKIP_EDITABLE}" -eq 0 ]]; then
    if lerobot_editable_with_extras_available; then
        log "LeRobot editable package with core_scripts/gui imports is already available; skipping."
    else
        log "Installing LeRobot editable package with core_scripts and gui extras."
        python -m pip install -e "${REPO_ROOT}[core_scripts,gui]"
    fi
fi

install_missing_pip_specs "lcm:lcm"

if [[ "${WITH_CONVERSION_DEPS}" -eq 1 ]]; then
    install_missing_pip_specs \
        "tensorflow:tensorflow" \
        "tensorflow-datasets:tensorflow_datasets" \
        "h5py:h5py" \
        "ray[default]:ray" \
        "datatrove[ray]:datatrove" \
        "apache-beam:apache_beam"
else
    log "Skipping optional data conversion packages. Use --with-conversion-deps when needed."
fi

if [[ "${SKIP_PICO_SDK}" -eq 0 ]]; then
    if python_module_available "xrobotoolkit_sdk"; then
        log "XRoboToolkit SDK Python module is already available; skipping SDK clone/install."
    else
        DEP_DIR="${REPO_ROOT}/dependencies"
        SDK_DIR="${DEP_DIR}/XRoboToolkit-PC-Service-Pybind"
        mkdir -p "${DEP_DIR}"

        if [[ -d "${SDK_DIR}/.git" ]]; then
            log "XRoboToolkit SDK checkout already exists; reusing it."
        else
            log "Cloning XRoboToolkit SDK checkout."
            git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git "${SDK_DIR}"
        fi

        log "Installing XRoboToolkit SDK."
        (
            cd "${SDK_DIR}"
            bash setup_ubuntu.sh
        )
        if ! python_module_available "xrobotoolkit_sdk"; then
            die "XRoboToolkit SDK install completed, but 'xrobotoolkit_sdk' is still not importable in conda env '${ENV_NAME}'. Check the SDK build output above."
        fi
    fi
fi

if [[ "${SKIP_VERIFY}" -eq 0 ]]; then
    log "Verifying Python imports."
    python - <<'PY'
import importlib
import importlib.util
import sys
from ctypes.util import find_library

required = [
    "pinocchio",
    "qpsolvers",
    "daqp",
    "viser",
    "yourdfpy",
    "lcm",
    "rerun",
    "PySide6",
]

missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")

if not any(importlib.util.find_spec(name) is not None for name in ("hppfcl", "coal")):
    missing.append("hppfcl/coal: no collision backend importable")

if not find_library("xcb-cursor"):
    missing.append(
        "libxcb-cursor0: Qt xcb runtime library not found. "
        "Install Ubuntu Qt/xcb packages or rerun without --skip-system-packages."
    )

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
print("Launch the GUI with: lerobot-wheeled-arm-gui")
PY
fi

log "Done. Activate with: conda activate ${ENV_NAME}"
