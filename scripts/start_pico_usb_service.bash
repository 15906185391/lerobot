#!/usr/bin/env bash

set -Eeuo pipefail

PORT=63901
SERVICE_DIR="/opt/apps/roboticsservice"
LOG_FILE="${TMPDIR:-/tmp}/xrobotoolkit-pc-service.log"
ADB_SERIAL=""
START_SERVICE=1
REMOVE_REVERSE=0
WAIT_TIMEOUT_S=5

usage() {
    cat <<EOF
Usage: bash scripts/start_pico_usb_service.bash [options]

Starts XRoboToolkit PC Service on this computer and configures USB-C ADB
reverse port forwarding for a PICO headset.

Headset-side setting:
  In the XRoboToolkit app on PICO, enter 127.0.0.1 as the PC service IP.

Options:
  --port PORT           TCP port to reverse. Default: 63901.
  --serial SERIAL       ADB device serial to use when multiple devices exist.
  --service-dir DIR     XRoboToolkit PC Service directory.
                        Default: /opt/apps/roboticsservice.
  --log-file PATH       PC service log file. Default: /tmp/xrobotoolkit-pc-service.log.
  --no-service          Only configure adb reverse; do not start PC Service.
  --remove              Remove adb reverse for --port and exit.
  --wait-timeout SEC    Seconds to wait for local service port. Default: 5.
  -h, --help            Show this help.

Examples:
  bash scripts/start_pico_usb_service.bash
  bash scripts/start_pico_usb_service.bash --serial 0123456789ABCDEF
  bash scripts/start_pico_usb_service.bash --remove
EOF
}

log() {
    echo "[pico-usb-service] $*"
}

warn() {
    echo "[pico-usb-service][WARN] $*" >&2
}

die() {
    echo "[pico-usb-service][ERROR] $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            [[ $# -ge 2 ]] || die "--port requires a value"
            PORT="$2"
            shift 2
            ;;
        --serial)
            [[ $# -ge 2 ]] || die "--serial requires a value"
            ADB_SERIAL="$2"
            shift 2
            ;;
        --service-dir)
            [[ $# -ge 2 ]] || die "--service-dir requires a value"
            SERVICE_DIR="$2"
            shift 2
            ;;
        --log-file)
            [[ $# -ge 2 ]] || die "--log-file requires a value"
            LOG_FILE="$2"
            shift 2
            ;;
        --no-service)
            START_SERVICE=0
            shift
            ;;
        --remove)
            REMOVE_REVERSE=1
            shift
            ;;
        --wait-timeout)
            [[ $# -ge 2 ]] || die "--wait-timeout requires a value"
            WAIT_TIMEOUT_S="$2"
            shift 2
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

[[ "${PORT}" =~ ^[0-9]+$ ]] || die "--port must be an integer"
[[ "${WAIT_TIMEOUT_S}" =~ ^[0-9]+$ ]] || die "--wait-timeout must be an integer"

require_adb() {
    if ! command -v adb >/dev/null 2>&1; then
        die "adb was not found. Install it with: sudo apt-get update && sudo apt-get install -y android-tools-adb"
    fi
}

adb_cmd() {
    if [[ -n "${ADB_SERIAL}" ]]; then
        adb -s "${ADB_SERIAL}" "$@"
    else
        adb "$@"
    fi
}

print_adb_hints() {
    cat >&2 <<EOF

PICO USB checklist:
  1. Connect the PICO headset to this computer with a USB-C data cable.
  2. Enable developer mode / USB debugging on the headset.
  3. Accept the USB debugging authorization prompt inside the headset.
  4. Run 'adb devices' and make sure the device state is 'device'.
EOF
}

print_usb_summary() {
    if ! command -v lsusb >/dev/null 2>&1; then
        return
    fi

    echo "" >&2
    echo "Current USB devices that may be relevant:" >&2
    lsusb | awk 'BEGIN { found = 0 } /PICO|Oculus|Meta|Qualcomm|Android|ByteDance|HTC|Google|Pico|XR/ { found = 1; print "  " $0 } END { if (!found) print "  No obvious PICO/Android USB device found by lsusb." }' >&2
}

select_adb_device() {
    require_adb
    adb start-server >/dev/null

    if [[ -n "${ADB_SERIAL}" ]]; then
        local state
        state="$(adb -s "${ADB_SERIAL}" get-state 2>/dev/null || true)"
        [[ "${state}" == "device" ]] || die "ADB device '${ADB_SERIAL}' is not ready. Current state: ${state:-unknown}"
        log "Using ADB device: ${ADB_SERIAL}"
        return
    fi

    local devices=()
    local unauthorized=()
    mapfile -t devices < <(adb devices | awk 'NR > 1 && $2 == "device" { print $1 }')
    mapfile -t unauthorized < <(adb devices | awk 'NR > 1 && $2 == "unauthorized" { print $1 }')

    if [[ "${#unauthorized[@]}" -gt 0 ]]; then
        warn "Found unauthorized ADB device(s): ${unauthorized[*]}"
        print_adb_hints
        die "Authorize USB debugging in the PICO headset, then rerun this script."
    fi

    if [[ "${#devices[@]}" -eq 0 ]]; then
        adb devices >&2 || true
        print_adb_hints
        die "No ready ADB device found."
    fi

    if [[ "${#devices[@]}" -gt 1 ]]; then
        printf '%s\n' "${devices[@]}" >&2
        die "Multiple ADB devices found. Rerun with --serial SERIAL."
    fi

    ADB_SERIAL="${devices[0]}"
    log "Using ADB device: ${ADB_SERIAL}"
}

service_is_running() {
    pgrep -f "${SERVICE_DIR}/RoboticsServiceProcess" >/dev/null 2>&1 || pgrep -x RoboticsServiceProcess >/dev/null 2>&1
}

start_pc_service() {
    if [[ "${START_SERVICE}" -eq 0 ]]; then
        log "Skipping PC Service startup because --no-service was set."
        return
    fi

    if service_is_running; then
        log "XRoboToolkit PC Service already appears to be running."
        return
    fi

    [[ -d "${SERVICE_DIR}" ]] || die "XRoboToolkit PC Service directory not found: ${SERVICE_DIR}"
    [[ -x "${SERVICE_DIR}/RoboticsServiceProcess" ]] || die "Executable not found: ${SERVICE_DIR}/RoboticsServiceProcess"

    mkdir -p "$(dirname "${LOG_FILE}")"
    log "Starting XRoboToolkit PC Service from ${SERVICE_DIR}."
    log "Service log: ${LOG_FILE}"

    (
        cd "${SERVICE_DIR}"
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${SERVICE_DIR}:${SERVICE_DIR}/lib:${SERVICE_DIR}/SDK/x64"
        export QT_PLUGIN_PATH="${SERVICE_DIR}/plugins/:${QT_PLUGIN_PATH:-}"
        export QT_QML_PATH="${SERVICE_DIR}/qml/:${QT_QML_PATH:-}"
        nohup ./RoboticsServiceProcess >>"${LOG_FILE}" 2>&1 &
    )

    sleep 1
    if service_is_running; then
        log "XRoboToolkit PC Service started."
    else
        warn "XRoboToolkit PC Service did not stay running. Check: ${LOG_FILE}"
    fi
}

wait_for_local_port() {
    if ! command -v ss >/dev/null 2>&1; then
        return
    fi

    local elapsed=0
    while [[ "${elapsed}" -lt "${WAIT_TIMEOUT_S}" ]]; do
        if ss -ltn | awk -v port=":${PORT}" '$4 ~ port"$" { found = 1 } END { exit !found }'; then
            log "Local TCP port ${PORT} is listening."
            return
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    warn "Did not see a local listener on TCP ${PORT} within ${WAIT_TIMEOUT_S}s. The service may still be starting, or it may open the port lazily."
}

configure_reverse() {
    if [[ "${REMOVE_REVERSE}" -eq 1 ]]; then
        log "Removing adb reverse tcp:${PORT}."
        adb_cmd reverse --remove "tcp:${PORT}" || true
        adb_cmd reverse --list || true
        return
    fi

    log "Configuring adb reverse: PICO 127.0.0.1:${PORT} -> PC 127.0.0.1:${PORT}."
    adb_cmd reverse "tcp:${PORT}" "tcp:${PORT}"

    if adb_cmd reverse --list | grep -F "tcp:${PORT} tcp:${PORT}" >/dev/null; then
        log "ADB reverse is active: tcp:${PORT} tcp:${PORT}."
    else
        adb_cmd reverse --list >&2 || true
        die "Failed to verify adb reverse for tcp:${PORT}."
    fi
}

select_adb_device
if [[ "${REMOVE_REVERSE}" -eq 1 ]]; then
    configure_reverse
    exit 0
fi
start_pc_service
configure_reverse
wait_for_local_port

log "Done. In the PICO XRoboToolkit app, set the PC service IP to: 127.0.0.1"
