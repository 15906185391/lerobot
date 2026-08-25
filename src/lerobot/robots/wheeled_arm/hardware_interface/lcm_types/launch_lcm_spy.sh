#!/usr/bin/env bash
set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
JAVA_DIR="${DIR}/java"
DEFAULT_PORT=8880
DEFAULT_TTL=255

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [port]
  $(basename "$0") --lcm-url URL

Examples:
  $(basename "$0")
  $(basename "$0") 8880
  $(basename "$0") --lcm-url 'udpm://239.255.76.67:8880?ttl=255'
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "${JAVA_DIR}" || ! -f "${JAVA_DIR}/my_types.jar" || ! -f "${JAVA_DIR}/lcm.jar" ]]; then
  echo "Missing LCM Java types under ${JAVA_DIR}." >&2
  echo "Run ${DIR}/make_types.sh first, then launch lcm-spy again." >&2
  exit 1
fi

if [[ "${1:-}" == "--lcm-url" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Missing value after --lcm-url." >&2
    usage >&2
    exit 2
  fi
  lcm_url="$2"
elif [[ "${1:-}" == *"://"* ]]; then
  lcm_url="$1"
else
  port="${1:-${DEFAULT_PORT}}"
  lcm_url="udpm://239.255.76.67:${port}?ttl=${DEFAULT_TTL}"
fi

cd "${JAVA_DIR}"
export CLASSPATH="${JAVA_DIR}/my_types.jar:${JAVA_DIR}/lcm.jar"
exec lcm-spy --lcm-url="${lcm_url}"
