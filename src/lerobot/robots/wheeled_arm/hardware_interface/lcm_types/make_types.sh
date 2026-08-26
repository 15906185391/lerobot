#!/usr/bin/env bash
set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
JAVA_DIR="${DIR}/java"
CLASSES_LIST="${JAVA_DIR}/classes.list"
PLUGIN_CLASSES_DIR="${JAVA_DIR}/plugin_classes"
PLUGIN_SOURCES_LIST="${JAVA_DIR}/plugin_sources.list"

find_lcm_jar() {
  if [[ -n "${LCM_JAR:-}" && -f "${LCM_JAR}" ]]; then
    printf '%s\n' "${LCM_JAR}"
    return 0
  fi

  local candidate
  for candidate in \
    "/usr/share/java/lcm.jar" \
    "/usr/local/share/java/lcm.jar" \
    "${CONDA_PREFIX:-}/share/java/lcm.jar" \
    "${CONDA_PREFIX:-}/lib/python3.12/site-packages/share/java/lcm.jar"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

mkdir -p "${JAVA_DIR}"

lcm_jar="$(find_lcm_jar)" || {
  echo "Unable to find lcm.jar. Set LCM_JAR=/path/to/lcm.jar and retry." >&2
  exit 1
}

if ! command -v jar >/dev/null 2>&1; then
  echo "Unable to find the Java jar command." >&2
  exit 1
fi

cp "${lcm_jar}" "${JAVA_DIR}/lcm.jar"
rm -rf "${PLUGIN_CLASSES_DIR}"
mkdir -p "${PLUGIN_CLASSES_DIR}"
find "${DIR}/plugins" -name '*.java' -print > "${PLUGIN_SOURCES_LIST}"
if [[ -s "${PLUGIN_SOURCES_LIST}" ]]; then
  if ! command -v javac >/dev/null 2>&1; then
    echo "Unable to find javac for compiling LCM Spy plugins." >&2
    exit 1
  fi
  javac -encoding UTF-8 -classpath "${JAVA_DIR}/lcm.jar:${DIR}" \
    -d "${PLUGIN_CLASSES_DIR}" @"${PLUGIN_SOURCES_LIST}"
fi

find "${DIR}/hal" -name '*.class' -printf '%P\n' \
  | sed -e 's#^#hal/#' > "${CLASSES_LIST}.hal"
find "${DIR}/manipulation" -name '*.class' -printf '%P\n' \
  | sed -e 's#^#manipulation/#' > "${CLASSES_LIST}.manipulation"
find "${PLUGIN_CLASSES_DIR}" -name '*.class' -printf '%P\n' > "${CLASSES_LIST}.plugins"
cat "${CLASSES_LIST}.hal" "${CLASSES_LIST}.manipulation" > "${CLASSES_LIST}"
rm -f "${CLASSES_LIST}.hal" "${CLASSES_LIST}.manipulation"

if [[ ! -s "${CLASSES_LIST}" ]]; then
  echo "No generated .class files found under ${DIR}/hal or ${DIR}/manipulation." >&2
  exit 1
fi

(
  cd "${DIR}"
  jar --create --file "${JAVA_DIR}/my_types.jar" @"${CLASSES_LIST}"
)

if [[ -s "${CLASSES_LIST}.plugins" ]]; then
  (
    cd "${PLUGIN_CLASSES_DIR}"
    jar --update --file "${JAVA_DIR}/my_types.jar" @"${CLASSES_LIST}.plugins"
  )
fi

rm -f "${CLASSES_LIST}" "${CLASSES_LIST}.plugins" "${PLUGIN_SOURCES_LIST}"
rm -rf "${PLUGIN_CLASSES_DIR}"

echo "Generated ${JAVA_DIR}/my_types.jar"
echo "Copied ${JAVA_DIR}/lcm.jar"
