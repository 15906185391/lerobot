#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN} Starting LCM type generation...${NC}"

find_lcm_java_jar() {
    if [ -n "${LCM_JAVA_JAR:-}" ] && [ -f "${LCM_JAVA_JAR}" ]; then
        printf '%s
' "${LCM_JAVA_JAR}"
        return 0
    fi

    local search_roots=()
    if [ -n "${CONDA_PREFIX:-}" ]; then
        search_roots+=("${CONDA_PREFIX}")
    fi
    search_roots+=("/usr/local" "/usr")

    local root candidate
    for root in "${search_roots[@]}"; do
        if [ -f "${root}/share/java/lcm.jar" ]; then
            printf '%s
' "${root}/share/java/lcm.jar"
            return 0
        fi
        candidate="$(find "${root}" -path '*/share/java/lcm.jar' -type f 2>/dev/null | head -n 1)"
        if [ -n "${candidate}" ]; then
            printf '%s
' "${candidate}"
            return 0
        fi
    done

    return 1
}

rm -rf cpp python java
mkdir -p cpp python java

find src -name "*.lcm" | while read -r file; do
    lcm-gen -jxp "$file"
done

LCM_JAVA_JAR="$(find_lcm_java_jar || true)"
if [ -z "${LCM_JAVA_JAR}" ]; then
    echo "Failed to locate lcm.jar. Set LCM_JAVA_JAR or install the LCM Java runtime."
    exit 1
fi

cp "${LCM_JAVA_JAR}" .
javac -cp lcm.jar $(find . -name "*.java" -not -path "./src/*")
jar cf java/my_types.jar $(find . -name "*.class" -not -path "./src/*")
mv lcm.jar java

find . -name "*.hpp" -not -path "./src/*" -not -path "./cpp/*" | while read -r file; do
    rel_path="${file#./}"
    mkdir -p "cpp/$(dirname "$rel_path")"
    mv "$file" "cpp/$rel_path"
done

find . -name "*.py" -not -path "./src/*" -not -path "./python/*" | while read -r file; do
    rel_path="${file#./}"
    mkdir -p "python/$(dirname "$rel_path")"
    mv "$file" "python/$rel_path"
done

find . -type d -empty -not -path "./src/*" -not -path "./cpp/*" -not -path "./python/*" -not -path "./java/*" -delete

echo -e "${GREEN} Done with LCM type generation${NC}"
