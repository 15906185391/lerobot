#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN} Starting LCM type generation...${NC}"

rm -rf cpp python java
mkdir -p cpp python java

find src -name "*.lcm" | while read -r file; do
    lcm-gen -jxp "$file"
done

cp /usr/local/share/java/lcm.jar .
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