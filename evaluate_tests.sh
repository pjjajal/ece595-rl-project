#!/bin/bash
MODEL=$1
TEST_NAME=$2
LLAMA_VERSION=""

# ANSI Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
COLOR_ESCAPE='\033[0m'

GAME_PREFIX="thunt_"

for i in {1..30}
do
    output_file="runs/${GAME_PREFIX}${i}_${MODEL}_ver_${LLAMA_VERSION}_${TEST_NAME}.txt"
    if grep -q "win: True" ${output_file}; then
        echo -e "Test ${i}: ${GREEN}WIN${COLOR_ESCAPE}"
    else
        echo -e "Test ${i}: ${RED}LOSE${COLOR_ESCAPE}"
    fi
done