#!/bin/bash

mkdir runs

MODEL="llama"
TEST_NAME=$1

GAME_PREFIX="thunt_"

for i in {1..30}
do
    COMMAND="python run.py --game games/${GAME_PREFIX}${i}.z8 --model ${MODEL} --max-episodes 12"
    ${COMMAND} > "runs/${GAME_PREFIX}${i}_${MODEL}_ver_${LLAMA_VERSION}_${TEST_NAME}.txt"
done