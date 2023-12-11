#!/bin/bash
mkdir runs

MODEL=$1
TEST_NAME=$2
LLAMA_VERSION=""

echo "Evaluating model ${MODEL} on all games"

GAME_PREFIX="thunt_"

for i in {1..30}
do
    COMMAND="python run.py --game games/${GAME_PREFIX}${i}.z8 --model ${MODEL} --max-episodes 12"
    ${COMMAND} > "runs/${GAME_PREFIX}${i}_${MODEL}_ver_${LLAMA_VERSION}_${TEST_NAME}.txt"
done