#!/bin/bash

mkdir runs

MODEL=$1
LLAMA_VERSION=$2

GAME_PREFIX="thunt_"

for i in {1..30}
do
    COMMAND="python run.py --game games/${GAME_PREFIX}${i}.z8 --model ${MODEL} --llama-version ${LLAMA_VERSION}"
    ${COMMAND} > "runs/${GAME_PREFIX}${i}_${MODEL}_ver_${LLAMA_VERSION}.txt"
done