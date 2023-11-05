#!/bin/bash

mkdir games

for i in {1..30}
do
    echo "Generating treasure hunt game level: $i"
    tw-make tw-treasure_hunter --level "$i"  --output "./games/thunt_$i.z8" --seed 42 
done