#!/bin/bash

echo "~~~~~~~~~~~~~~ Part 1: ES Results" 

python data/Eval/evalResult.py data/ES/dev.out data/ES/dev.p1.out

echo "~~~~~~~~~~~~~~ Part 1: RU Results" 

python data/Eval/evalResult.py data/RU/dev.out data/RU/dev.p1.out

echo "~~~~~~~~~~~~~~ Part 2: ES Results" 

python data/Eval/evalResult.py data/ES/dev.out data/ES/dev.p2.out

echo "~~~~~~~~~~~~~~ Part 2: RU Results" 

python data/Eval/evalResult.py data/RU/dev.out data/RU/dev.p2.out

$SHELL