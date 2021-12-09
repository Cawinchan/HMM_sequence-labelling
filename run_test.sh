
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" >> results.txt

echo "~~~~~~~~~~~~~~ Part 1: ES Results" >> results.txt

python data/Eval/evalResult.py data/ES/dev.out data/ES/dev.p1.out >> results.txt

echo "~~~~~~~~~~~~~~ Part 1: RU Results" >> results.txt

python data/Eval/evalResult.py data/RU/dev.out data/RU/dev.p1.out >> results.txt

echo "~~~~~~~~~~~~~~ Part 2: ES Results" >> results.txt

python data/Eval/evalResult.py data/ES/dev.out data/ES/dev.p2.out >> results.txt

echo "~~~~~~~~~~~~~~ Part 2: RU Results" >> results.txt

python data/Eval/evalResult.py data/RU/dev.out data/RU/dev.p2.out >> results.txt
    