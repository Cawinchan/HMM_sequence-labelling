
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" >> results.txt

echo "~~~~~~~~~~~~~~ Part 1: ES Results" >> results.txt

python data/Eval/evalResult.py data/ES/dev.out data/ES/dev.p1.out >> results.txt

echo "~~~~~~~~~~~~~~ Part 1: RU Results" >> results.txt

python data/Eval/evalResult.py data/RU/dev.out data/RU/dev.p1.out >> results.txt

echo "~~~~~~~~~~~~~~ Part 2: ES Results" >> results.txt

python data/Eval/evalResult.py data/ES/dev.out data/ES/dev.p2.out >> results.txt

echo "~~~~~~~~~~~~~~ Part 2: RU Results" >> results.txt

python data/Eval/evalResult.py data/RU/dev.out data/RU/dev.p2.out >> results.txt

for i in 1 2 3 4 5
do
	echo "~~~~~~~~~~~~~~ Part 3: $i th best of ES Results" >> results.txt

    python data/Eval/evalResult.py data/ES/dev.out data/ES/dev.p3.out $i>> results.txt

    echo "~~~~~~~~~~~~~~ Part 3: $i th best of RU Results" >> results.txt

    python data/Eval/evalResult.py data/RU/dev.out data/RU/dev.p3.out $i>> results.txt
done

echo "~~~~~~~~~~~~~~ Part 4: ES Results" >> results.txt

python data/Eval/evalResult.py data/ES/dev.out data/ES/dev.p4.out >> results.txt

echo "~~~~~~~~~~~~~~ Part 4: RU Results" >> results.txt

python data/Eval/evalResult.py data/RU/dev.out data/RU/dev.p4.out >> results.txt

echo "~~~~~~~~~~~~~~ Test: ES Results" >> results.txt

python data/Eval/evalResult.py data/ES/dev.out data/ES/test.p4.out >> results.txt

echo "~~~~~~~~~~~~~~ Test: RU Results" >> results.txt

python data/Eval/evalResult.py data/RU/dev.out data/RU/test.p4.out >> results.txt
