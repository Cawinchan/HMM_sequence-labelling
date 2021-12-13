# HMM_sequence-labelling

# File Architecture 
   
```
  ├── data
  │   ├── ES        <- Spanish train and test set
  |        ├── dev.in                   <- Test data
  |        ├── dev.out                  <- Gold Standard
  |        ├── dev.p1.out               <- Generated
  |        ├── dev.p2.out               <- Generated
  |        ├── dev.p3.out               <- Generated
  |        ├── dev.p4.out               <- Generated
  |        ├── test.in                  <- Final Test data
  |        ├── train                    <- Train data
  |        └── p4_tests                 <- Folder for k, r test outputs for extended HMM

  │   ├── RU        <- Russian train and test set
  │   └── Eval      <- Evaluation Script
  │
  ├── src           <- Source code for use in this project.
  │   ├── emission.py             <- Part 1 solution
  │   ├── transition.py           <- Part 2a solution
  │   ├── viterbi.py              <- Part 2b solution
  │   ├── best_k_viterbi.py       <- Part 3 solution
  │   ├── part_3.py               <- Part 3 solution
  │   ├── part_4.py               <- Part 4 solution
  │   └── test_paramters.py       <- Test for best_k_viterbi
  │
  ├── main.py       <- main python file
  │
  ├── run_test.sh   <- Script to generate all outputs
  |
  └── results.txt   <- Log of outputs
```
# How to run 

## Generate outputs for Part 1-3
 
 ```bash
 $ python main.py
 ```

 To generate part_4, run project_part_4.ipynb.

## Evaluate generated outputs
 
 ```bash
 $ chmod 777 run_test.sh 
 $ run_test.sh
 ```
 
 The output can be found in results.txt 
