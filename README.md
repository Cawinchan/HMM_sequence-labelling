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
  |        ├── embeddings-l-model.bin   <- Large Binary FastText model
  |        ├── embeddings-s-model.bin   <- Small Binary FastText model
  |        └── p4_tests                 <- Folder for k, r test outputs for extended HMM
  │   ├── RU        <- Russian train and test set
  │   └── Eval      <- Evaluation Script
  │
  ├── src           <- Source code for use in this project.
  │   ├── emission.py             <- Part 1 solution
  │   ├── transition.py           <- Part 2a solution
  │   ├── viterbi.py              <- Part 2b solution
  │   ├── best_k_viterbi.py       <- Part 3 solution
  │   └── test_parameters.py       <- Test for best_k_viterbi
  │
  ├── main.py       <- main python file
  │
  ├── run_test.sh   <- Script to generate all outputs
  |
  └── results.txt   <- Log of outputs
```
# How to run 

## Generate outputs 
 
 ```bash
 $ python main.py
 ```

## Evaluate generated outputs
 
 ```bash
 $ chmod 777 run_test.sh 
 $ run_test.sh
 ```
 
 The output can be found in results.txt 
