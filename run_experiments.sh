#! /bin/bash

START_EXP=1
END_EXP=9
N_TRIALS=10

for exp in $(seq $START_EXP $END_EXP); do
    echo "Running experiment $exp"
    for i in $(seq 1 $N_TRIALS); do
        echo "Trial $i of $N_TRIALS"
        sudo python run_experiments.py --exp $exp
    done
done




