#! /bin/bash

START_EXP=2
END_EXP=5
N_TRIALS=300

for exp in $(seq $START_EXP $END_EXP); do
    echo "Running experiment $exp"
    for i in $(seq 1 $N_TRIALS); do
        echo "Trial $i of $N_TRIALS"
        echo "Removing files in /opt/home_dir/StarryNet/starlink-*"
        sudo rm -rf /opt/home_dir/StarryNet/starlink-*
        sudo python run_experiments.py --exp $exp
    done
done

