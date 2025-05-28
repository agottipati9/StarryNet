#! /bin/bash

N_TRIALS=10

# Churn and Non Churn with Queue Model
echo "Running with Queue Model"
for exp in 2 4; do
    echo "Running experiment $exp"
    for i in $(seq 1 $N_TRIALS); do
        echo "Trial $i of $N_TRIALS"
        echo "Removing files in /opt/home_dir/StarryNet/starlink-*"
        sudo rm -rf /opt/home_dir/StarryNet/starlink-*
        sudo python run_experiments.py --exp $exp
    done
done

# Churn and Non Churn with Default Queue
echo "Running with Default Queue"
for exp in 2 4; do
    echo "Running experiment $exp"
    for i in $(seq 1 $N_TRIALS); do
        echo "Trial $i of $N_TRIALS"
        echo "Removing files in /opt/home_dir/StarryNet/starlink-*"
        sudo rm -rf /opt/home_dir/StarryNet/starlink-*
        sudo python run_experiments.py --exp $exp --use_default_queue
    done
done