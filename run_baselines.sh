#! /bin/bash

N_TRIALS=10

# Churn and Non Churn with Satellite Context
echo "Running with Satellite Context"
for exp in 2 4; do
    echo "Running experiment $exp"
    for i in $(seq 1 $N_TRIALS); do
        echo "Trial $i of $N_TRIALS"
        echo "Removing files in /opt/home_dir/StarryNet/starlink-*"
        sudo rm -rf /opt/home_dir/StarryNet/starlink-*
        sudo python run_experiments.py --exp $exp --satellite_context
    done
done

# Churn and Non Churn without Satellite Context
echo "Running without Satellite Context"
for exp in 2 4; do
    echo "Running experiment $exp"
    for i in $(seq 1 $N_TRIALS); do
        echo "Trial $i of $N_TRIALS"
        echo "Removing files in /opt/home_dir/StarryNet/starlink-*"
        sudo rm -rf /opt/home_dir/StarryNet/starlink-*
        sudo python run_experiments.py --exp $exp
    done
done