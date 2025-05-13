#! /bin/bash

START_EXP=2
END_EXP=5
N_TRIALS=500

queue_sizes=(100 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000)

for exp in $(seq $START_EXP $END_EXP); do
    echo "Running experiment $exp"
    for i in $(seq 1 $N_TRIALS); do
        echo "Trial $i of $N_TRIALS"
        echo "Removing files in /opt/home_dir/StarryNet/starlink-*"
        sudo rm -rf /opt/home_dir/StarryNet/starlink-*
        QUEUE_SIZE=${queue_sizes[$((RANDOM % ${#queue_sizes[@]}))]}
        echo "Setting queue size to $QUEUE_SIZE"
        sudo python adjust_alphartc_queue.py --queue_size $QUEUE_SIZE
        sudo python run_experiments.py --exp $exp
        cp /opt/home_dir/StarryNet/paced_sender.cc /opt/home_dir/AlphaRTC/modules/pacing/paced_sender.cc  # restore original paced_sender.cc
    done
done
