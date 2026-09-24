#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@ --first_seed dummy --last_seed dummy
FIRST_SEED=$SLURM_ARRAY_TASK_ID
LAST_SEED=$((N_PARALLEL_SEEDS + SLURM_ARRAY_TASK_ID - 1))

source env/bin/activate
for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    python3 experiments/$ALGO_NAME.py --experiment_name $EXPERIMENT_NAME --seed $seed $ARGS &> experiments/exp_out/$EXPERIMENT_NAME/$TASK/logs/train_$seed.out &
done
wait