#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

echo "launch train $ALGO_NAME"

sbatch --job-name $EXPERIMENT_NAME-$ALGO_NAME --array=$FIRST_SEED-$LAST_SEED:$N_PARALLEL_SEEDS --gres=gpu:1 --cpus-per-task=$((N_PARALLEL_SEEDS * (N_ENVS + 2))) --mem-per-cpu=1200M --time=08:00:00 --partition stud \
--output=experiments/exp_out/$EXPERIMENT_NAME/$TASK/slurm/train_$FIRST_SEED-$LAST_SEED.out \
launch_job/train.sh --algo_name $ALGO_NAME --env_name $ENV_NAME --experiment_name $EXPERIMENT_NAME $ARGS --n_parallel_seeds $N_PARALLEL_SEEDS

