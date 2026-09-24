
#TASK="FrankaPickAndPlaceSparse-v0"
TASK="FrankaPushDense-v0"

SHARED_ARGS="--features 512 512 512 --buffer_size 1_000_000 --batch_size 2048 \
    --gamma 0.95 --learning_rate 1e-3 --tau 0.05 --learning_starts 1_000 --n_steps 1_000_000 --task ${TASK} --n_envs 8"

PLATFORM="cluster/cluster"  # cluster/cluster lichtenberg/cluster local/local

launch_job/${PLATFORM}_tqc.sh --first_seed 1 --last_seed 1 --n_parallel_seeds 1 $SHARED_ARGS \
    --experiment_name "tqc_test_push${TASK}"

