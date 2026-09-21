
TASK="FrankaPickAndPlaceSparse-v0"

SHARED_ARGS="--features 256 256 256 --buffer_size 100_000 --batch_size 2048 \
    --gamma 0.95 --learning_rate 1e-3 --tau 0.05 --learning_starts 10_000 --n_steps 100_000 --task ${TASK}"

PLATFORM="cluster/cluster"  # cluster/cluster lichtenberg/cluster local/local

launch_job/${PLATFORM}_tqc.sh --first_seed 1 --last_seed 1 --n_parallel_seeds 1 $SHARED_ARGS \
    --experiment_name "tqc_test_${TASK}"

