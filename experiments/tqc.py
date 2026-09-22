
from sbx import TQC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils.arg_handling import with_args

N_ENVS = 12 #cpu cores; run physics engine in all CPU cores but do gradient steps on GPU with jax.
#ENV_NAME = "FrankaPickAndPlaceSparse-v0"
LOG_DIR = "./exp_out/logs"


@with_args
def run(args):

    ENV_ID = f"panda_mujoco_gym:{args.task}"
    env = make_vec_env(ENV_ID, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(ENV_ID, n_envs=1)

    model = TQC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        learning_starts=args.learning_starts,
        policy_kwargs=dict(net_arch=args.features),
        verbose=1,
        gradient_steps=-1,
        train_freq=1,
        seed=args.seed
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=f"{LOG_DIR}/{args.task}/{args.seed}",
        log_path=f"{LOG_DIR}/{args.task}/{args.seed}",
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(total_timesteps=args.n_steps, callback=eval_callback)

    model.save(f"{LOG_DIR}/tqc_her_final/{args.task}/{args.seed}")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
if __name__ == "__main__":
    run()

