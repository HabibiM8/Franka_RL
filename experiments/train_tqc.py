import sys
import wandb

from sbx import TQC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback


N_ENVS = 12 #cpu cores; run physics engine in all CPU cores but do gradient steps on GPU with jax.
ENV_NAME = "FrankaPickAndPlaceSparse-v0"
ENV_ID = f"panda_mujoco_gym:{ENV_NAME}"
LOG_DIR = "./logs"

def run(argvs=sys.argv[1:]):

    # writer = wandb.init(
    #     project="FrankaRL",
    #     mode="online",
    #     settings=wandb.Settings(_disable_stats=True),
    # )

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
        learning_rate=1e-3,
        buffer_size=100_000,
        batch_size=2048,
        gamma=0.95,
        tau=0.05,
        learning_starts=10_000,
        policy_kwargs=dict(net_arch=[256,256,256]),
        verbose=1,
        gradient_steps=-1,
        train_freq=1,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=f"{LOG_DIR}/{ENV_ID}/",
        log_path=f"{LOG_DIR}/{ENV_ID}/",
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(total_timesteps=12_000, callback=eval_callback)

    model.save(f"{LOG_DIR}/tqc_her_final/")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
if __name__ == "__main__":
    run()


# add wandb
# build config management and seeding
