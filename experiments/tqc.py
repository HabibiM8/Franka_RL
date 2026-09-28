import os
import jax

from sbx import TQC
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils.arg_handling import with_args, prep_exp
from utils.domain_randomization import DRConfig, DomainRandomization

#N_ENVS = 12 #cpu cores; run physics engine in all CPU cores but do gradient steps on GPU with jax.


@with_args
def run(args):

    # ... DRConfig(max_delay=args.max_delay) ...
    domain_randomizer = dict(wrapper_class=DomainRandomization,
                             wrapper_kwargs=dict(cfg=DRConfig())) if args.domain_randomization else {}

    env_id = f"panda_mujoco_gym:{args.task}"
    env = make_vec_env(env_id, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv, **domain_randomizer)
    eval_env = make_vec_env(env_id, n_envs=1)

    log_dir, model_dir = prep_exp(args)

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
        seed=args.seed,
        tensorboard_log=log_dir,
    )
    print(f"CPU cores available: {len(os.sched_getaffinity(0))} | n_envs: {args.n_envs} | jax backend: {jax.default_backend()}")
    print("args:", vars(args))
    print("policy_kwargs:", model.policy_kwargs, "| batch_size:", model.batch_size, "| gamma:", model.gamma,
          "| tau:", model.tau, "| learning_starts:", model.learning_starts, "| gradient_steps:", model.gradient_steps)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(total_timesteps=args.n_steps, callback=eval_callback)

    model.save(f"{model_dir}/tqc_her_final")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    run()

