import time

from sbx import TQC
import gymnasium as gym


ENV_NAME = "FrankaPickAndPlaceSparse-v0"
ENV_ID = f"panda_mujoco_gym:{ENV_NAME}"
MODEL_PATH=f"./logs/{ENV_ID}/best_model"

N_EPISODES = 50

def run():
    #gym.register_envs(gymnasium_robotics)
    #env = gym.make(ENV_ID, render_mode='human')
    env = gym.make(ENV_ID, render_mode="human")


    model = TQC.load(
        MODEL_PATH,
        env=env,
        custom_objects={"buffer_size": 1, "learning_satrts": 0},
        device="cpu",
    )

    successes = 0
    for ep in range(N_EPISODES):
        obs, info = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.03)


        success = bool(info.get("is_success", 0.0))
        successes += success
        print(f"episode: {ep +1 }, success: {success}")

    print(f"success rate: {successes/N_EPISODES}")
    env.close()


if __name__ == "__main__":
    run()