import time

from sbx import TQC
import gymnasium as gym

from gymnasium.wrappers import RecordVideo

#TODO: clean up this mess
#ENV_NAME = "FrankaPickAndPlaceSparse-v0"
ENV_NAME = "FrankaPushSparse-v0"

ENV_ID = f"panda_mujoco_gym:{ENV_NAME}"

#MODEL_PATH=f"exp_out/tqc_test_FrankaPickAndPlaceSparse-v0/FrankaPickAndPlaceSparse-v0/1/models/best_model"
MODEL_PATH=f"exp_out/tqc_test_FrankaPushSparse-v0/FrankaPushSparse-v0/1/models/best_model"
N_EPISODES = 50
N_RECORD=10

def run():
    env = gym.make(ENV_ID, render_mode="rgb_array") # human
    max_steps = env.spec.max_episode_steps

    env = RecordVideo(
        env,
        video_folder="videos",
        name_prefix= f"tqc_{ENV_NAME}",
        episode_trigger=lambda ep: ep == 0,
        video_length= N_RECORD * (max_steps + 1),
        #fps = env.metadata["render_fps"] // 2,
    )

    model = TQC.load(
        MODEL_PATH,
        env=env,
        custom_objects={"buffer_size": 1, "learning_starts": 0},
        device="cpu",
    )

    successes = 0
    for ep in range(N_EPISODES):
        obs, info = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.05)


        success = bool(info.get("is_success", 0.0))
        successes += success
        print(f"episode: {ep +1 }, success: {success}")

    print(f"success rate: {successes/N_EPISODES}")
    env.close()


if __name__ == "__main__":
    run()