from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np


@dataclass
class DRConfig:
    ee_noise_std: tuple = (0.001, 0.003)  # [m] per-episode for EE position
    action_gain: tuple = (0.8, 1.2)       # per-episode multiplier
    action_noise_std: float = 0.02        # per-step
    max_delay: int = 2                    # delay in steps; from 0 to max
    mass_scale: tuple = (0.95, 1.05)      # cube mass multiplier
    joint_noise: float = 0.1              # [rad] uniform noise


class DomainRandomization(gym.Wrapper):
    def __init__(self, env, cfg=None):
        super().__init__(env)
        self.cfg = cfg or DRConfig()
        model = self.unwrapped.model
        self.obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.nominal_mass = model.body_mass[self.obj_id].copy()
        self.nominal_inertia = model.body_inertia[self.obj_id].copy()

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        rng, cfg, model = self.np_random, self.cfg, self.unwrapped.model

        # sample this episode's parameters
        self.ee_std = rng.uniform(*cfg.ee_noise_std)
        self.gain = rng.uniform(*cfg.action_gain, size=self.action_space.shape)
        delay = rng.integers(0, cfg.max_delay + 1)
        self.queue = deque([np.zeros(self.action_space.shape, dtype=np.float32)] * delay)

        scale = rng.uniform(*cfg.mass_scale)
        model.body_mass[self.obj_id] = self.nominal_mass * scale
        model.body_inertia[self.obj_id] = self.nominal_inertia * scale

        if cfg.joint_noise > 0:
            obs = self._randomize_arm()
        return self._add_obs_noise(obs), info

    def step(self, action):
        noise = self.np_random.normal(0.0, self.cfg.action_noise_std, size=action.shape)
        action = np.clip(action * self.gain + noise, self.action_space.low, self.action_space.high)

        self.queue.append(action.astype(np.float32))

        obs, reward, terminated, truncated, info = self.env.step(self.queue.popleft())
        return self._add_obs_noise(obs), reward, terminated, truncated, info

    def _add_obs_noise(self, obs):
        obs = dict(obs)
        obs["observation"] = obs["observation"].copy()
        obs["observation"][:3] += self.np_random.normal(0.0, self.ee_std, size=3)  # EE pos only
        return obs

    def _randomize_arm(self):
        env = self.unwrapped
        model, data = env.model, env.data
        obj_qpos = data.qpos[9:16].copy()  # qpos layout: 7 arm, 2 fingers, 7 cube free joint

        noise = self.np_random.uniform(-self.cfg.joint_noise, self.cfg.joint_noise, size=7)
        data.qpos[:7] = np.clip(data.qpos[:7] + noise, model.jnt_range[:7, 0], model.jnt_range[:7, 1])
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        env.set_mocap_pose(env.get_ee_position().copy(), env.grasp_site_pose)
        mujoco.mj_step(model, data, nstep=500)

        data.qpos[9:16] = obj_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        return env._get_obs()