"""
Gymnasium-compatible MuJoCo VTOL landing environment.

This file is intentionally separate from simulation.py so the original
LunarLander assignment environment stays unchanged.

Run a random-policy smoke test:
    python3 mujoco_vtol_env.py
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


MODEL_PATH = Path(__file__).resolve().parent / "assets" / "mujoco" / "vtol_assembly_scene.xml"


class MujocoVtolLandingEnv(gym.Env):
    """
    Discrete-action VTOL landing task using the imported MuJoCo mesh.

    Actions match the original LunarLander-style assignment:
        0 = no-op
        1 = left engine / rotate right
        2 = main engine
        3 = right engine / rotate left

    Observation:
        [x, z, x_velocity, z_velocity, pitch, pitch_velocity, altitude, uprightness]
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.max_steps = int(max_steps)

        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "vtol_body")
        self.ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.collision_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "vtol_collision")

        if self.body_id < 0 or self.ground_id < 0 or self.collision_id < 0:
            raise RuntimeError("MuJoCo model is missing required body or geom names.")

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.inf, 0.0, -1.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.pi, np.inf, np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.main_thrust = 16.0
        self.side_thrust = 5.0
        self.pitch_torque = 1.2
        self.landing_x_limit = 0.35
        self.safe_vertical_speed = 0.45
        self.safe_horizontal_speed = 0.35
        self.safe_pitch = 0.35

        self.viewer = None
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        start_x = self.np_random.uniform(-0.35, 0.35)
        start_z = self.np_random.uniform(2.2, 2.8)
        start_pitch = self.np_random.uniform(-0.15, 0.15)

        self.data.qpos[0:3] = np.array([start_x, 0.0, start_z])
        self.data.qpos[3:7] = self._pitch_to_quat(start_pitch)
        self.data.qvel[0:6] = 0.0
        self.data.qvel[0] = self.np_random.uniform(-0.15, 0.15)
        self.data.qvel[2] = self.np_random.uniform(-0.2, 0.0)

        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        self.steps += 1

        self.data.xfrc_applied[:, :] = 0.0
        force, torque = self._action_to_force(action)
        self.data.xfrc_applied[self.body_id, 0:3] = force
        self.data.xfrc_applied[self.body_id, 3:6] = torque

        for _ in range(4):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()

        landed = info["ground_contact"]
        crashed = landed and not info["safe_landing"]
        out_of_bounds = abs(obs[0]) > 3.0 or obs[1] < -0.2
        terminated = bool(info["safe_landing"] or crashed or out_of_bounds)
        truncated = self.steps >= self.max_steps

        reward = self._calculate_reward(obs, info, action)

        if info["safe_landing"]:
            reward += 100.0
        elif crashed:
            reward -= 100.0
        elif out_of_bounds:
            reward -= 50.0

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                "overview",
            )

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _action_to_force(self, action):
        force = np.zeros(3, dtype=np.float64)
        torque = np.zeros(3, dtype=np.float64)

        if action == 1:
            force[2] = self.side_thrust
            torque[1] = self.pitch_torque
        elif action == 2:
            force[2] = self.main_thrust
        elif action == 3:
            force[2] = self.side_thrust
            torque[1] = -self.pitch_torque

        return force, torque

    def _get_obs(self):
        x_pos = float(self.data.qpos[0])
        z_pos = float(self.data.qpos[2])
        x_vel = float(self.data.qvel[0])
        z_vel = float(self.data.qvel[2])
        pitch = float(self._quat_to_pitch(self.data.qpos[3:7]))
        pitch_vel = float(self.data.qvel[4])
        altitude = max(0.0, z_pos)
        uprightness = float(np.cos(pitch))

        return np.array(
            [x_pos, z_pos, x_vel, z_vel, pitch, pitch_vel, altitude, uprightness],
            dtype=np.float32,
        )

    def _get_info(self):
        obs = self._get_obs()
        ground_contact = self._has_ground_contact()
        safe_landing = (
            ground_contact
            and abs(obs[0]) <= self.landing_x_limit
            and abs(obs[2]) <= self.safe_horizontal_speed
            and abs(obs[3]) <= self.safe_vertical_speed
            and abs(obs[4]) <= self.safe_pitch
        )

        return {
            "ground_contact": bool(ground_contact),
            "safe_landing": bool(safe_landing),
            "step": self.steps,
        }

    def _calculate_reward(self, obs, info, action):
        x_pos, _, x_vel, z_vel, pitch, pitch_vel, altitude, _ = obs

        reward = 1.0
        reward -= 1.2 * abs(x_pos)
        reward -= 0.4 * abs(x_vel)
        reward -= 0.5 * abs(z_vel)
        reward -= 0.8 * abs(pitch)
        reward -= 0.1 * abs(pitch_vel)
        reward -= 0.03 * float(action != 0)
        reward -= 0.02 * altitude

        if info["ground_contact"] and not info["safe_landing"]:
            reward -= 10.0

        return reward

    def _has_ground_contact(self):
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.ground_id in pair and self.collision_id in pair:
                return True
        return False

    @staticmethod
    def _pitch_to_quat(pitch):
        half_angle = 0.5 * pitch
        return np.array([np.cos(half_angle), 0.0, np.sin(half_angle), 0.0], dtype=np.float64)

    @staticmethod
    def _quat_to_pitch(quat):
        w, _, y, _ = quat
        return np.arctan2(2.0 * w * y, 1.0 - 2.0 * y * y)


def main():
    env = MujocoVtolLandingEnv(render_mode="human")
    observation, info = env.reset()

    try:
        for _ in range(1000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(f"Episode finished | reward={reward:.2f} | info={info}")
                observation, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
