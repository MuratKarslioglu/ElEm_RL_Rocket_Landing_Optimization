import time
from typing import Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from stable_baselines3 import PPO


# =========================
# MODEL PATHS
# =========================

TAKEOFF_MODEL_PATH = "ppo_takeoff_hover_best.zip"
LANDING_MODEL_PATH = "ppo_lunar_lander_model.zip"


# =========================
# PHASES
# =========================

PHASE_TAKEOFF_HOVER = 0
PHASE_LANDING = 1


# =========================
# SETTINGS
# =========================

FPS = 50
FPS_DELAY = 0.02
PAUSED_DELAY = 0.05

MAX_EPISODE_STEPS = 1000

HOVER_SECONDS = 3
HOVER_REQUIRED_STEPS = HOVER_SECONDS * FPS

TARGET_HEIGHT_MIN = 0.45
TARGET_HEIGHT_MAX = 0.75

MAX_HOVER_HORIZONTAL_SPEED = 0.18
MAX_REASONABLE_HORIZONTAL_SPEED = 0.45
MAX_REASONABLE_HEIGHT_MARGIN = 0.45

GROUND_BODY_OFFSET = 0.65

LANDING_ZONE_LIMIT = 0.20
MAX_VERTICAL_SPEED = 0.35
MAX_HORIZONTAL_SPEED = 0.25
MAX_ANGLE = 0.20
MAX_ANGULAR_VELOCITY = 1.00

PANEL_WIDTH = 360
PANEL_HEIGHT = 215
PANEL_MARGIN = 10
PANEL_ALPHA = 210
FONT_SIZE = 14


ACTION_MEANINGS = {
    0: "No-op",
    1: "Left engine",
    2: "Main engine",
    3: "Right engine",
}


# =========================
# LOAD MODELS
# =========================

takeoff_model = PPO.load(TAKEOFF_MODEL_PATH)
landing_model = PPO.load(LANDING_MODEL_PATH)


# =========================
# ENV WRAPPER
# =========================

class CombinedTakeoffLandingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.target_height = TARGET_HEIGHT_MIN
        self.hover_steps = 0
        self.step_count = 0
        self.phase = PHASE_TAKEOFF_HOVER

        low = np.concatenate([
            self.env.observation_space.low,
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ])

        high = np.concatenate([
            self.env.observation_space.high,
            np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32),
        ])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.target_height = float(np.random.uniform(TARGET_HEIGHT_MIN, TARGET_HEIGHT_MAX))
        self.hover_steps = 0
        self.step_count = 0
        self.phase = PHASE_TAKEOFF_HOVER

        self._place_lander_on_ground()
        obs = self._get_raw_observation_after_manual_reset(obs)

        info = dict(info)
        info["phase"] = self.phase
        info["target_height"] = self.target_height
        info["hover_steps"] = self.hover_steps

        return self._augment_observation(obs), info

    def step(self, action):
        raw_obs, original_reward, original_terminated, original_truncated, info = self.env.step(action)
        self.step_count += 1

        if self.phase == PHASE_TAKEOFF_HOVER:
            reward, phase_done, status = self._takeoff_hover_reward(raw_obs)

            if phase_done:
                self.phase = PHASE_LANDING
                reward += 100.0
                status = "HOVER COMPLETE / LANDING"

            terminated = False

        else:
            reward = float(original_reward)
            terminated = bool(original_terminated)

            if terminated:
                landing_status, landing_success, task_score, physical_score = self._evaluate_landing(raw_obs, reward)
                status = landing_status

                if landing_success:
                    reward += 200.0
                else:
                    reward -= 100.0
            else:
                status = "LANDING"

        truncated = bool(original_truncated or self.step_count >= MAX_EPISODE_STEPS)

        if truncated:
            status = "TIME LIMIT"

        info = dict(info)
        info["phase"] = self.phase
        info["target_height"] = self.target_height
        info["hover_steps"] = self.hover_steps
        info["hover_required_steps"] = HOVER_REQUIRED_STEPS
        info["status"] = status

        return self._augment_observation(raw_obs), float(reward), terminated, truncated, info

    def _place_lander_on_ground(self):
        base = self.env.unwrapped
        lander = getattr(base, "lander", None)

        if lander is None:
            return

        viewport_w = float(getattr(base, "VIEWPORT_W", 600))
        scale = float(getattr(base, "SCALE", 30))
        helipad_y = float(getattr(base, "helipad_y", 100 / scale))

        start_x = viewport_w / scale / 2.0
        start_y = helipad_y + GROUND_BODY_OFFSET

        old_x = float(lander.position.x)
        old_y = float(lander.position.y)

        dx = start_x - old_x
        dy = start_y - old_y

        lander.position = (start_x, start_y)
        lander.linearVelocity = (0.0, 0.0)
        lander.angle = 0.0
        lander.angularVelocity = 0.0
        lander.force = (0.0, 0.0)
        lander.torque = 0.0
        lander.awake = True

        for leg in getattr(base, "legs", []):
            leg.position = (float(leg.position.x) + dx, float(leg.position.y) + dy)
            leg.linearVelocity = (0.0, 0.0)
            leg.angle = 0.0
            leg.angularVelocity = 0.0
            leg.force = (0.0, 0.0)
            leg.torque = 0.0
            leg.awake = True

    def _get_raw_observation_after_manual_reset(self, fallback_obs):
        base = self.env.unwrapped

        if hasattr(base, "_get_observation"):
            return base._get_observation()

        if hasattr(base, "_get_obs"):
            return base._get_obs()

        obs = np.array(fallback_obs, dtype=np.float32).copy()
        obs[0] = 0.0
        obs[1] = 0.0
        obs[2] = 0.0
        obs[3] = 0.0
        obs[4] = 0.0
        obs[5] = 0.0
        obs[6] = 1.0
        obs[7] = 1.0
        return obs

    def _augment_observation(self, raw_obs):
        raw_obs = np.array(raw_obs, dtype=np.float32)

        y_pos = float(raw_obs[1])
        is_above_target = 1.0 if y_pos >= self.target_height else 0.0
        hover_progress = min(1.0, self.hover_steps / HOVER_REQUIRED_STEPS)
        phase_value = float(self.phase)

        extra = np.array([
            self.target_height,
            hover_progress,
            is_above_target,
            phase_value,
        ], dtype=np.float32)

        return np.concatenate([raw_obs, extra]).astype(np.float32)

    def _takeoff_hover_reward(self, obs) -> Tuple[float, bool, str]:
        x_pos = float(obs[0])
        y_pos = float(obs[1])
        x_vel = float(obs[2])
        y_vel = float(obs[3])
        angle = float(obs[4])
        angular_vel = float(obs[5])
        left_leg = float(obs[6])
        right_leg = float(obs[7])

        is_on_ground = left_leg > 0.5 or right_leg > 0.5
        is_above_target = y_pos >= self.target_height
        horizontal_ok = abs(x_vel) <= MAX_HOVER_HORIZONTAL_SPEED
        angle_ok = abs(angle) <= 0.25

        stable_hover = (
            is_above_target
            and not is_on_ground
            and horizontal_ok
            and angle_ok
        )

        reward = 0.0

        if is_on_ground:
            reward -= 3.0

        height_error = max(0.0, self.target_height - y_pos)
        reward += 2.0 * (1.0 - min(1.0, height_error / self.target_height))

        if stable_hover:
            reward += 3.0
            self.hover_steps += 1
        else:
            self.hover_steps = 0

        if is_above_target and not is_on_ground:
            reward += 1.0 * max(0.0, 1.0 - abs(x_vel) / MAX_HOVER_HORIZONTAL_SPEED)
            reward -= 8.0 * max(0.0, abs(x_vel) - MAX_HOVER_HORIZONTAL_SPEED)
        else:
            reward -= 0.5 * abs(x_vel)

        reward -= 0.25 * abs(y_vel)
        reward -= 0.6 * abs(angle)
        reward -= 0.15 * abs(angular_vel)
        reward -= 0.15 * abs(x_pos)

        max_reasonable_height = self.target_height + MAX_REASONABLE_HEIGHT_MARGIN
        if y_pos > max_reasonable_height:
            reward -= 2.0 * (y_pos - max_reasonable_height)

        reward += 0.005 * self.hover_steps

        failed = (
            abs(x_pos) > 1.4
            or abs(angle) > 1.2
            or abs(x_vel) > MAX_REASONABLE_HORIZONTAL_SPEED * 4
        )

        if failed:
            return reward - 80.0, False, "FAILED / TAKEOFF UNSTABLE"

        phase_done = self.hover_steps >= HOVER_REQUIRED_STEPS

        if phase_done:
            return reward + 150.0, True, "HOVER COMPLETE"

        if is_on_ground:
            return reward, False, "TAKEOFF"
        if not is_above_target:
            return reward, False, "CLIMBING"
        if not horizontal_ok:
            return reward, False, "HOVER / X SPEED HIGH"

        return reward, False, "HOVERING"

    def _evaluate_landing(self, obs, reward):
        x_pos = float(obs[0])
        x_vel = float(obs[2])
        y_vel = float(obs[3])
        angle = float(obs[4])
        angular_vel = float(obs[5])
        left_leg = float(obs[6])
        right_leg = float(obs[7])

        both_legs_contact = int(left_leg) == 1 and int(right_leg) == 1

        if not both_legs_contact:
            return "FAILED / CRASHED", False, 0, 0

        in_landing_zone = abs(x_pos) <= LANDING_ZONE_LIMIT
        slow_vertical = abs(y_vel) <= MAX_VERTICAL_SPEED
        slow_horizontal = abs(x_vel) <= MAX_HORIZONTAL_SPEED
        stable_angle = abs(angle) <= MAX_ANGLE
        stable_rotation = abs(angular_vel) <= MAX_ANGULAR_VELOCITY

        physical_score = 5

        if in_landing_zone:
            physical_score += 30
        if slow_vertical:
            physical_score += 25
        if slow_horizontal:
            physical_score += 15
        if stable_angle:
            physical_score += 15
        if stable_rotation:
            physical_score += 10

        success = (
            both_legs_contact
            and in_landing_zone
            and slow_vertical
            and slow_horizontal
            and stable_angle
            and stable_rotation
        )

        if success:
            return "SUCCESSFUL TAKEOFF + LANDING", True, 100, physical_score

        return "LANDING FAILED / LOW QUALITY", False, physical_score, physical_score


# =========================
# ACTION SELECTION
# =========================

def choose_action(observation, phase):
    if phase == PHASE_TAKEOFF_HOVER:
        takeoff_obs = observation[:11]
        action, _ = takeoff_model.predict(takeoff_obs, deterministic=True)
        return int(action)

    landing_obs = observation[:8]
    action, _ = landing_model.predict(landing_obs, deterministic=True)
    return int(action)


def sanitize_action(action):
    try:
        action = int(action)
    except Exception:
        return 0

    if action not in ACTION_MEANINGS:
        return 0

    return action


# =========================
# UI
# =========================

def draw_info_panel(env, font, episode_data):
    screen = getattr(env.unwrapped, "screen", None)

    if screen is None:
        return

    width, _ = screen.get_size()
    panel_x = width - PANEL_WIDTH - PANEL_MARGIN
    panel_y = PANEL_MARGIN

    panel_surface = pygame.Surface((PANEL_WIDTH, PANEL_HEIGHT))
    panel_surface.set_alpha(PANEL_ALPHA)
    panel_surface.fill((20, 20, 20))
    screen.blit(panel_surface, (panel_x, panel_y))

    phase_text = "TAKEOFF/HOVER" if episode_data["phase"] == PHASE_TAKEOFF_HOVER else "LANDING"

    lines = [
        f"Phase: {phase_text}",
        f"Status: {episode_data['status']}",
        f"Step: {episode_data['step_count']}/{MAX_EPISODE_STEPS}",
        f"Total Reward: {episode_data['total_reward']:.2f}",
        f"Target Height: {episode_data['target_height']:.2f}",
        f"Hover Step: {episode_data['hover_steps']}/{HOVER_REQUIRED_STEPS}",
        f"Last Action: {episode_data['last_action']}",
        f"SPACE: pause / next",
    ]

    y_offset = panel_y + 10

    for line in lines:
        color = (255, 255, 255)
        if line.startswith("Status"):
            if "SUCCESSFUL" in line or "COMPLETE" in line:
                color = (80, 255, 120)
            elif "FAILED" in line or "TIME LIMIT" in line:
                color = (255, 90, 90)
            else:
                color = (255, 190, 80)

        text_surface = font.render(line, True, color)
        screen.blit(text_surface, (panel_x + 10, y_offset))
        y_offset += 21

    pygame.display.flip()


def reset_episode(env, episode):
    observation, info = env.reset()

    episode_data = {
        "episode": episode,
        "step_count": 0,
        "total_reward": 0.0,
        "last_action": None,
        "status": "RUNNING",
        "phase": PHASE_TAKEOFF_HOVER,
        "target_height": float(info.get("target_height", 0.0)),
        "hover_steps": 0,
        "waiting_for_space": False,
        "paused": False,
    }

    return observation, info, episode_data


def print_episode_result(episode_data):
    print("-" * 50)
    print(f"Episode: {episode_data['episode']}")
    print(f"Step: {episode_data['step_count']}")
    print(f"Total Reward: {episode_data['total_reward']:.2f}")
    print(f"Status: {episode_data['status']}")
    print(f"Phase: {episode_data['phase']}")
    print(f"Target Height: {episode_data['target_height']:.2f}")
    print(f"Hover Step: {episode_data['hover_steps']}/{HOVER_REQUIRED_STEPS}")
    print("Yeni simülasyon için SPACE tuşuna bas.")
    print("-" * 50)


def handle_keydown(event, env, observation, info, episode_data):
    if event.key != pygame.K_SPACE:
        return observation, info, episode_data

    if episode_data["waiting_for_space"]:
        next_episode = episode_data["episode"] + 1
        return reset_episode(env, next_episode)

    episode_data["paused"] = not episode_data["paused"]
    return observation, info, episode_data


# =========================
# MAIN
# =========================

def main():
    pygame.init()

    base_env = gym.make("LunarLander-v3", render_mode="human")
    env = CombinedTakeoffLandingWrapper(base_env)

    observation, info, episode_data = reset_episode(env, episode=1)
    font = pygame.font.SysFont("Arial", FONT_SIZE)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                observation, info, episode_data = handle_keydown(
                    event,
                    env,
                    observation,
                    info,
                    episode_data,
                )

        if not running:
            break

        if episode_data["waiting_for_space"] or episode_data["paused"]:
            draw_info_panel(env, font, episode_data)
            time.sleep(PAUSED_DELAY)
            continue

        action = choose_action(observation, episode_data["phase"])
        action = sanitize_action(action)

        observation, reward, terminated, truncated, info = env.step(action)

        episode_data["last_action"] = action
        episode_data["total_reward"] += float(reward)
        episode_data["step_count"] += 1
        episode_data["phase"] = int(info.get("phase", episode_data["phase"]))
        episode_data["status"] = str(info.get("status", "RUNNING"))
        episode_data["target_height"] = float(info.get("target_height", episode_data["target_height"]))
        episode_data["hover_steps"] = int(info.get("hover_steps", 0))

        if terminated or truncated:
            episode_data["waiting_for_space"] = True
            episode_data["paused"] = False
            print_episode_result(episode_data)

        draw_info_panel(env, font, episode_data)
        time.sleep(FPS_DELAY)

    env.close()
    pygame.quit()
    print("Program kapatıldı.")


if __name__ == "__main__":
    main()