import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from direct_landing_reward import DirectLandingRewardWrapper


MODEL_DIR = "models"
LOG_DIR = "logs"

OLD_MODEL_PATH = os.path.join(MODEL_DIR, "direct_landing_from_scratch_v2.zip")
NEW_MODEL_PATH = os.path.join(MODEL_DIR, "direct_landing_strict_v1")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env():
    env = gym.make("LunarLander-v3")
    env = DirectLandingRewardWrapper(env)
    env = Monitor(env)
    return env


def main():
    env = make_env()

    model = PPO.load(OLD_MODEL_PATH, env=env)

    # Fine-tuning: modeli fazla dağıtmadan yeni reward'a adapte et.
    model.learning_rate = 1e-4
    model.ent_coef = 0.005

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=MODEL_DIR,
        name_prefix="direct_landing_strict_checkpoint",
    )

    model.learn(
        total_timesteps=1_000_000,
        reset_num_timesteps=False,
        tb_log_name="direct_landing_strict_v1",
        callback=checkpoint_callback,
    )

    model.save(NEW_MODEL_PATH)

    env.close()


if __name__ == "__main__":
    main()