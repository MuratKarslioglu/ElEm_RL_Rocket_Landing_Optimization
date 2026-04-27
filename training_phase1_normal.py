import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from direct_landing_reward import DirectLandingRewardWrapper


MODEL_DIR = "models"
LOG_DIR = "logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def make_env():
    env = gym.make("LunarLander-v3")
    env = DirectLandingRewardWrapper(env)
    env = Monitor(env)
    return env


def main():
    env = make_env()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    model.learn(
        total_timesteps=500_000,
        tb_log_name="direct_landing_from_scratch",
    )

    model.save(os.path.join(MODEL_DIR, "direct_landing_from_scratch"))

    env.close()


if __name__ == "__main__":
    main()