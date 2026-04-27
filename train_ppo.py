import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = gym.make("LunarLander-v3")
env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./tensorboard_logs/ppo/"
)

model.learn(
    total_timesteps=750000,
    tb_log_name="PPO_LunarLander",
    reset_num_timesteps=True
)

model.save("ppo_lunar_lander_model")

env.close()