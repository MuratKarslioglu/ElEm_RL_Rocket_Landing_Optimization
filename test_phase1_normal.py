import gymnasium as gym
from stable_baselines3 import PPO


MODEL_PATH = "models/direct_landing_from_scratch.zip"


def main():
    env = gym.make("LunarLander-v3", render_mode="human")
    model = PPO.load(MODEL_PATH)

    for episode in range(20):
        obs, info = env.reset()

        terminated = False
        truncated = False
        total_reward = 0.0

        first_contact_x = None
        max_post_contact_drift = 0.0

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            x_pos = float(obs[0])
            left_leg = float(obs[6])
            right_leg = float(obs[7])
            any_contact = left_leg > 0.5 or right_leg > 0.5

            if any_contact and first_contact_x is None:
                first_contact_x = x_pos

            if first_contact_x is not None:
                drift = abs(x_pos - first_contact_x)
                max_post_contact_drift = max(max_post_contact_drift, drift)

        print(
            f"Episode {episode + 1} | "
            f"Total reward: {total_reward:.2f} | "
            f"Max post-contact drift: {max_post_contact_drift:.4f}"
        )

    env.close()


if __name__ == "__main__":
    main()