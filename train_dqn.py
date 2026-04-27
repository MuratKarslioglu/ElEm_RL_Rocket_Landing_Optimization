import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


ENV_NAME = "LunarLander-v3"
MODEL_PATH = "dqn_lunar_lander_model"


def main():
    # Eğitim sırasında render açmıyoruz.
    # Çünkü görüntü çizdirmek eğitimi çok yavaşlatır.
    env = gym.make(ENV_NAME)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=20_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.35,
        exploration_final_eps=0.01,
        verbose=1,
)


    # İlk deneme için orta seviye eğitim.
    # Sonra bunu artıracağız.
    model.learn(total_timesteps=3_000_000)

    model.save(MODEL_PATH)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=50,
        deterministic=True,
    )

    print("Model kaydedildi:", MODEL_PATH)
    print(f"Ortalama reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()