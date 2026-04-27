import gymnasium as gym
from stable_baselines3 import PPO


MODEL_PATH = "models/direct_landing_strict_v1.zip"

N_EPISODES = 20

# Başarı kriterleri
MIN_REWARD_FOR_SUCCESS = 150.0
MAX_ALLOWED_DRIFT = 0.05


def choose_safe_action(model, obs):
    """
    Test sırasında güvenlik filtresi uygular.

    Eğer roket iki bacakla yere temas ettiyse artık motor çalıştırılmaz.
    Böylece yerde motor basıp kendini kaydırması engellenir.
    """

    left_leg = float(obs[6])
    right_leg = float(obs[7])

    both_contact = left_leg > 0.5 and right_leg > 0.5

    if both_contact:
        return 0

    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def main():
    env = gym.make("LunarLander-v3", render_mode="human")
    model = PPO.load(MODEL_PATH)

    success_count = 0
    total_rewards = []
    drift_values = []

    consecutive_success = 0
    max_consecutive_success = 0

    for episode in range(1, N_EPISODES + 1):
        obs, info = env.reset()

        terminated = False
        truncated = False
        total_reward = 0.0

        first_contact_x = None
        max_post_contact_drift = 0.0

        step_count = 0

        final_x = 0.0
        final_x_vel = 0.0
        final_y_vel = 0.0
        final_angle = 0.0
        final_angular_vel = 0.0
        final_left_leg = 0.0
        final_right_leg = 0.0

        while not terminated and not truncated:
            action = choose_safe_action(model, obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1

            x_pos = float(obs[0])
            x_vel = float(obs[2])
            y_vel = float(obs[3])
            angle = float(obs[4])
            angular_vel = float(obs[5])
            left_leg = float(obs[6])
            right_leg = float(obs[7])

            any_contact = left_leg > 0.5 or right_leg > 0.5

            if any_contact and first_contact_x is None:
                first_contact_x = x_pos

            if first_contact_x is not None:
                drift = abs(x_pos - first_contact_x)
                max_post_contact_drift = max(max_post_contact_drift, drift)

            final_x = x_pos
            final_x_vel = x_vel
            final_y_vel = y_vel
            final_angle = angle
            final_angular_vel = angular_vel
            final_left_leg = left_leg
            final_right_leg = right_leg

        both_contact_final = final_left_leg > 0.5 and final_right_leg > 0.5

        in_landing_zone = abs(final_x) <= 0.20
        slow_horizontal = abs(final_x_vel) <= 0.25
        slow_vertical = abs(final_y_vel) <= 0.35
        stable_angle = abs(final_angle) <= 0.20
        stable_rotation = abs(final_angular_vel) <= 1.00

        physically_correct = (
            both_contact_final
            and in_landing_zone
            and slow_horizontal
            and slow_vertical
            and stable_angle
            and stable_rotation
        )

        direct_success = (
            total_reward >= MIN_REWARD_FOR_SUCCESS
            and max_post_contact_drift <= MAX_ALLOWED_DRIFT
            and physically_correct
            and not truncated
        )

        if direct_success:
            success_count += 1
            consecutive_success += 1
        else:
            consecutive_success = 0

        max_consecutive_success = max(max_consecutive_success, consecutive_success)

        total_rewards.append(total_reward)
        drift_values.append(max_post_contact_drift)

        print(
            f"Episode {episode:02d} | "
            f"Reward: {total_reward:8.2f} | "
            f"Drift: {max_post_contact_drift:7.4f} | "
            f"Steps: {step_count:4d} | "
            f"Physical: {'YES' if physically_correct else 'NO ':3s} | "
            f"Direct Success: {'YES' if direct_success else 'NO'}"
        )

    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_drift = sum(drift_values) / len(drift_values)
    max_drift = max(drift_values)

    print("-" * 90)
    print(f"Total episodes: {N_EPISODES}")
    print(f"Direct success count: {success_count}/{N_EPISODES}")
    print(f"Success rate: {(success_count / N_EPISODES) * 100:.1f}%")
    print(f"Max consecutive success: {max_consecutive_success}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average post-contact drift: {avg_drift:.4f}")
    print(f"Max post-contact drift: {max_drift:.4f}")
    print("-" * 90)

    env.close()


if __name__ == "__main__":
    main()