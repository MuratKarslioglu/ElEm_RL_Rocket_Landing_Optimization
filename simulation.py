"""
LunarLander-v3 öğrenci görev ortamı.

Öğrenciler yalnızca STUDENT AREA bölümündeki choose_action(observation)
fonksiyonunu değiştirecektir. Ortam kontrolü, metrikler, iniş değerlendirmesi,
panel çizimi ve simülasyon akışı değiştirilmemelidir.

Kontroller:
- SPACE: Simülasyon çalışırken pause/resume yapar.
- SPACE: Episode bittiyse yeni simülasyon başlatır.
- Pencere kapatılırsa program kapanır.
"""

import random
import time

import gymnasium as gym
import pygame


# ============================================================
# STUDENT AREA
# Öğrenciler sadece bu fonksiyonu değiştirecek.
# ============================================================
from stable_baselines3 import PPO

MODEL_PATH = "models/direct_landing_from_scratch.zip"

ppo_model = PPO.load(MODEL_PATH)
def choose_action(observation):
    """
    Öğrencilerin görevi:
    observation değerlerine bakarak 0, 1, 2 veya 3 aksiyonlarından
    birini döndürmek.

    Aksiyonlar:
    0 = hiçbir şey yapma
    1 = sol yön motoru
    2 = ana motor
    3 = sağ yön motoru

    Observation:
    observation[0] = x konumu
    observation[1] = y konumu
    observation[2] = x hızı
    observation[3] = y hızı
    observation[4] = açı
    observation[5] = açısal hız
    observation[6] = sol bacak temas durumu
    observation[7] = sağ bacak temas durumu
    """
    left_leg = float(observation[6])
    right_leg = float(observation[7])

    if left_leg > 0.5 and right_leg > 0.5:
        return 0

    action, _ = ppo_model.predict(observation, deterministic=True)
    return int(action)
    # Başlangıç kontrolcüsü: rastgele hareket eder.
    # Öğrenciler bu satırı kendi karar algoritmalarıyla değiştirecek.
    #return random.randint(0, 3)


# ============================================================
# TEACHER / ENVIRONMENT AREA
# Öğrenciler bu bölümün altını değiştirmemeli.
# ============================================================

ACTION_MEANINGS = {
    0: "No-op",
    1: "Left engine",
    2: "Main engine",
    3: "Right engine",
}

# Göreve özel fiziksel iniş kalite eşikleri
LANDING_ZONE_LIMIT = 0.20
MAX_VERTICAL_SPEED = 0.35
MAX_HORIZONTAL_SPEED = 0.25
MAX_ANGLE = 0.20
MAX_ANGULAR_VELOCITY = 1.00

# Reward tabanlı başarı eşikleri
MIN_TOTAL_REWARD_FOR_CORRECT_LANDING = 0.0
MIN_TOTAL_REWARD_FOR_EXCELLENT_LANDING = 200.0

# Final görev skoru ayarları
# Fiziksel olarak düzgün inse bile total_reward negatifse öğrenciye 100/100 verilmez.
LOW_REWARD_SCORE_CAP = 70

# Simülasyon ayarları
FPS_DELAY = 0.02
PAUSED_DELAY = 0.05

# Panel ayarları
PANEL_WIDTH = 285
PANEL_HEIGHT = 145
PANEL_MARGIN = 10
PANEL_ALPHA = 210
FONT_SIZE = 14


def sanitize_action(action):
    """
    Öğrenci fonksiyonundan gelen aksiyonu güvenli hale getirir.
    Geçersiz aksiyon verilirse 0 aksiyonuna çevrilir.
    """

    if isinstance(action, bool):
        return 0

    try:
        action = int(action)
    except (TypeError, ValueError):
        return 0

    if action not in ACTION_MEANINGS:
        return 0

    return action


def extract_lander_state(observation):
    """
    Observation dizisini isimlendirilmiş değerlere dönüştürür.
    """

    return {
        "x_pos": float(observation[0]),
        "y_pos": float(observation[1]),
        "x_vel": float(observation[2]),
        "y_vel": float(observation[3]),
        "angle": float(observation[4]),
        "angular_vel": float(observation[5]),
        "left_leg": float(observation[6]),
        "right_leg": float(observation[7]),
    }


def calculate_landing_score(state):
    """
    İniş kalitesini 0-100 arasında puanlar.

    Kritik kural:
    İki bacak temas etmiyorsa skor doğrudan 0'dır.
    Böylece çakılma veya tek bacaklı temas durumunda kısmi puan verilmez.
    """

    both_legs_contact = int(state["left_leg"]) == 1 and int(state["right_leg"]) == 1

    checks = {
        "both_legs_contact": both_legs_contact,
        "in_landing_zone": abs(state["x_pos"]) <= LANDING_ZONE_LIMIT,
        "slow_vertical_landing": abs(state["y_vel"]) <= MAX_VERTICAL_SPEED,
        "slow_horizontal_motion": abs(state["x_vel"]) <= MAX_HORIZONTAL_SPEED,
        "stable_angle": abs(state["angle"]) <= MAX_ANGLE,
        "stable_angular_velocity": abs(state["angular_vel"]) <= MAX_ANGULAR_VELOCITY,
    }

    if not checks["both_legs_contact"]:
        return 0, checks

    score = 0

    if checks["in_landing_zone"]:
        score += 45
    if checks["slow_vertical_landing"]:
        score += 35
    if checks["slow_horizontal_motion"]:
        score += 20
    if checks["stable_angle"]:
        score += 20
    if checks["stable_angular_velocity"]:
        score += 20

    # İki bacak teması zorunlu ön koşuldur, küçük tamamlayıcı puan verilir.
    score += 5

    return score, checks


def calculate_task_score(physical_score, total_reward, physically_correct_landing):
    """
    Öğrenci görev puanını hesaplar.

    Physical Score sadece son iniş geometrisini ve hızını ölçer.
    Task Score ise görevin nihai puanıdır ve Total Reward değerini de dikkate alır.

    Mantık:
    - Fiziksel iniş doğru değilse task score fiziksel puanı aşamaz.
    - Fiziksel iniş doğru ama total_reward negatifse 100/100 verilmez.
    - total_reward negatif olduğunda puan ciddi biçimde düşürülür.
    - total_reward >= 0 ise fiziksel iniş puanı korunur.
    """

    if not physically_correct_landing:
        return physical_score

    if total_reward < MIN_TOTAL_REWARD_FOR_CORRECT_LANDING:
        penalized_score = physical_score + total_reward
        penalized_score = max(0, int(round(penalized_score)))
        return min(LOW_REWARD_SCORE_CAP, penalized_score)

    return physical_score


def evaluate_landing(observation, total_reward, terminated, truncated):
    """
    Episode sonucunu göreve göre değerlendirir.

    Başarı mantığı:
    - Çakılma / tek bacak / temassız bitiş: 0 puan ve başarısız.
    - İki bayrak arası, yavaş, dengeli ve düzgün açılı iniş: fiziksel olarak doğru iniş.
    - Correct Landing için fiziksel doğru iniş + total_reward >= 0 gerekir.
    - Excellent Landing için fiziksel doğru iniş + total_reward >= 200 gerekir.
    """

    if truncated:
        return "TIME LIMIT", False, 0, 0

    if not terminated:
        return "RUNNING", False, 0, 0

    state = extract_lander_state(observation)
    physical_score, checks = calculate_landing_score(state)

    if not checks["both_legs_contact"]:
        return "FAILED / CRASHED", False, 0, 0

    physically_correct_landing = (
        checks["in_landing_zone"]
        and checks["slow_vertical_landing"]
        and checks["slow_horizontal_motion"]
        and checks["stable_angle"]
        and checks["stable_angular_velocity"]
    )

    task_score = calculate_task_score(
        physical_score=physical_score,
        total_reward=total_reward,
        physically_correct_landing=physically_correct_landing,
    )

    reward_is_acceptable = total_reward >= MIN_TOTAL_REWARD_FOR_CORRECT_LANDING
    correct_landing = physically_correct_landing and reward_is_acceptable

    if physically_correct_landing and total_reward >= MIN_TOTAL_REWARD_FOR_EXCELLENT_LANDING:
        status = "EXCELLENT LANDING"
    elif correct_landing:
        status = "CORRECT LANDING"
    elif physically_correct_landing and not reward_is_acceptable:
        status = "LANDING OK / LOW REWARD"
    elif not checks["in_landing_zone"]:
        status = "OUTSIDE LANDING ZONE"
    elif not checks["slow_vertical_landing"]:
        status = "TOO FAST LANDING"
    elif not checks["slow_horizontal_motion"]:
        status = "TOO MUCH HORIZONTAL SPEED"
    elif not checks["stable_angle"]:
        status = "BAD ANGLE"
    elif not checks["stable_angular_velocity"]:
        status = "UNSTABLE ROTATION"
    elif physical_score >= 75:
        status = "ROUGH BUT ACCEPTABLE"
    else:
        status = "FAILED / LOW SCORE"

    return status, correct_landing, task_score, physical_score


def get_status_color(status, success):
    """
    Duruma göre panel yazı rengini belirler.
    """

    if success:
        return 80, 255, 120

    if status in {"FAILED / CRASHED", "TIME LIMIT", "FAILED / LOW SCORE"}:
        return 255, 90, 90

    if status in {
        "OUTSIDE LANDING ZONE",
        "TOO FAST LANDING",
        "TOO MUCH HORIZONTAL SPEED",
        "BAD ANGLE",
        "UNSTABLE ROTATION",
        "LANDING OK / LOW REWARD",
        "ROUGH BUT ACCEPTABLE",
    }:
        return 255, 190, 80

    return 255, 255, 255


def get_simulation_label(waiting_for_space, paused):
    """
    Simülasyon durumunu panel etiketi olarak döndürür.
    """

    if waiting_for_space:
        return "FINISHED"
    if paused:
        return "PAUSED"
    return "RUNNING"


def draw_info_panel(env, font, total_reward, task_score, physical_score, status,
                    success, waiting_for_space, paused):
    """
    Sağ üst köşeye sade sonuç paneli çizer.
    """

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

    result_text = "YES" if success else "NO"
    simulation_text = get_simulation_label(waiting_for_space, paused)

    lines = [
        f"Simulation: {simulation_text}",
        f"Task Score: {task_score}/100",
        f"Physical Score: {physical_score}/100",
        f"Total Reward: {total_reward:.2f}",
        f"Status: {status}",
        f"Correct Landing: {result_text}",
    ]

    y_offset = panel_y + 10
    status_color = get_status_color(status, success)

    for line in lines:
        color = status_color if line.startswith("Status") else (255, 255, 255)
        text_surface = font.render(line, True, color)
        screen.blit(text_surface, (panel_x + 10, y_offset))
        y_offset += 18

    if waiting_for_space:
        message = "SPACE: next simulation"
    elif paused:
        message = "SPACE: resume"
    else:
        message = "SPACE: pause"

    message_surface = font.render(message, True, (255, 220, 80))
    screen.blit(message_surface, (panel_x + 10, panel_y + PANEL_HEIGHT - 25))

    pygame.display.flip()


def reset_episode(env, episode):
    """
    Yeni episode başlatır ve sayaçları sıfırlar.
    """

    observation, info = env.reset()

    episode_data = {
        "episode": episode,
        "step_count": 0,
        "total_reward": 0.0,
        "last_reward": 0.0,
        "last_action": None,
        "status": "RUNNING",
        "success": False,
        "task_score": 0,
        "physical_score": 0,
        "waiting_for_space": False,
        "paused": False,
    }

    return observation, info, episode_data


def print_episode_result(episode_data):
    """
    Episode sonucunu terminale düzenli biçimde yazdırır.
    Panel sade tutulduğu için teknik sayaçlar terminalde bırakılır.
    """

    print("-" * 50)
    print(f"Episode: {episode_data['episode']}")
    print(f"Step: {episode_data['step_count']}")
    print(f"Total Reward: {episode_data['total_reward']:.2f}")
    print(f"Task Score: {episode_data['task_score']}/100")
    print(f"Physical Score: {episode_data['physical_score']}/100")
    print(f"Status: {episode_data['status']}")
    print(f"Correct Landing: {'YES' if episode_data['success'] else 'NO'}")
    print(f"Min Reward Required: {MIN_TOTAL_REWARD_FOR_CORRECT_LANDING:.0f}")
    print("Yeni simülasyon için SPACE tuşuna bas.")
    print("-" * 50)


def draw_current_panel(env, font, episode_data):
    """
    Aktif episode verilerinden paneli çizer.
    """

    draw_info_panel(
        env=env,
        font=font,
        total_reward=float(episode_data["total_reward"]),
        task_score=int(episode_data["task_score"]),
        physical_score=int(episode_data["physical_score"]),
        status=str(episode_data["status"]),
        success=bool(episode_data["success"]),
        waiting_for_space=bool(episode_data["waiting_for_space"]),
        paused=bool(episode_data["paused"]),
    )


def handle_keydown(event, env, observation, info, episode_data):
    """
    Klavye girdilerini yönetir.

    SPACE:
    - Episode bitmişse yeni episode başlatır.
    - Episode devam ediyorsa pause/resume yapar.
    """

    if event.key != pygame.K_SPACE:
        return observation, info, episode_data

    if episode_data["waiting_for_space"]:
        next_episode = int(episode_data["episode"]) + 1
        return reset_episode(env, next_episode)

    episode_data["paused"] = not bool(episode_data["paused"])
    return observation, info, episode_data


def main():
    pygame.init()

    env = gym.make("LunarLander-v3", render_mode="human")
    observation, info, episode_data = reset_episode(env, episode=1)
    font = pygame.font.SysFont("Arial", FONT_SIZE)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                observation, info, episode_data = handle_keydown(
                    event=event,
                    env=env,
                    observation=observation,
                    info=info,
                    episode_data=episode_data,
                )

        if not running:
            break

        if episode_data["waiting_for_space"] or episode_data["paused"]:
            draw_current_panel(env, font, episode_data)
            time.sleep(PAUSED_DELAY)
            continue

        raw_action = choose_action(observation)
        action = sanitize_action(raw_action)

        observation, reward, terminated, truncated, info = env.step(action)

        episode_data["last_action"] = action
        episode_data["last_reward"] = float(reward)
        episode_data["total_reward"] = float(episode_data["total_reward"]) + float(reward)
        episode_data["step_count"] = int(episode_data["step_count"]) + 1

        if terminated or truncated:
            status, success, task_score, physical_score = evaluate_landing(
                observation=observation,
                total_reward=float(episode_data["total_reward"]),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

            episode_data["status"] = status
            episode_data["success"] = success
            episode_data["task_score"] = task_score
            episode_data["physical_score"] = physical_score
            episode_data["waiting_for_space"] = True
            episode_data["paused"] = False

            print_episode_result(episode_data)
        else:
            episode_data["status"] = "RUNNING"
            episode_data["success"] = False
            episode_data["task_score"] = 0
            episode_data["physical_score"] = 0

        draw_current_panel(env, font, episode_data)
        time.sleep(FPS_DELAY)

    env.close()
    pygame.quit()
    print("Program kapatıldı.")


if __name__ == "__main__":
    main()