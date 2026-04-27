import gymnasium as gym


class DirectLandingRewardWrapper(gym.Wrapper):
    """
    Doğrudan ve kaymadan inişi öğretmek için reward wrapper.

    Ana hedefler:
    - Havada gereksiz kalmayı cezalandırmak
    - Time limit'e kadar inmeme davranışını ağır cezalandırmak
    - Yere temas etmeyi kötü göstermemek
    - İki bacakla dengeli inişi ödüllendirmek
    - Temas sonrası kaymayı episode sonunda değerlendirmek
    """

    def __init__(self, env):
        super().__init__(env)
        self.air_steps = 0
        self.first_contact_x = None
        self.max_post_contact_drift = 0.0

    def reset(self, **kwargs):
        self.air_steps = 0
        self.first_contact_x = None
        self.max_post_contact_drift = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        x_pos = float(observation[0])
        y_pos = float(observation[1])
        x_vel = float(observation[2])
        y_vel = float(observation[3])
        angle = float(observation[4])
        angular_vel = float(observation[5])
        left_leg = float(observation[6])
        right_leg = float(observation[7])

        left_contact = left_leg > 0.5
        right_contact = right_leg > 0.5
        any_contact = left_contact or right_contact
        both_contact = left_contact and right_contact

        extra_reward = 0.0

        # ----------------------------------------------------
        # 1. Havada gereksiz kalmayı cezalandır
        # ----------------------------------------------------
        if not any_contact:
            self.air_steps += 1

            # Her havada kalınan step için küçük ceza
            extra_reward -= 0.03

            # Episode uzadıkça havada kalma daha pahalı olsun
            if self.air_steps > 250:
                extra_reward -= 0.08
            if self.air_steps > 500:
                extra_reward -= 0.20
            if self.air_steps > 750:
                extra_reward -= 0.50

            # Yere yakınken hâlâ havada kalıyorsa daha fazla ceza
            if y_pos < 0.50:
                extra_reward -= 0.10
            if y_pos < 0.30:
                extra_reward -= 0.25

        # ----------------------------------------------------
        # 2. Motor kullanımını hafif cezalandır
        # ----------------------------------------------------
        # Çok sert yapmıyoruz; yoksa model motor kullanmaktan korkar.
        if int(action) == 2:
            extra_reward -= 0.01      # ana motor
        elif int(action) in (1, 3):
            extra_reward -= 0.005     # yan motorlar

        # ----------------------------------------------------
        # 3. Yere yaklaşırken inişe hazırlanmayı teşvik et
        # ----------------------------------------------------
        if y_pos < 0.45 and not any_contact:
            extra_reward -= 1.5 * abs(x_vel)
            extra_reward -= 0.8 * abs(angle)
            extra_reward -= 0.4 * abs(angular_vel)

            # Yere yakınken çok hızlı düşüyorsa ceza
            if y_vel < -0.50:
                extra_reward -= 2.0 * abs(y_vel)

        # ----------------------------------------------------
        # 4. İlk temas noktasını kaydet
        # ----------------------------------------------------
        if any_contact and self.first_contact_x is None:
            self.first_contact_x = x_pos

        # ----------------------------------------------------
        # 5. Temas sonrası kaymayı sadece takip et
        # ----------------------------------------------------
        # Burada her frame ceza vermiyoruz.
        # Çünkü sürekli temas cezası modelin yere inmekten korkmasına sebep olur.
        if any_contact and self.first_contact_x is not None:
            drift = abs(x_pos - self.first_contact_x)
            self.max_post_contact_drift = max(self.max_post_contact_drift, drift)

        # ----------------------------------------------------
        # 6. Episode sonunda asıl iniş değerlendirmesi
        # ----------------------------------------------------
        if terminated or truncated:
            in_landing_zone = abs(x_pos) <= 0.20

            stable_landing = (
                both_contact
                and in_landing_zone
                and abs(x_vel) <= 0.20
                and abs(y_vel) <= 0.35
                and abs(angle) <= 0.20
                and abs(angular_vel) <= 1.00
            )

            direct_landing = (
                stable_landing
                and self.max_post_contact_drift <= 0.05
            )

            if truncated:
                # Time limit'e kadar havada kalmak en kötü davranışlardan biri.
                extra_reward -= 1000.0

            elif direct_landing:
                # Hem düzgün indi hem de kaymadı.
                extra_reward += 300.0

            elif stable_landing:
                # Düzgün indi ama biraz kaydı.
                extra_reward += 180.0
                extra_reward -= 100.0 * self.max_post_contact_drift

            elif both_contact:
                # İki bacak temas var ama iniş kalitesi iyi değil.
                extra_reward += 60.0
                extra_reward -= 80.0 * self.max_post_contact_drift

            else:
                # Çakılma, tek bacak, kötü bitiş.
                extra_reward -= 150.0

        shaped_reward = reward + extra_reward

        info["direct_landing_extra_reward"] = extra_reward
        info["max_post_contact_drift"] = self.max_post_contact_drift
        info["air_steps"] = self.air_steps

        return observation, shaped_reward, terminated, truncated, info