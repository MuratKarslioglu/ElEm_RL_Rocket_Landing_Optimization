import gymnasium as gym


class DirectLandingRewardWrapper(gym.Wrapper):
    """
    Direct landing reward wrapper.

    Hedef:
    - Yere indikten sonra motor basıp kaymayı engellemek
    - İlk temas anında düşük yatay hızla oturmayı öğretmek
    - Kayarak iyi reward almayı engellemek
    - Sadece gerçekten direkt ve dengeli inişe büyük ödül vermek
    """

    def __init__(self, env):
        super().__init__(env)

        self.air_steps = 0
        self.first_contact_x = None
        self.max_post_contact_drift = 0.0
        self.first_contact_done = False
        self.landed_lock = False

    def reset(self, **kwargs):
        self.air_steps = 0
        self.first_contact_x = None
        self.max_post_contact_drift = 0.0
        self.first_contact_done = False
        self.landed_lock = False

        return self.env.reset(**kwargs)

    def step(self, action):
        action = int(action)

        # İki bacakla indikten sonra artık motor yok.
        if self.landed_lock:
            action = 0

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
        # 1. Havada gereksiz kalma cezası
        # ----------------------------------------------------
        if not any_contact:
            self.air_steps += 1

            extra_reward -= 0.02

            if self.air_steps > 400:
                extra_reward -= 0.05
            if self.air_steps > 700:
                extra_reward -= 0.15

        # ----------------------------------------------------
        # 2. Motor kullanımı hafif ceza
        # ----------------------------------------------------
        # Çok sert değil; iniş için motor gerekli.
        if action == 2:
            extra_reward -= 0.005
        elif action in (1, 3):
            extra_reward -= 0.003

        # ----------------------------------------------------
        # 3. Havadayken iniş alanına yaklaşmayı teşvik et
        # ----------------------------------------------------
        if not any_contact:
            # Merkezden uzaksa küçük ceza
            extra_reward -= 0.10 * abs(x_pos)

            # Yere yakınken yatay hız özellikle önemli
            if y_pos < 0.50:
                extra_reward -= 1.5 * abs(x_vel)
                extra_reward -= 0.6 * abs(angle)
                extra_reward -= 0.3 * abs(angular_vel)

                if y_vel < -0.50:
                    extra_reward -= 1.5 * abs(y_vel)

        # ----------------------------------------------------
        # 4. İlk temas anını değerlendir
        # ----------------------------------------------------
        if any_contact and not self.first_contact_done:
            self.first_contact_done = True
            self.first_contact_x = x_pos

            extra_reward += 5.0

            if both_contact:
                extra_reward += 10.0
            else:
                extra_reward -= 25.0

            if abs(x_vel) < 0.04:
                extra_reward += 45.0
            elif abs(x_vel) < 0.08:
                extra_reward += 25.0
            elif abs(x_vel) < 0.14:
                extra_reward += 5.0
            else:
                extra_reward -= 160.0 * abs(x_vel)

            if abs(y_vel) < 0.25:
                extra_reward += 20.0
            elif abs(y_vel) < 0.35:
                extra_reward += 8.0
            else:
                extra_reward -= 70.0 * abs(y_vel)

            if abs(angle) < 0.12:
                extra_reward += 20.0
            elif abs(angle) < 0.20:
                extra_reward += 8.0
            else:
                extra_reward -= 50.0 * abs(angle)

            if abs(x_pos) <= 0.20:
                extra_reward += 20.0
            else:
                extra_reward -= 120.0 * abs(x_pos)
        # ----------------------------------------------------
        # 5. Temas sonrası drift takip
        # ----------------------------------------------------
        if any_contact and self.first_contact_x is not None:
            drift = abs(x_pos - self.first_contact_x)
            self.max_post_contact_drift = max(self.max_post_contact_drift, drift)

        # İki bacak temas ettiyse sonraki adımlarda motorları kapat.
        if both_contact:
            self.landed_lock = True

        # ----------------------------------------------------
        # 6. Episode sonu değerlendirme
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

            very_direct_landing = (
                stable_landing
                and self.max_post_contact_drift <= 0.03
            )

            direct_landing = (
                stable_landing
                and self.max_post_contact_drift <= 0.05
            )

            almost_direct = (
                stable_landing
                and self.max_post_contact_drift <= 0.10
            )

            if truncated:
                extra_reward -= 500.0

            elif very_direct_landing:
                extra_reward += 220.0

            elif direct_landing:
                extra_reward += 150.0

            elif almost_direct:
                extra_reward += 20.0
                extra_reward -= 250.0 * self.max_post_contact_drift

            elif stable_landing:
                extra_reward -= 350.0 * self.max_post_contact_drift

            elif both_contact:
                extra_reward -= 100.0
                extra_reward -= 250.0 * self.max_post_contact_drift

            else:
                extra_reward -= 150.0
        shaped_reward = reward + extra_reward

        info["direct_landing_extra_reward"] = extra_reward
        info["max_post_contact_drift"] = self.max_post_contact_drift
        info["air_steps"] = self.air_steps
        info["landed_lock"] = self.landed_lock

        return observation, shaped_reward, terminated, truncated, info