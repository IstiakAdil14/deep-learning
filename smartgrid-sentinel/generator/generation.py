"""
Generation Engine
Version 3

Simulates electricity generation based on
demand, hour and weather.
"""

import random

from config import SEED

random.seed(SEED)


class GenerationSimulator:

    def __init__(self):

        self.current_generation = 180

        self.maintenance = False
        self.maintenance_hours = 0

    # --------------------------------------------------

    def next_generation(self, demand, hour, weather_state):

        # ----------------------------------------
        # Reserve Margin
        # ----------------------------------------

        reserve = 0.08

        # Evening peak
        if 18 <= hour <= 22:
            reserve = 0.12

        # Midnight
        elif 0 <= hour <= 5:
            reserve = 0.06

        # Rainy weather requires more reserve
        if weather_state == "Rainy":
            reserve += 0.03

        target = demand * (1 + reserve)

        # ----------------------------------------
        # Rain effect
        # ----------------------------------------

        if weather_state == "Rainy":
            target -= random.uniform(3, 8)

        # ----------------------------------------
        # Plant Maintenance
        # ----------------------------------------

        if self.maintenance_hours == 0:

            if random.random() < 0.015:

                self.maintenance = True
                self.maintenance_hours = random.randint(6, 18)

        if self.maintenance:

            target -= random.uniform(5, 12)

            self.maintenance_hours -= 1

            if self.maintenance_hours <= 0:
                self.maintenance = False

        # ----------------------------------------
        # Noise
        # ----------------------------------------

        target += random.uniform(-2, 2)

        # ----------------------------------------
        # Smooth response
        # ----------------------------------------

        self.current_generation += (
            target - self.current_generation
        ) * 0.60

        # ----------------------------------------
        # Limits
        # ----------------------------------------

        self.current_generation = max(
            demand * 0.95,
            min(300, self.current_generation)
        )

        return round(self.current_generation, 2)


# Module-level instance
generation = GenerationSimulator()