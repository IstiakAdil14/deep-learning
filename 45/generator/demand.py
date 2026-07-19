"""
Demand Engine
Version 2

Creates realistic electricity demand
with temporal continuity.
"""

import math
import random

from config import SEED

random.seed(SEED)


class DemandSimulator:

    def __init__(self):

        self.current_demand = 170.0
        self.previous_target = 170.0

    # ---------------------------------------------------------
    # Base Daily Curve
    # ---------------------------------------------------------

    def hour_effect(self, hour):

        base = 145

        morning_peak = 55 * math.exp(-((hour - 9) ** 2) / 10)

        afternoon_load = 20 * math.exp(-((hour - 14) ** 2) / 18)

        evening_peak = 90 * math.exp(-((hour - 20) ** 2) / 7)

        return base + morning_peak + afternoon_load + evening_peak

    # ---------------------------------------------------------
    # Weather Effect
    # ---------------------------------------------------------

    def weather_effect(self, weather):

        effect = 0

        temp = weather["temperature"]
        humidity = weather["humidity"]
        rainfall = weather["rainfall"]

        if temp > 30:
            effect += (temp - 30) * 8

        elif temp < 18:
            effect += (18 - temp) * 3

        if humidity > 90:
            effect += 6

        if rainfall > 15:
            effect += 20

        elif rainfall > 8:
            effect += 12

        elif rainfall > 3:
            effect += 6

        return effect

    # ---------------------------------------------------------
    # Weekday Effect
    # ---------------------------------------------------------

    def weekday_effect(self, dt):

        factors = {

            0: 1.02,   # Monday
            1: 1.03,
            2: 1.01,
            3: 1.00,
            4: 0.98,
            5: 0.93,   # Saturday
            6: 0.90    # Sunday

        }

        return factors[dt.weekday()]

    # ---------------------------------------------------------
    # Smooth Transition
    # ---------------------------------------------------------

    def smooth(self, target):

        momentum = self.current_demand - self.previous_target

        self.previous_target = self.current_demand

        self.current_demand += (

            (target - self.current_demand) * 0.35
            + momentum * 0.20
            + random.uniform(-2, 2)

        )

        self.current_demand = max(
            120,
            min(280, self.current_demand)
        )

        return self.current_demand

    # ---------------------------------------------------------
    # Main
    # ---------------------------------------------------------

    def next_demand(self, dt, weather):

        target = self.hour_effect(dt.hour)

        target += self.weather_effect(weather)

        target *= self.weekday_effect(dt)

        demand = self.smooth(target)

        return round(demand, 2)


# Module-level instance
demand = DemandSimulator()