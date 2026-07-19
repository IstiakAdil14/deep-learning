"""
Risk Engine
Version 7.1

More realistic SmartGrid risk model.

Uses weighted continuous scoring instead of
hard thresholds.

Produces balanced Low / Medium / High classes.
"""

import random

from config import SEED

random.seed(SEED)


class RiskSimulator:

    def __init__(self):

        self.previous_score = 10

    # --------------------------------------------------

    def next_risk(
        self,
        weather,
        demand,
        generation,
        transformer,
        hour=None
    ):

        score = 0

        load = transformer["transformer_load"]
        reserve = generation - demand

        temp = weather["temperature"]
        rain = weather["rainfall"]
        humidity = weather["humidity"]
        wind = weather["wind_speed"]
        state = weather["weather_state"]

        # --------------------------------------------------
        # Transformer loading (largest factor)
        # --------------------------------------------------

        score += max(0, (load - 55) * 1.5)

        # --------------------------------------------------
        # Reserve margin
        # --------------------------------------------------

        if reserve < 20:
            score += (20 - reserve) * 1.5

        # --------------------------------------------------
        # Temperature
        # --------------------------------------------------

        score += max(0, (temp - 25) * 2.0)

        # --------------------------------------------------
        # Rainfall
        # --------------------------------------------------

        score += min(rain * 0.8, 20)

        # --------------------------------------------------
        # Wind
        # --------------------------------------------------

        score += max(0, (wind - 5) * 1.5)

        # --------------------------------------------------
        # Humidity
        # --------------------------------------------------

        score += max(0, (humidity - 80) * 0.3)

        # --------------------------------------------------
        # Weather bonus
        # --------------------------------------------------

        if state == "Rainy":
            score += 8

        elif state == "Cloudy":
            score += 2

        # --------------------------------------------------
        # Time of day — evening peak
        # --------------------------------------------------

        if hour is not None and 18 <= hour <= 22:
            score += 5

        # --------------------------------------------------
        # Random uncertainty
        # --------------------------------------------------

        score += random.uniform(-8, 8)

        # --------------------------------------------------
        # Temporal smoothing
        # --------------------------------------------------

        score = (
            self.previous_score * 0.10 +
            score * 0.90
        )

        self.previous_score = score

        score = max(0, min(100, score))

        # --------------------------------------------------
        # Risk level
        # --------------------------------------------------

        if score < 30:
            level = "Low"

        elif score < 60:
            level = "Medium"

        else:
            level = "High"

        return {

            "risk_score": round(score, 2),

            "risk_level": level

        }


risk = RiskSimulator()