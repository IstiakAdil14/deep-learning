"""
Weather Engine
Version 3

State-based weather simulator for SmartGrid Sentinel.
Generates realistic continuous weather for every 2-hour timestep.
"""

import math
import random

from config import SEED

random.seed(SEED)


class WeatherSimulator:

    def __init__(self):

        self.current_state = "Cloudy"

        self.days_left = 2

        self.temperature = 26.0
        self.humidity = 82.0
        self.wind_speed = 5.0
        self.rainfall = 0.0

        self.last_day = None

    # -------------------------------------------------

    def update_weather_state(self, dt):

        today = dt.date()

        if today != self.last_day:

            self.last_day = today

            self.days_left -= 1

            if self.days_left <= 0:

                self.current_state = random.choices(
                    ["Sunny", "Cloudy", "Rainy"],
                    weights=[45, 35, 20],
                    k=1
                )[0]

                if self.current_state == "Sunny":
                    self.days_left = random.randint(2, 5)

                elif self.current_state == "Cloudy":
                    self.days_left = random.randint(1, 3)

                else:
                    self.days_left = random.randint(1, 2)

    # -------------------------------------------------

    def update_temperature(self, dt):

        hour = dt.hour

        daily_curve = math.sin(
            ((hour - 8) / 24) * 2 * math.pi
        )

        base = 27

        if self.current_state == "Sunny":
            base += 3

        elif self.current_state == "Rainy":
            base -= 3

        target = base + 6 * daily_curve

        self.temperature += (target - self.temperature) * 0.35

        self.temperature += random.uniform(-0.3, 0.3)

        self.temperature = max(18, min(38, self.temperature))

    # -------------------------------------------------

    def update_humidity(self):

        target = 100 - (self.temperature - 20) * 3

        if self.current_state == "Rainy":
            target += 10

        self.humidity += (target - self.humidity) * 0.25

        self.humidity += random.uniform(-1.5, 1.5)

        self.humidity = max(45, min(100, self.humidity))

    # -------------------------------------------------

    def update_wind(self):

        target = 4

        if self.current_state == "Sunny":
            target = 3

        elif self.current_state == "Cloudy":
            target = 5

        elif self.current_state == "Rainy":
            target = 9

        self.wind_speed += (target - self.wind_speed) * 0.25

        self.wind_speed += random.uniform(-0.5, 0.5)

        self.wind_speed = max(0.5, min(20, self.wind_speed))

    # -------------------------------------------------

    def update_rainfall(self):

        if self.current_state == "Sunny":

            self.rainfall *= 0.20

        elif self.current_state == "Cloudy":

            if random.random() < 0.10:
                self.rainfall = random.uniform(0.2, 2)

            else:
                self.rainfall *= 0.60

        elif self.current_state == "Rainy":

            if self.rainfall < 1:

                self.rainfall = random.uniform(3, 10)

            else:

                self.rainfall += random.uniform(-2, 3)

        self.rainfall = max(0, min(30, self.rainfall))

    # -------------------------------------------------

    def next_weather(self, dt):

        self.update_weather_state(dt)

        self.update_temperature(dt)

        self.update_humidity()

        self.update_wind()

        self.update_rainfall()

        return {

            "weather_state": self.current_state,

            "temperature": round(self.temperature, 2),

            "humidity": round(self.humidity, 2),

            "rainfall": round(self.rainfall, 2),

            "wind_speed": round(self.wind_speed, 2)

        }


# Module-level instance
weather = WeatherSimulator()