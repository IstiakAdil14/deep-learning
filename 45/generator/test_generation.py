"""
Test Generation Engine
"""

from timeline import generate_timeline
from weather import WeatherSimulator
from demand import DemandSimulator
from generation import GenerationSimulator


timeline = generate_timeline()

weather = WeatherSimulator()
demand = DemandSimulator()
generation = GenerationSimulator()


print("=" * 120)
print("SMARTGRID GENERATION SIMULATOR")
print("=" * 120)

print(
    f"{'Datetime':20}"
    f"{'State':>10}"
    f"{'Temp':>10}"
    f"{'Rain':>10}"
    f"{'Demand':>14}"
    f"{'Generation':>14}"
)

print("-" * 120)


for dt in timeline[:60]:

    # Weather
    w = weather.next_weather(dt)

    # Demand
    d = demand.next_demand(dt, w)

    # Generation
    g = generation.next_generation(
        demand=d,
        hour=dt.hour,
        weather_state=w["weather_state"]
    )

    print(
        f"{str(dt):20}"
        f"{w['weather_state']:>10}"
        f"{w['temperature']:10.2f}"
        f"{w['rainfall']:10.2f}"
        f"{d:14.2f}"
        f"{g:14.2f}"
    )