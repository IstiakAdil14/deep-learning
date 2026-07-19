from timeline import generate_timeline

from weather import WeatherSimulator
from demand import DemandSimulator
from generation import GenerationSimulator
import transformer

timeline = generate_timeline()

weather_engine = WeatherSimulator()
demand_engine = DemandSimulator()
generation_engine = GenerationSimulator()

print("=" * 120)
print("SMARTGRID TRANSFORMER SIMULATOR")
print("=" * 120)

print(
    f"{'Datetime':20}"
    f"{'State':>12}"
    f"{'Demand':>12}"
    f"{'Generation':>14}"
    f"{'Transformer %':>18}"
)

print("-" * 120)

for dt in timeline[:60]:

    # Weather
    w = weather_engine.next_weather(dt)

    # Demand
    d = demand_engine.next_demand(dt, w)

    # Generation
    g = generation_engine.next_generation(
        demand=d,
        hour=dt.hour,
        weather_state=w["weather_state"]
    )

    # Transformer
    t = transformer.next_transformer(d, g)

    print(
        f"{dt.strftime('%Y-%m-%d %H:%M:%S'):20}"
        f"{w['weather_state']:>12}"
        f"{d:12.2f}"
        f"{g:14.2f}"
        f"{t['transformer_load']:18.2f}"
    )