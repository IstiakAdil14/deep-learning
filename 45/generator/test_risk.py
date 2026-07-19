from timeline import generate_timeline

from weather import WeatherSimulator
from demand import DemandSimulator
from generation import GenerationSimulator
from transformer import next_transformer
from risk import RiskSimulator


# ----------------------------------------------------
# Create simulator objects
# ----------------------------------------------------

weather = WeatherSimulator()
demand = DemandSimulator()
generation = GenerationSimulator()
risk = RiskSimulator()

timeline = generate_timeline()

print("=" * 150)
print("SMARTGRID RISK SIMULATOR")
print("=" * 150)

print(
    f"{'Datetime':20}"
    f"{'State':>10}"
    f"{'Demand':>12}"
    f"{'Generation':>14}"
    f"{'Load %':>12}"
    f"{'Risk Score':>14}"
    f"{'Risk':>12}"
)

print("-" * 150)

for dt in timeline[:60]:

    # Weather
    w = weather.next_weather(dt)

    # Demand
    d = demand.next_demand(dt, w)

    # Generation
    g = generation.next_generation(
        d,
        dt.hour,
        w["weather_state"]
    )

    # Transformer
    t = next_transformer(
        d,
        g
    )

    # Risk
    r = risk.next_risk(
        w,
        d,
        g,
        t,
        hour=dt.hour
    )

    print(
        f"{dt.strftime('%Y-%m-%d %H:%M:%S'):20}"
        f"{w['weather_state']:>10}"
        f"{d:12.2f}"
        f"{g:14.2f}"
        f"{t['transformer_load']:12.2f}"
        f"{r['risk_score']:14.2f}"
        f"{r['risk_level']:>12}"
    )