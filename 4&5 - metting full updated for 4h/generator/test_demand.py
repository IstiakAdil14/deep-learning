from timeline import generate_timeline
from weather import WeatherSimulator
from demand import DemandSimulator

timeline = generate_timeline()

weather = WeatherSimulator()
demand = DemandSimulator()

print("=" * 105)
print("SMARTGRID DEMAND SIMULATOR")
print("=" * 105)

print(
    f"{'Datetime':20}"
    f"{'State':12}"
    f"{'Temp':>8}"
    f"{'Rain':>8}"
    f"{'Demand(MW)':>15}"
)

print("-" * 105)

for dt in timeline[:60]:

    w = weather.next_weather(dt)

    d = demand.next_demand(dt, w)

    print(
        f"{str(dt):20}"
        f"{w['weather_state']:12}"
        f"{w['temperature']:8.2f}"
        f"{w['rainfall']:8.2f}"
        f"{d:15.2f}"
    )