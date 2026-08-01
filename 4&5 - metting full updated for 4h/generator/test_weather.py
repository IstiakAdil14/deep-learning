from timeline import generate_timeline
from weather import WeatherSimulator

timeline = generate_timeline()

weather = WeatherSimulator()

print("=" * 90)
print("SMARTGRID WEATHER SIMULATOR")
print("=" * 90)

print(
    f"{'Datetime':20}"
    f"{'State':12}"
    f"{'Temp':>8}"
    f"{'Hum':>8}"
    f"{'Rain':>10}"
    f"{'Wind':>10}"
)

print("-" * 90)

for dt in timeline[:60]:

    w = weather.next_weather(dt)

    print(
        f"{str(dt):20}"
        f"{w['weather_state']:12}"
        f"{w['temperature']:8.2f}"
        f"{w['humidity']:8.2f}"
        f"{w['rainfall']:10.2f}"
        f"{w['wind_speed']:10.2f}"
    )