"""Analyze risk score distribution over full timeline."""
from timeline import generate_timeline
from weather import WeatherSimulator
from demand import DemandSimulator
from generation import GenerationSimulator
from transformer import next_transformer
from risk import RiskSimulator

weather = WeatherSimulator()
demand = DemandSimulator()
generation = GenerationSimulator()
risk = RiskSimulator()

timeline = generate_timeline()

low = 0
medium = 0
high = 0
scores = []

for dt in timeline:
    w = weather.next_weather(dt)
    d = demand.next_demand(dt, w)
    g = generation.next_generation(d, dt.hour, w["weather_state"])
    t = next_transformer(d, g)
    r = risk.next_risk(w, d, g, t, hour=dt.hour)
    
    scores.append(r["risk_score"])
    
    if r["risk_level"] == "Low":
        low += 1
    elif r["risk_level"] == "Medium":
        medium += 1
    else:
        high += 1

total = low + medium + high
print(f"Total records: {total}")
print(f"Low:    {low:4d} ({low/total*100:.1f}%)")
print(f"Medium: {medium:4d} ({medium/total*100:.1f}%)")
print(f"High:   {high:4d} ({high/total*100:.1f}%)")
print(f"\nScore stats: min={min(scores):.1f}, max={max(scores):.1f}, mean={sum(scores)/len(scores):.1f}")