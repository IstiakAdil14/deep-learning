"""
Generate SmartGrid Dataset
Version 2 - Added Upazila support

Creates:
smartgrid_risk_dataset.csv
"""

import pandas as pd

from timeline import generate_timeline
from weather import weather
from demand import demand
from generation import generation
from transformer import next_transformer
from risk import risk
from grid import get_grid_features


def generate_dataset():

    rows = []

    timeline = generate_timeline()

    for dt in timeline:

        # Weather
        w = weather.next_weather(dt)

        # Demand
        d = demand.next_demand(
            dt,
            w
        )

        # Renewable generation
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
            dt.hour
        )

        # Grid features
        grid = get_grid_features()

        rows.append({

            "datetime": dt,

            # Date features (will be dropped in preprocessing)
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "weekday": dt.weekday(),

            # Weather features
            "temperature": w["temperature"],
            "humidity": w["humidity"],
            "rainfall": w["rainfall"],
            "wind_speed": w["wind_speed"],
            "weather_state": w["weather_state"],

            # Grid load features
            "electricity_demand": round(d, 2),
            "renewable_generation": round(g, 2),

            "transformer_load": round(
                t["transformer_load"], 2
            ),

            "risk_score": r["risk_score"],
            "risk_level": r["risk_level"],

            # Geographic features
            "district": grid["district"],
            "upazila": grid["upazila"],
            "area_type": grid["area_type"],
            "substation_id": grid["substation_id"],
            "feeder_id": grid["feeder_id"],

            # Asset features
            "transformer_age": grid["transformer_age"],
            "transformer_capacity": grid["transformer_capacity"],
            "outage_history": grid["outage_history"],
            "maintenance_due": grid["maintenance_due"],
            "population_density": grid["population_density"],
            "industrial_load_ratio": grid["industrial_load_ratio"]

        })

    df = pd.DataFrame(rows)

    df.to_csv(
        "smartgrid_risk_dataset.csv",
        index=False
    )

    print("=" * 60)
    print("Dataset Generated Successfully")
    print("=" * 60)
    print(df.head())
    print()
    print("Shape:", df.shape)
    print()
    print(df["risk_level"].value_counts())


if __name__ == "__main__":

    generate_dataset()