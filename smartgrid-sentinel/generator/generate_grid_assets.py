import pandas as pd
import random

random.seed(42)

upazilas = {
    "Sylhet": [
        "Balaganj", "Beanibazar", "Bishwanath", "Companiganj",
        "Dakshin Surma", "Fenchuganj", "Golapganj", "Gowainghat",
        "Jaintiapur", "Kanaighat", "Osmaninagar",
        "Sylhet Sadar", "Zakiganj"
    ],

    "Habiganj": [
        "Ajmiriganj", "Bahubal", "Baniachong", "Chunarughat",
        "Habiganj Sadar", "Lakhai", "Madhabpur",
        "Nabiganj", "Shayestaganj"
    ],

    "Moulvibazar": [
        "Barlekha", "Juri", "Kamalganj",
        "Kulaura", "Moulvibazar Sadar",
        "Rajnagar", "Sreemangal"
    ],

    "Sunamganj": [
        "Bishwamvarpur", "Chhatak",
        "Dakshin Sunamganj", "Derai",
        "Dharamapasha", "Dowarabazar",
        "Jagannathpur", "Jamalganj",
        "Sulla", "Sunamganj Sadar"
    ]
}

rows = []

substation_counter = 1
feeder_counter = 1

for district, up_list in upazilas.items():

    for upazila in up_list:

        area_type = random.choice(["Urban", "Rural"])

        transformer_capacity = random.choice(
            [200, 250, 300, 350, 400]
        )

        transformer_age = random.randint(3, 15)

        population_density = random.randint(
            500, 2000
        )

        industrial_ratio = round(
            random.uniform(0.08, 0.45), 2
        )

        outage_history = random.randint(0, 8)

        maintenance_due = random.choice(
            ["Yes", "No"]
        )

        base_demand = random.randint(150, 220)

        demand_night = base_demand - random.randint(10, 25)
        demand_morning = base_demand
        demand_afternoon = base_demand + random.randint(10, 20)
        demand_evening = base_demand + random.randint(20, 35)

        renewable_night = random.randint(160, 190)
        renewable_morning = random.randint(180, 220)
        renewable_afternoon = random.randint(200, 240)
        renewable_evening = random.randint(170, 210)

        transformer_load_night = random.randint(55, 70)
        transformer_load_morning = random.randint(60, 75)
        transformer_load_afternoon = random.randint(70, 85)
        transformer_load_evening = random.randint(75, 92)

        rows.append({
            "district": district,
            "upazila": upazila,
            "area_type": area_type,

            "substation_id": f"SS_{substation_counter:02d}",
            "feeder_id": f"FDR_{feeder_counter:02d}",

            "transformer_age": transformer_age,
            "transformer_capacity": transformer_capacity,

            "outage_history": outage_history,
            "maintenance_due": maintenance_due,

            "population_density": population_density,
            "industrial_load_ratio": industrial_ratio,

            "demand_night": demand_night,
            "demand_morning": demand_morning,
            "demand_afternoon": demand_afternoon,
            "demand_evening": demand_evening,

            "renewable_night": renewable_night,
            "renewable_morning": renewable_morning,
            "renewable_afternoon": renewable_afternoon,
            "renewable_evening": renewable_evening,

            "transformer_load_night": transformer_load_night,
            "transformer_load_morning": transformer_load_morning,
            "transformer_load_afternoon": transformer_load_afternoon,
            "transformer_load_evening": transformer_load_evening
        })

        substation_counter += 1
        feeder_counter += 1

df = pd.DataFrame(rows)

df.to_csv(
    "grid_assets.csv",
    index=False
)

print("Created grid_assets.csv")
print("Rows:", len(df))
print(df.head())