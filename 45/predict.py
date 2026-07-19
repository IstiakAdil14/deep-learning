import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================
# Load trained model
# ==========================
model = joblib.load("models/logistic_regression.pkl")
scaler = joblib.load("models/scaler.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

# Get the correct feature order from the scaler
feature_order = list(scaler.feature_names_in_)

# ==========================
# User Inputs
# ==========================

print("="*60)
print(" SMART GRID RISK PREDICTION (NEXT 4 HOURS)")
print("="*60)

temperature = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
rainfall = float(input("Rainfall (mm): "))
wind_speed = float(input("Wind Speed (m/s): "))

electricity_demand = float(input("Electricity Demand (MW): "))
renewable_generation = float(input("Renewable Generation (MW): "))
transformer_load = float(input("Transformer Load (%): "))

transformer_age = int(input("Transformer Age (years): "))
transformer_capacity = float(input("Transformer Capacity (kVA): "))
outage_history = int(input("Previous Outages: "))
population_density = float(input("Population Density: "))
industrial_load_ratio = float(input("Industrial Load Ratio (0-1): "))

hour = int(input("Current Hour (0-23): "))
weekday = int(input("Weekday (0=Mon ... 6=Sun): "))

district = input("District: ")
upazila = input("Upazila: ")
area_type = input("Area Type (Urban/Rural): ")
substation_id = input("Substation ID: ")
feeder_id = input("Feeder ID: ")
maintenance_due = input("Maintenance Due (Yes/No): ")
weather_state = input("Weather (Sunny/Cloudy/Rainy): ")

# =====================================================
# Label Encoding
# (Must match preprocessing notebook)
# =====================================================

weather_map = {
    "Sunny": 0,
    "Cloudy": 1,
    "Rainy": 2
}

area_map = {
    "Urban": 0,
    "Rural": 1
}

maintenance_map = {
    "No": 0,
    "Yes": 1
}

district_map = {
    "Sylhet": 0,
    "Habiganj": 1,
    "Moulvibazar": 2,
    "Sunamganj": 3
}

upazila_map = {
    # Sylhet upazilas
    "Balaganj": 0, "Beanibazar": 1, "Bishwanath": 2, "Companiganj": 3,
    "Dakshin Surma": 4, "Fenchuganj": 5, "Golapganj": 6, "Gowainghat": 7,
    "Jaintiapur": 8, "Kanaighat": 9, "Osmani Nagar": 10, "Sylhet Sadar": 11,
    "Zakiganj": 12,
    # Habiganj upazilas
    "Ajmiriganj": 13, "Bahubal": 14, "Baniachang": 15, "Chunarughat": 16,
    "Habiganj Sadar": 17, "Lakhai": 18, "Madhabpur": 19, "Nabiganj": 20,
    "Shaistaganj": 21,
    # Moulvibazar upazilas
    "Barlekha": 22, "Juri": 23, "Kamalganj": 24, "Kulaura": 25,
    "Moulvibazar Sadar": 26, "Rajnagar": 27, "Sreemangal": 28,
    # Sunamganj upazilas
    "Bishwamvarpur": 29, "Chhatak": 30, "Dakshin Sunamganj": 31, "Derai": 32,
    "Dharmapasha": 33, "Dowarabazar": 34, "Jagannathpur": 35, "Jamalganj": 36,
    "Sullah": 37, "Sunamganj Sadar": 38, "Tahirpur": 39, "Shantiganj": 40
}

# Encode categorical inputs
weather_state = weather_map.get(weather_state, 0)
area_type = area_map.get(area_type, 0)
maintenance_due = maintenance_map.get(maintenance_due, 0)
district = district_map.get(district, 0)
upazila = upazila_map.get(upazila, 0)

# Convert IDs into integers
substation_id = int(substation_id.replace("SS_", ""))
feeder_id = int(feeder_id.replace("FDR_", ""))

# ==========================
# Feature Vector
# ==========================

# Create input data with correct feature order matching the training data
data = {
    'hour': hour,
    'weekday': weekday,
    'temperature': temperature,
    'humidity': humidity,
    'rainfall': rainfall,
    'wind_speed': wind_speed,
    'weather_state': weather_state,
    'electricity_demand': electricity_demand,
    'renewable_generation': renewable_generation,
    'transformer_load': transformer_load,
    'district': district,
    'upazila': upazila,
    'area_type': area_type,
    'substation_id': substation_id,
    'feeder_id': feeder_id,
    'transformer_age': transformer_age,
    'transformer_capacity': transformer_capacity,
    'outage_history': outage_history,
    'maintenance_due': maintenance_due,
    'population_density': population_density,
    'industrial_load_ratio': industrial_load_ratio
}

X = pd.DataFrame([data])

# Reorder columns to match the scaler's expected order
X = X[feature_order]

# ==========================
# Scale
# ==========================

X_scaled = scaler.transform(X)

# ==========================
# Prediction
# ==========================

prediction = model.predict(X_scaled)[0]
probability = model.predict_proba(X_scaled)[0]

risk = target_encoder.inverse_transform([prediction])[0]

# Calculate prediction time
current_time = datetime.now().replace(
    hour=hour,
    minute=0,
    second=0,
    microsecond=0
)
prediction_time = current_time + timedelta(hours=4)

# ==========================
# Display Results
# ==========================

print("\n")
print("="*60)
print("PREDICTION RESULTS")
print("="*60)

print(f"\nCurrent Time   : {current_time.strftime('%H:%M')}")
print(f"Prediction Time: {prediction_time.strftime('%H:%M')} (+4 Hours)")
print(f"\nPredicted Risk : {risk}")

print("\nConfidence:")

for label, prob in zip(target_encoder.classes_, probability):
    print(f"{label:<8}: {prob*100:.2f}%")

print("\n")

# ==========================
# Recommendations
# ==========================

if risk == "Low":
    print("Grid Status : Stable")
    print("\nRecommendation:")
    print("✓ Grid operating normally.")
    print("✓ No preventive action required.")

elif risk == "Medium":
    print("Grid Status : Moderate Stress")
    print("\nRecommendation:")
    print("• Monitor transformer loading.")
    print("• Reduce peak demand if possible.")
    print("• Prepare standby generation.")

else:  # High
    print("Grid Status : Critical")
    print("\nRecommendation:")
    print("⚠ High overload risk detected.")
    print("⚠ Dispatch maintenance team.")
    print("⚠ Prepare load shedding plan.")
    print("⚠ Increase reserve generation if available.")