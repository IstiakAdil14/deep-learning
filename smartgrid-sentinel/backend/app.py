from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import urllib.request
import json

app = FastAPI(title="SmartGrid Risk Prediction API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and data
model = load_model("../models/lstm_model.keras")
scaler = joblib.load("../models/scaler.pkl")
target_encoder = joblib.load("../models/target_encoder.pkl")
grid_assets = pd.read_csv("../dataset/grid_assets.csv")

# Mappings
weather_map = {"Sunny": 0, "Cloudy": 1, "Rainy": 2, "Stormy": 2}
area_map = {"Urban": 0, "Rural": 1}
maintenance_map = {"No": 0, "Yes": 1}
district_map = {"Sylhet": 0, "Habiganj": 1, "Moulvibazar": 2, "Sunamganj": 3}
upazila_map = {
    "Balaganj": 0, "Beanibazar": 1, "Bishwanath": 2, "Companiganj": 3,
    "Dakshin Surma": 4, "Fenchuganj": 5, "Golapganj": 6, "Gowainghat": 7,
    "Jaintiapur": 8, "Kanaighat": 9, "Osmani Nagar": 10, "Sylhet Sadar": 11,
    "Zakiganj": 12, "Ajmiriganj": 13, "Bahubal": 14, "Baniachang": 15, 
    "Chunarughat": 16, "Habiganj Sadar": 17, "Lakhai": 18, "Madhabpur": 19, 
    "Nabiganj": 20, "Shaistaganj": 21, "Barlekha": 22, "Juri": 23, 
    "Kamalganj": 24, "Kulaura": 25, "Moulvibazar Sadar": 26, "Rajnagar": 27, 
    "Sreemangal": 28, "Bishwamvarpur": 29, "Chhatak": 30, "Dakshin Sunamganj": 31, 
    "Derai": 32, "Dharmapasha": 33, "Dowarabazar": 34, "Jagannathpur": 35, 
    "Jamalganj": 36, "Sullah": 37, "Sunamganj Sadar": 38, "Tahirpur": 39, 
    "Shantiganj": 40
}

district_coords = {
    "Sylhet": {"lat": 24.8949, "lon": 91.8687},
    "Habiganj": {"lat": 24.3840, "lon": 91.4169},
    "Moulvibazar": {"lat": 24.4842, "lon": 91.7685},
    "Sunamganj": {"lat": 25.0664, "lon": 91.4074}
}


class PredictionRequest(BaseModel):
    district: str
    upazila: str


class PredictionResponse(BaseModel):
    risk_level: str
    confidence: dict
    weather: dict
    prediction_time: str
    recommendation: List[str]


@app.get("/districts")
def get_districts():
    """Get list of available districts."""
    districts = grid_assets["district"].unique().tolist()
    return districts


@app.get("/upazilas/{district}")
def get_upazilas(district: str):
    """Get upazilas for a given district."""
    upazilas = grid_assets[grid_assets["district"] == district]["upazila"].unique().tolist()
    return upazilas


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(request: PredictionRequest):
    """Predict risk level for the next 4 hours using Live Weather and LSTM."""
    
    # 1. Fetch grid asset data
    asset = grid_assets[
        (grid_assets["district"] == request.district) & 
        (grid_assets["upazila"] == request.upazila)
    ].iloc[0]
    
    # 2. Get prediction time (+4 Hours)
    now = datetime.now()
    pred_time = now + timedelta(hours=4)
    hour = pred_time.hour
    weekday = pred_time.weekday()
    
    # Determine the time period for grid loads
    if 6 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 18:
        period = "afternoon"
    elif 18 <= hour < 22:
        period = "evening"
    else:
        period = "night"
        
    electricity_demand = float(asset[f"demand_{period}"])
    renewable_generation = float(asset[f"renewable_{period}"])
    transformer_load = float(asset[f"transformer_load_{period}"])

    # 3. Fetch Live Weather from Open-Meteo
    base_coords = district_coords.get(request.district, district_coords["Sylhet"])
    
    # Generate deterministic geographic offset for the specific upazila
    # This ensures each upazila queries a slightly different weather grid cell (~20km radius)
    upazila_hash = sum(ord(c) for c in request.upazila)
    lat_offset = ((upazila_hash % 100) / 100.0) * 0.4 - 0.2
    lon_offset = (((upazila_hash * 3) % 100) / 100.0) * 0.4 - 0.2
    
    target_lat = round(base_coords['lat'] + lat_offset, 4)
    target_lon = round(base_coords['lon'] + lon_offset, 4)

    url = f"https://api.open-meteo.com/v1/forecast?latitude={target_lat}&longitude={target_lon}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"
    
    api_error = None
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'SmartGrid/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            weather_data = json.loads(response.read().decode())
        current_w = weather_data['current']
        temp = current_w['temperature_2m']
        hum = current_w['relative_humidity_2m']
        rain = current_w['precipitation']
        wind = current_w['wind_speed_10m']
        w_code = current_w['weather_code']
    except Exception as e:
        print(f"Weather API failed: {e}")
        api_error = str(e)
        # Fallback to sensible defaults
        temp, hum, rain, wind, w_code = 30.0, 70.0, 0.0, 5.0, 0
        
    # Map WMO weather code to frontend condition
    if w_code in [0, 1]:
        cond_str = "Sunny"
    elif w_code in [2, 3, 45, 48]:
        cond_str = "Cloudy"
    elif w_code in [95, 96, 99]:
        cond_str = "Stormy"
    else:
        cond_str = "Rainy"

    if api_error:
        cond_str = f"API Error: {api_error[:15]}"

    weather = {
        "temperature": temp,
        "humidity": hum,
        "rainfall": rain,
        "wind_speed": wind,
        "condition": cond_str
    }

    # 4. Process Categorical Encodings
    substation_id = int(str(asset["substation_id"]).replace("SS_", ""))
    feeder_id = int(str(asset["feeder_id"]).replace("FDR_", ""))
    area_type = area_map.get(asset["area_type"], 0)
    maintenance_due = maintenance_map.get(asset["maintenance_due"], 0)

    # 5. Build Feature Vector
    feature_dict = {
        'hour': hour,
        'weekday': weekday,
        'temperature': temp,
        'humidity': hum,
        'rainfall': rain,
        'wind_speed': wind,
        'weather_state': weather_map.get(cond_str, 0),
        'electricity_demand': electricity_demand,
        'renewable_generation': renewable_generation,
        'transformer_load': transformer_load,
        'district': district_map.get(request.district, 0),
        'upazila': upazila_map.get(request.upazila, 0),
        'area_type': area_type,
        'substation_id': substation_id,
        'feeder_id': feeder_id,
        'transformer_age': int(asset["transformer_age"]),
        'transformer_capacity': float(asset["transformer_capacity"]),
        'outage_history': int(asset["outage_history"]),
        'maintenance_due': maintenance_due,
        'population_density': float(asset["population_density"]),
        'industrial_load_ratio': float(asset["industrial_load_ratio"])
    }

    feature_order = list(scaler.feature_names_in_)
    X = pd.DataFrame([feature_dict])[feature_order]

    # 6. Scale and Predict
    X_scaled = scaler.transform(X)
    
    if model is not None:
        # LSTM expects 3D input: (batch, timesteps, features)
        X_lstm = X_scaled.reshape(1, 1, -1)
        prob_array = model.predict(X_lstm, verbose=0)[0]
        prediction_idx = int(np.argmax(prob_array))
    else:
        prob_array = [0.1, 0.7, 0.2]
        prediction_idx = 1
        
    risk_level = target_encoder.inverse_transform([prediction_idx])[0]
    
    # Map confidence back to classes (e.g. ['High', 'Low', 'Medium'])
    conf_dict = {}
    for label, prob in zip(target_encoder.classes_, prob_array):
        conf_dict[label] = float(prob)
        
    # 7. Generate Recommendations
    if risk_level == "Low":
        recs = ["✓ Grid operating normally.", "✓ No preventive action required."]
    elif risk_level == "Medium":
        recs = ["• Monitor transformer loading.", "• Reduce peak demand if possible.", "• Prepare standby generation."]
    else:
        recs = ["⚠ High overload risk detected.", "⚠ Dispatch maintenance team.", "⚠ Prepare load shedding plan.", "⚠ Increase reserve generation if available."]

    return {
        "risk_level": risk_level,
        "confidence": conf_dict,
        "weather": weather,
        "prediction_time": pred_time.strftime("%H:%M"),
        "recommendation": recs
    }