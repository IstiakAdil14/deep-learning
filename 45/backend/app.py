from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import urllib.request
import urllib.parse
import json as json_module
from datetime import datetime, timedelta

app = FastAPI(title="SmartGrid Risk Prediction API")

# District/Upazila coordinates for Sylhet Division, Bangladesh
# Approximate center coordinates for each upazila
# (Names must match exactly with encoder classes)
LOCATION_COORDINATES = {
    "Sylhet": {
        "Balaganj": (24.8500, 91.4167),
        "Beanibazar": (24.9333, 91.4667),
        "Bishwanath": (24.8000, 91.4000),
        "Companiganj": (24.9500, 91.4833),
        "Dakshin Surma": (24.8833, 91.3833),
        "Fenchuganj": (24.8833, 91.4000),
        "Golapganj": (24.9000, 91.3667),
        "Gowainghat": (24.7000, 91.4500),
        "Jaintiapur": (24.9000, 91.4333),
        "Kanaighat": (24.9667, 91.4333),
        "Osmani Nagar": (24.9000, 91.4167),
        "Sylhet Sadar": (24.9000, 91.4000),
        "Zakiganj": (24.8167, 91.3667)
    },
    "Habiganj": {
        "Ajmiriganj": (24.3833, 91.4333),
        "Bahubal": (24.4000, 91.4500),
        "Baniachang": (24.3500, 91.4667),
        "Chunarughat": (24.3667, 91.4167),
        "Habiganj Sadar": (24.3833, 91.4167),
        "Lakhai": (24.2833, 91.4000),
        "Madhabpur": (24.2667, 91.4333),
        "Nabiganj": (24.4167, 91.4333),
        "Shaistaganj": (24.4333, 91.4500)
    },
    "Moulvibazar": {
        "Barlekha": (24.3500, 91.7667),
        "Juri": (24.4167, 91.7833),
        "Kamalganj": (24.3667, 91.7833),
        "Kulaura": (24.3167, 91.7667),
        "Moulvibazar Sadar": (24.3333, 91.7667),
        "Rajnagar": (24.3833, 91.7833),
        "Sreemangal": (24.3000, 91.7833)
    },
    "Sunamganj": {
        "Bishwamvarpur": (24.9667, 91.2333),
        "Chhatak": (24.9333, 91.2500),
        "Dakshin Sunamganj": (24.9500, 91.2333),
        "Derai": (24.8167, 91.2833),
        "Dharmapasha": (24.8500, 91.3000),
        "Dowarabazar": (24.7833, 91.2333),
        "Jagannathpur": (24.7667, 91.2500),
        "Jamalganj": (24.8000, 91.2667),
        "Sullah": (24.8667, 91.2667),
        "Tahirpur": (24.8667, 91.2833),
        "Sunamganj Sadar": (24.9333, 91.2500),
        "Shantiganj": (24.9000, 91.2500)
    }
}


def get_weather_from_api(latitude: float, longitude: float) -> dict:
    """Fetch current weather data from Open-Meteo API."""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = urllib.parse.urlencode({
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,rain",
            "timezone": "Asia/Dhaka"
        })
        full_url = f"{url}?{params}"
        req = urllib.request.Request(full_url)
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json_module.loads(response.read().decode())
            current = data.get("current", {})
            
            # Map weather condition based on rain and cloud cover
            rainfall = float(current.get("rain", 0) or 0)
            condition = "Sunny" if rainfall == 0 else ("Rainy" if rainfall > 2 else "Cloudy")
            
            return {
                "temperature": float(current.get("temperature_2m", 26.0) or 26.0),
                "humidity": float(current.get("relative_humidity_2m", 70) or 70),
                "rainfall": rainfall,
                "wind_speed": float(current.get("wind_speed_10m", 5.0) or 5.0),
                "condition": condition
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    
    # Fallback to simulated weather
    return {
        "temperature": 26.0 + np.random.uniform(-2, 2),
        "humidity": 70 + np.random.uniform(-10, 10),
        "rainfall": np.random.uniform(0, 5),
        "wind_speed": 5 + np.random.uniform(-2, 3),
        "condition": "Sunny"
    }

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and encoders
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rf_model = joblib.load(os.path.join(BASE_DIR, "..", "models", "random_forest.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "..", "models", "scaler.pkl"))
target_encoder = joblib.load(os.path.join(BASE_DIR, "..", "models", "target_encoder.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "..", "models", "encoders.pkl"))
grid_assets = pd.read_csv(os.path.join(BASE_DIR, "..", "dataset", "grid_assets.csv"))
feature_order = list(scaler.feature_names_in_)


class PredictionRequest(BaseModel):
    district: str
    upazila: str


class PredictionResponse(BaseModel):
    risk_level: str
    confidence: dict
    weather: dict
    prediction_time: str
    recommendation: List[str]


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Smart Grid Sentinel API"}


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
    """Predict risk level for the next 4 hours."""
    # Find grid asset info
    asset = grid_assets[
        (grid_assets["district"] == request.district) & 
        (grid_assets["upazila"] == request.upazila)
    ].iloc[0]
    
    # Get current time features
    now = datetime.now()
    current_hour = now.hour
    weekday = now.weekday()
    prediction_time = now + timedelta(hours=4)
    
    # Get coordinates for the selected upazila
    coords = LOCATION_COORDINATES.get(request.district, {}).get(request.upazila)
    if coords:
        weather = get_weather_from_api(coords[0], coords[1])
    else:
        # Fallback to simulated weather
        weather = {
            "temperature": 26.0 + np.random.uniform(-2, 2),
            "humidity": 70 + np.random.uniform(-10, 10),
            "rainfall": np.random.uniform(0, 5),
            "wind_speed": 5 + np.random.uniform(-2, 3),
            "condition": "Sunny"
        }
    
    # Encode weather condition
    weather_condition = weather.get("condition", "Sunny")
    try:
        weather_state = list(encoders["weather_state"].classes_).index(weather_condition)
    except ValueError:
        weather_state = 0  # Default to Sunny
    
    # Encode district and upazila using trained encoders
    try:
        district_encoded = list(encoders["district"].classes_).index(request.district)
    except ValueError:
        district_encoded = 0
    
    try:
        upazila_encoded = list(encoders["upazila"].classes_).index(request.upazila)
    except ValueError:
        upazila_encoded = 0
    
    # Get asset features
    area_type = asset["area_type"]
    try:
        area_type_encoded = list(encoders["area_type"].classes_).index(area_type)
    except ValueError:
        area_type_encoded = 0
    
    # Build features with proper encoding
    features = {
        "hour": current_hour,
        "weekday": weekday,
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "rainfall": weather["rainfall"],
        "wind_speed": weather["wind_speed"],
        "weather_state": weather_state,
        "electricity_demand": 150,
        "renewable_generation": 120,
        "transformer_load": 75,
        "district": district_encoded,
        "upazila": upazila_encoded,  # Now properly encoded!
        "area_type": area_type_encoded,
        "substation_id": int(asset["substation_id"].replace("SS_", "")) if isinstance(asset["substation_id"], str) else int(asset["substation_id"]),
        "feeder_id": int(asset["feeder_id"].replace("FDR_", "")) if isinstance(asset["feeder_id"], str) else int(asset["feeder_id"]),
        "transformer_age": int(asset["transformer_age"]),
        "transformer_capacity": float(asset["transformer_capacity"]),
        "outage_history": int(asset["outage_history"]),
        "maintenance_due": list(encoders["maintenance_due"].classes_).index(asset["maintenance_due"]) if asset["maintenance_due"] in encoders["maintenance_due"].classes_ else 0,
        "population_density": float(asset["population_density"]),
        "industrial_load_ratio": float(asset["industrial_load_ratio"])
    }
    
    # Create DataFrame with explicit feature names
    X = pd.DataFrame([features], columns=feature_order)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = rf_model.predict(X_scaled)[0]
    probability = rf_model.predict_proba(X_scaled)[0]
    risk = target_encoder.inverse_transform([prediction])[0]
    
    # Generate recommendations based on risk level
    recommendations = []
    if risk == "Low":
        recommendations = [
            "Monitor grid status",
            "Grid operating normally"
        ]
    elif risk == "Medium":
        recommendations = [
            "Monitor transformer loading",
            "Prepare reserve generation",
            "Inform maintenance team"
        ]
    else:  # High
        recommendations = [
            "High overload risk detected - Dispatch maintenance team",
            "Prepare load shedding plan",
            "Increase reserve generation if available"
        ]
    
    # Convert probability to dict
    confidence = {label: float(prob) for label, prob in zip(target_encoder.classes_, probability)}
    
    return {
        "risk_level": risk,
        "confidence": confidence,
        "weather": weather,
        "prediction_time": prediction_time.strftime("%Y-%m-%d %H:%M"),
        "recommendation": recommendations
    }