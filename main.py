from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import joblib
import pandas as pd

app = FastAPI()

# Enable CORS so the React frontend can connect without being blocked
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load the brain you just trained
model = joblib.load('thermal_model_v3.pkl')

# 2. Define the exact inputs your friend's UI will send
class ShelterInput(BaseModel):
    lat: float
    lon: float
    shape_code: int    
    length: float
    width: float
    height: float
    r_value: float     
    window_area: float
    occupants: int
    initial_temp: float = 15.0 # Baseline starting temp

@app.post("/simulate")
def run_simulation(data: ShelterInput):
    # 3. Fetch live weather from NASA API
    nasa_url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point?"
        f"parameters=T2M,ALLSKY_SFC_SW_DWN,WS10M&community=RE&"
        f"longitude={data.lon}&latitude={data.lat}&"
        f"start=20260901&end=20260901&format=JSON"
    )
    nasa_data = requests.get(nasa_url).json()
    temps = list(nasa_data['properties']['parameter']['T2M'].values())[:24]
    solar = list(nasa_data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].values())[:24]
    wind = list(nasa_data['properties']['parameter']['WS10M'].values())[:24]

    results = {"hourly_inside_temp": [], "hourly_heat_loss": [], "hourly_solar_gain": []}
    
    current_temp = data.initial_temp 
    total_passive_heat_joules = 0
    
    # 4. Predict the 24-hour cycle using the ML model
    for hour in range(24):
        features = pd.DataFrame([[
            data.shape_code, data.length, data.width, data.height, 
            data.r_value, data.window_area, data.occupants, 
            temps[hour], solar[hour], wind[hour], current_temp
        ]], columns=[
            'Shape_Code', 'Length', 'Width', 'Height', 'R_Value', 
            'Window_Area', 'Occupants', 'Outside_Temp', 'Solar_Power', 'Wind_Speed', 'Previous_Temp'
        ])
        
        prediction = model.predict(features)[0]
        predicted_temp = float(prediction[0])
        solar_gain_watts = float(prediction[1])
        heat_loss_watts = float(prediction[2])
        
        results["hourly_inside_temp"].append(predicted_temp)
        results["hourly_solar_gain"].append(solar_gain_watts)
        results["hourly_heat_loss"].append(heat_loss_watts)
        
        total_passive_heat_joules += (solar_gain_watts + (data.occupants * 100)) * 3600
        current_temp = predicted_temp

    # 5. Calculate Kerosene Savings & Carbon Reduction
    liters_saved = total_passive_heat_joules / 35_000_000
    co2_prevented = liters_saved * 2.5
    
    results["sustainability"] = {
        "kerosene_saved_liters": round(liters_saved, 2),
        "co2_emissions_prevented_kg": round(co2_prevented, 2)
    }

    # 6. Calculate Livability Score (Out of 100)
    score = 100.0
    for t in results["hourly_inside_temp"]:
        if t < 15.0: score -= 1.5
        elif t > 25.0: score -= 1.0
        
    results["livability_score"] = max(0, round(score))

    return results