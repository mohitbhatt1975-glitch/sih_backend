import pandas as pd
from xgboost import XGBRegressor
import joblib

print("Step 1: Reading the training_data.csv file...")
df = pd.read_csv('training_data.csv')

X = df[['Shape_Code', 'Length', 'Width', 'Height', 'R_Value', 'Window_Area', 'Occupants', 
        'Outside_Temp', 'Solar_Power', 'Wind_Speed', 'Previous_Temp']]
y = df[['Inside_Temp', 'Solar_Gain', 'Heat_Loss']]

print("Step 2: Training the AI Model (This takes a few seconds)...")
model = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X, y)

print("Step 3: Saving the brain...")
joblib.dump(model, 'thermal_model_v3.pkl')
print("SUCCESS: thermal_model_v3.pkl has been created!")