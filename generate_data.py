import pandas as pd
import random
import math

data = []

print("Building the fully bounded dataset...")

for _ in range(2000):
    shape_code = random.choice([1, 2, 3, 4, 5]) 
    length = random.uniform(3.0, 10.0)
    width = random.uniform(3.0, 10.0)
    height = random.uniform(2.5, 4.0)
    r_value = random.uniform(1.0, 6.0) 
    window_area = random.uniform(1.0, 8.0) 
    occupants = random.randint(0, 15)
    
    if shape_code == 1: 
        envelope_area = (2 * length * height) + (2 * width * height) + (length * width)
        volume = length * width * height
    elif shape_code == 2: 
        r = length / 2.0
        envelope_area = 2 * math.pi * (r ** 2)
        volume = (2/3) * math.pi * (r ** 3)
    elif shape_code == 3: 
        slant = math.sqrt((width / 2)**2 + height**2)
        envelope_area = (2 * slant * length) + (width * height)
        volume = 0.5 * width * length * height
    elif shape_code == 4: 
        r = length / 2.0
        envelope_area = (2 * math.pi * r * height) + (math.pi * (r ** 2))
        volume = math.pi * (r ** 2) * height
    else: 
        r = width / 2.0
        envelope_area = (math.pi * r * length) + (math.pi * (r ** 2))
        volume = 0.5 * math.pi * (r ** 2) * length

    current_inside_temp = random.uniform(5, 20)

    for hour in range(24):
        outside_temp = random.uniform(-30, 10) 
        solar_power = random.uniform(0, 1100)
        wind_speed = random.uniform(0, 20) 
        
        solar_gain = solar_power * window_area * 0.6 
        internal_heat = occupants * 100.0 
        
        u_value = 1 / r_value
        wind_multiplier = 1 + (wind_speed * 0.05) 
        heat_loss = u_value * (envelope_area - window_area) * (current_inside_temp - outside_temp) * wind_multiplier
        
        net_heat = solar_gain + internal_heat - heat_loss
        
        # Physics calculation
        raw_temp_change = (net_heat * 3600) / (150000.0 * volume) 
        
        # CLAMP 1: Max change of 3 degrees per hour to prevent oscillation
        temp_change = max(-3.0, min(3.0, raw_temp_change))
        next_inside_temp = current_inside_temp + temp_change
        
        # CLAMP 2: Absolute physical limits (-30C to 45C)
        next_inside_temp = max(-30.0, min(45.0, next_inside_temp))
        
        data.append([
            shape_code, length, width, height, r_value, window_area, occupants, 
            outside_temp, solar_power, wind_speed, current_inside_temp, 
            next_inside_temp, solar_gain, heat_loss
        ])
        current_inside_temp = next_inside_temp

df = pd.DataFrame(data, columns=[
    'Shape_Code', 'Length', 'Width', 'Height', 'R_Value', 'Window_Area', 'Occupants', 
    'Outside_Temp', 'Solar_Power', 'Wind_Speed', 'Previous_Temp', 
    'Inside_Temp', 'Solar_Gain', 'Heat_Loss'
])
df.to_csv('training_data.csv', index=False)
print("Dataset created successfully!")