import os
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

count = 0
# set input and output directories
input_base = 'data/problems'
output_base = 'data/problem2/normalized_data'
scaler_path = "data/scalers/minmax_scaler.pkl"
scaler = joblib.load(scaler_path)

# loop through all CSV files in the input directory
for root, dirs, files in os.walk(input_base):
    for file in files:
        if file.endswith('.csv') :
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_base)
            output_path = os.path.join(output_base, rel_path)
            df = pd.read_csv(input_path)
            # Read only vibration, pressure, temperature columns
            features = df[['vibration', 'pressure', 'temperature']].values          # Normalize
            normalized = scaler.transform(features)
            # Replace with normalized values
            df[['vibration', 'pressure', 'temperature']] = normalized
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Save to processed folder
            df.to_csv(output_path, index=False)
            count = count + 1 
print("Total normalized files:", count)