import os
import pandas as pd
import joblib
from pathlib import Path

# Load the saved scaler
scaler = joblib.load('data/scalers/minmax_scaler.pkl')
count = 0


# set input and output directories
input_base = 'data/problems'
output_base = 'data/problems/normalized_data'


# loop through all CSV files in the input directory
for root, dirs, files in os.walk(input_base):
    for file in files:
        if file.endswith('.csv') :
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_base)
            output_path = os.path.join(output_base, rel_path)
            # Read only vibration, pressure, temperature columns
            df = pd.read_csv(input_path)

            features = df[['vibration', 'pressure', 'temperature']]            # Normalize
            normalized = scaler.transform(features.values)
            # Replace with normalized values
            df[['vibration', 'pressure', 'temperature']] = normalized
            # Ensure output directory exists
            Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
            # Save to processed folder
            df.to_csv(output_path, index=False)
            count = count + 1

print(f"Processed and saved: {output_path}") 
print(count)

           


                      