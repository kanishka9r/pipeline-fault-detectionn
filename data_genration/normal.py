import numpy as np
import pandas as pd
import os
import random

# Constants for synthetic data
duration = 600  # 10 minutes
sample_rate = 1  # 1Hz
n_samples = duration * sample_rate
np.random.seed(42)
random.seed(42)

root_dir = os.path.join("..", "data", "normal")
metadata_file = os.path.join("..", "data", "metadata.csv")
os.makedirs(root_dir, exist_ok=True)
if not os.path.exists(metadata_file):
    pd.DataFrame(columns=["sample_id", "category", "subtype", "mode", "intensity","file_path"]).to_csv(metadata_file, index=False)

# Save normal data with metadata
def save_normal_with_metadata(df, uid , subtype):
    out_dir = os.path.join(root_dir, subtype)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{subtype}_{uid:04d}.csv"
    file_path = os.path.join(out_dir, filename)
    df.to_csv(file_path, index=False)
    row = {
        "sample_id": f"{subtype}_{uid:04d}",
        "category": "normal",
        "fault_type": "NA", 
        "mode": "NA",
        "intensity": "NA",
        "file_path": file_path
    }
    pd.DataFrame([row]).to_csv(metadata_file, mode="a", header=False, index=False)


# Function to generate normal_1
def generate_normal_1(uid):
    time = np.arange(n_samples)

    # scale variation
    drift = np.linspace(0, np.random.uniform(-0.2, 0.2), n_samples)
    vibration = np.random.normal(loc=1.0, scale=0.05, size=n_samples)
    pressure = np.random.normal(loc=50, scale=1.0, size=n_samples) + drift
    temperature = np.random.normal(loc=40, scale=0.5, size=n_samples)

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        'time': time,
        'vibration': vibration,
        'pressure': pressure,
        'temperature': temperature,
        'label': 'normal_1'
    })

    # Save the data
    save_normal_with_metadata(df, uid , "normal_1")


# Function to generate normal_2 (fluctuating normal data)
def generate_normal_2(uid):
    time = np.arange(n_samples)
    
    # scale variation
    drift = np.linspace(0, np.random.uniform(-1.5, 1.5), n_samples)
    vibration = np.random.normal(loc=1.0, scale=0.1, size=n_samples)
    pressure = np.random.normal(loc=50, scale=2.0, size=n_samples) + drift
    temperature = np.random.normal(loc=40, scale=1.0, size=n_samples)

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        'time': time,
        'vibration': vibration,
        'pressure': pressure,
        'temperature': temperature,
        'label': 'normal_2'
    })

    # Save the data
    save_normal_with_metadata(df, uid , "normal_2")


# run ALL generators
def generate_normal_data(n_samples=675):
    for i in range(n_samples):
        generate_normal_1(i)
        generate_normal_2(i)
    print(f" Generated {n_samples} samples each for normal_1 and normal_2")

#Run the generation
if __name__ == "__main__":
    generate_normal_data(n_samples=675)