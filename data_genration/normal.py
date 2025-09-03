import numpy as np
import pandas as pd
import os
import random

# Constants for synthetic data
DURATION = 600  # 10 minutes
SAMPLE_RATE = 1  # 1Hz
N_SAMPLES = DURATION * SAMPLE_RATE
np.random.seed(42)
random.seed(42)

ROOT_DIR = "synthetic_data"
METADATA_FILE = os.path.join(ROOT_DIR, "metadata.csv")
os.makedirs(ROOT_DIR, exist_ok=True)
if not os.path.exists(METADATA_FILE):
    pd.DataFrame(columns=["sample_id", "category", "subtype", "mode", "intensity","file_path"]).to_csv(METADATA_FILE, index=False)

# Save normal data with metadata
def save_normal_with_metadata(df, subtype, uid):
    out_dir = os.path.join(ROOT_DIR, "normal", subtype)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"normal_{subtype}_{uid:04d}.csv"
    file_path = os.path.join(out_dir, filename)
    df.to_csv(file_path, index=False)
    row = {
        "sample_id": f"normal_{subtype}_{uid:04d}",
        "category": "normal",
        "subtype": subtype, 
        "mode": "NA",
        "intensity": "NA",
        "file_path": file_path
    }
    pd.DataFrame([row]).to_csv(METADATA_FILE, mode="a", header=False, index=False)


# Function to generate normal_1
def generate_normal_1(uid):
    time = np.arange(N_SAMPLES)

    # scale variation
    drift = np.linspace(0, np.random.uniform(-0.2, 0.2), N_SAMPLES)
    vibration = np.random.normal(loc=1.0, scale=0.05, size=N_SAMPLES)
    pressure = np.random.normal(loc=50, scale=1.0, size=N_SAMPLES) + drift
    temperature = np.random.normal(loc=40, scale=0.5, size=N_SAMPLES)

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        'time': time,
        'vibration': vibration,
        'pressure': pressure,
        'temperature': temperature,
        'label': 'normal_1'
    })

    # Save the data
    save_normal_with_metadata(df, "normal_1", uid)


# Function to generate normal_2 (fluctuating normal data)
def generate_normal_2(uid):
    time = np.arange(N_SAMPLES)
    
    # scale variation
    drift = np.linspace(0, np.random.uniform(-1.5, 1.5), N_SAMPLES)
    vibration = np.random.normal(loc=1.0, scale=0.1, size=N_SAMPLES)
    pressure = np.random.normal(loc=50, scale=2.0, size=N_SAMPLES) + drift
    temperature = np.random.normal(loc=40, scale=1.0, size=N_SAMPLES)

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        'time': time,
        'vibration': vibration,
        'pressure': pressure,
        'temperature': temperature,
        'label': 'normal_2'
    })

    # Save the data
    save_normal_with_metadata(df, "normal_2", uid)


# run ALL generators
def generate_normal_data(n_samples=675):
    for i in range(n_samples):
        generate_normal_1(i)
        generate_normal_2(i)
    print(f" Generated {n_samples} samples each for normal_1 and normal_2")

#Run the generation
if __name__ == "__main__":
    generate_normal_data(n_samples=675)