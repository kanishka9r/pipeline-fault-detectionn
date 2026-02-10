import numpy as np
import pandas as pd
import os
import random

# Constants
duration = 600
sample_rate = 1
n_samples = duration * sample_rate

np.random.seed(42)
random.seed(42)

root_dir = os.path.join("..", "data", "normal")
metadata_file = os.path.join("..", "data", "metadata.csv")
if not os.path.exists(os.path.dirname(metadata_file)):
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
os.makedirs(root_dir, exist_ok=True)

# RESET metadata for clean run
pd.DataFrame(
    columns=["sample_id", "category", "fault_type", "mode", "intensity", "file_path"]
).to_csv(metadata_file, index=False)

# Save normal data with metadata
def save_normal_with_metadata(df, uid, subtype):
    out_dir = os.path.join(root_dir, subtype)
    os.makedirs(out_dir, exist_ok=True)

    filename = f"{subtype}_{uid:04d}.csv"
    file_path = os.path.join(out_dir, filename)
    df.to_csv(file_path, index=False)

    row = {
        "sample_id": f"{subtype}_{uid:04d}",
        "category": "normal",
        "fault_type": subtype,
        "mode": "NA",
        "intensity": "NA",
        "file_path": file_path
    }

    pd.DataFrame([row]).to_csv(metadata_file, mode="a", header=False, index=False)

# Normal type 1
def generate_normal_1(uid):
    time = np.arange(n_samples)

    drift = np.linspace(0, np.random.uniform(-0.2, 0.2), n_samples)
    vibration = np.random.normal(5.0, 1.0, n_samples)
    pressure = np.random.normal(50, 2.0, n_samples) + drift
    temperature = np.random.normal(40, 1.0, n_samples)

    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": np.array(["normal"]*n_samples, dtype=object)
    })

    save_normal_with_metadata(df, uid, "normal_1")

# Normal type 2
def generate_normal_2(uid):
    time = np.arange(n_samples)

    drift = np.linspace(0, np.random.uniform(-1.5, 1.5), n_samples)
    vibration = np.random.normal(5.0, 0.1, n_samples)
    pressure = np.random.normal(50, 2.0, n_samples) + drift
    temperature = np.random.normal(40, 1.0, n_samples)

    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": np.array(["normal"]*n_samples, dtype=object)
    })

    save_normal_with_metadata(df, uid, "normal_2")

# Generate all normal data
def generate_normal_data(n_files=675):
    for i in range(n_files):
        generate_normal_1(i)
        generate_normal_2(i)

    print(f"Generated {n_files*2} normal files")

if __name__ == "__main__":
    generate_normal_data()
