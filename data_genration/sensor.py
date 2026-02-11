import numpy as np
import pandas as pd
import os
import random

# Constants for synthetic data
duration = 600  # 10 minutes
sample_rate = 1 # 1Hz
n_sample = duration * sample_rate
np.random.seed(42)
random.seed(42)

root_dir = os.path.join("..", "data", "problems", "sensor")
metadata_file = os.path.join("..", "data", "metadata.csv")
os.makedirs(root_dir, exist_ok=True)
if not os.path.exists(os.path.dirname(metadata_file)):
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
pd.DataFrame(columns=["sample_id","category" , "sensor",  "mode" , "intensity",  "file_path"]).to_csv(metadata_file, index=False)

# save data with metadata
def save_with_metadata(df, sensor, mode,intensity ,  uid):
    out_dir = os.path.join(root_dir, sensor, intensity, mode)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{sensor}_{mode}_{intensity}_{uid:04d}.csv"
    file_path = os.path.join(out_dir, filename)
    df.to_csv(file_path, index=False)
    row = {
        "sample_id": f"{sensor}_{mode}_{intensity}_{uid:04d}",
        "category": "sensor_fault",
        "fault_type": f"{sensor}_noise",
        "mode": mode,
        "intensity": intensity,
        "file_path": file_path
    }
    pd.DataFrame([row]).to_csv(metadata_file, mode="a", header=False, index=False  ) 

# Function to get the fault window based on mode
def get_fault_window(mode):
    if mode == "start":
        start = int(random.uniform(0.1, 0.3) * n_sample)
        fault_len = int(random.uniform(0.4, 0.6) * n_sample)
        end = min(start + fault_len, n_sample)
        return start, end

    elif mode == "recover":
        start = random.randint(n_sample // 4, n_sample // 3)
        fault_len = random.randint(n_sample // 4, n_sample // 3)
        end = min(start + fault_len, n_sample)
        return start, end

    else:
        raise ValueError("Invalid mode")


# pressure Noise Fault
def generate_pressure_noise(sample_id, mode="start" , intensity="high"):
    time = np.arange(n_sample)

    # Generate values for each sensor
    vibration = np.random.normal(1.0, 0.05, n_sample)
    pressure = np.random.normal(50.0, 1.0, n_sample)
    temperature = np.random.normal(40.0, 0.5, n_sample)
    label = np.array(["normal"] * n_sample, dtype=object)
    start, end = get_fault_window(mode)
    fault_len = end - start

    # Intensity of noise
    if intensity == "low":
        noise_std = 3.0
    else:
        noise_std = 8.0


    # Inject noise
    if mode == "start":
        pressure[start:end] += np.random.normal(0, noise_std, fault_len)
    elif mode == "recover":
        half = fault_len // 2
        pressure[start:start+half] += np.random.normal(0, noise_std, half)
        pressure[start+half:end] += np.random.normal(0, noise_std*0.3, (end - (start+half)))
    label[start:end] = "pressurenoise"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save data
    save_with_metadata(df, "pressure", mode, intensity , sample_id)


# Temperature Noise Fault
def generate_temperature_noise(sample_id, mode="start" , intensity="high"):
    time = np.arange(n_sample)

    # Generate values for each sensor
    vibration = np.random.normal(1.0, 0.05, n_sample)
    pressure = np.random.normal(50.0, 1.0, n_sample)
    temperature = np.random.normal(40.0, 0.5, n_sample)
    label = np.array(["normal"] * n_sample, dtype=object)
    start, end = get_fault_window(mode)
    fault_len = end - start

     # Intensity of noise
    if intensity == "low":
        noise_std = 2.0
    else:
        noise_std = 6.0

    # Inject noise 
    if mode == "start":
        temperature[start:end] += np.random.normal(0, noise_std, fault_len)
    elif mode == "recover":
        half = fault_len // 2
        temperature[start:start+half] += np.random.normal(0, noise_std, half)
        temperature[start+half:end] += np.random.normal(0, noise_std*0.3, (end - (start+half)))
    label[start:end] = "temperaturenoise"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save data
    save_with_metadata(df, "temperature", mode, intensity , sample_id)


# Vibration Noise Fault
def generate_vibration_noise(sample_id, mode="start" , intensity="high"):
    time = np.arange(n_sample)
    
    # Generate  values for each sensor
    vibration = np.random.normal(1.0, 0.05, n_sample)
    pressure = np.random.normal(50.0, 1.0, n_sample)
    temperature = np.random.normal(40.0, 0.5, n_sample)
    label = np.array(["normal"] * n_sample, dtype=object)
    start, end = get_fault_window(mode)
    fault_len = end - start

    # Intensity of noise
    if intensity == "low":
        noise_std = 0.3
    else:
        noise_std = 0.8

    # Inject noise
    if mode == "start":
        vibration[start:end] += np.random.normal(0, noise_std, fault_len)
    elif mode == "recover":
        half = fault_len // 2
        vibration[start:start+half] += np.random.normal(0, noise_std, half)
        vibration[start+half:end] += np.random.normal(0, noise_std*0.3, (end - (start+half)))
    label[start:end] = "vibrationnoise"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save data
    save_with_metadata(df, "vibration", mode, intensity , sample_id)   

# Function to generate noise faults
def generate_sensor_fault_noise_dataset(n_samples=25):
    wid = 1 
    for mode in ["start" , "recover"]:
            for intensity in ["low", "high"]:
                for _ in range(n_samples):
                    generate_pressure_noise(wid, mode=mode , intensity = intensity)   ; wid += 1
                    generate_temperature_noise(wid, mode=mode , intensity = intensity) ; wid += 1
                    generate_vibration_noise(wid, mode=mode , intensity = intensity) ; wid += 1
    print(f" All sensor faults generated. Metadata saved at {metadata_file}")    
# Run the generation
if __name__ == "__main__":
    generate_sensor_fault_noise_dataset(n_samples=25)                        
