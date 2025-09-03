import numpy as np
import pandas as pd
import os
import random

# Constants for synthetic data
DURATION = 600  # 10 minutes
SAMPLE_RATE = 1 # 1Hz
N_SAMPLES = DURATION * SAMPLE_RATE
np.random.seed(42)
random.seed(42)

ROOT_DIR = os.path.join("..", "data", "problems", "sensor_fault")
METADATA_FILE = os.path.join("..", "data", "metadata.csv")
os.makedirs(ROOT_DIR, exist_ok=True)
if not os.path.exists(METADATA_FILE):
    pd.DataFrame(columns=["sample_id","category" , "sensor",  "mode" , "intensity",  "file_path"]).to_csv(METADATA_FILE, index=False)

# save data with metadata
def save_with_metadata(df, sensor, mode,intensity ,  uid):
    out_dir = os.path.join(ROOT_DIR, sensor, intensity, mode)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{sensor}_{mode}_{intensity}_{uid:04d}.csv"
    file_path = os.path.join(out_dir, filename)
    df.to_csv(file_path, index=False)
    row = {
        "sample_id": f"{sensor}_{mode}_{uid:04d}",
        "category": "sensor_fault",
        "sensor": sensor,
        "mode": mode,
        "intensity": intensity,
        "file_path": file_path
    }
    pd.DataFrame([row]).to_csv(METADATA_FILE, mode="a", header=False, index=False  ) 

# Function to get the fault window based on mode
def get_fault_window(mode):
    if mode == "start":
        return 0, N_SAMPLES
    elif mode == "random":
        start = random.randint(N_SAMPLES // 3, N_SAMPLES // 2)
        return start, N_SAMPLES
    elif mode == "recover":
        start = random.randint(N_SAMPLES // 4, N_SAMPLES // 3)
        return start, start + N_SAMPLES // 3
    else:
        raise ValueError("Invalid mode")


# pressure Noise Fault
def generate_pressure_noise(sample_id, mode="random" , intensity="high"):
    time = np.arange(N_SAMPLES)

    # Scale variation
    vib_loc = np.random.uniform(0.9, 1.1)
    vib_scale = 0.05
    pres_loc = np.random.uniform(48, 52)
    pres_scale = 1.0
    temp_loc = np.random.uniform(38, 42)
    temp_scale = 0.5

    # Generate values for each sensor
    vibration = np.random.normal(loc=vib_loc, scale=vib_scale, size=N_SAMPLES)
    pressure = np.random.normal(loc=pres_loc, scale=pres_scale, size=N_SAMPLES)
    temperature = np.random.normal(loc=temp_loc, scale=temp_scale, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    start , end = get_fault_window(mode)
    fault_len = end - start

    # Intensity of noise
    if intensity == "low":
        noise_loc, noise_scale = 60, 5
    else:
        noise_loc, noise_scale = 70, 15

    # Inject noise
    pressure[start:end] = np.random.normal(loc=noise_loc, scale=noise_scale, size=fault_len)
    label[start:end] = "pressure_noise"

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
def generate_temperature_noise(sample_id, mode="random" , intensity="high"):
    time = np.arange(N_SAMPLES)

    # Scale variation
    vib_loc = np.random.uniform(0.9, 1.1)
    vib_scale = 0.05
    pres_loc = np.random.uniform(48, 52)
    pres_scale = 1.0
    temp_loc = np.random.uniform(38, 42)
    temp_scale = 0.5


    # Generate values for each sensor
    vibration = np.random.normal(loc=vib_loc, scale=vib_scale, size=N_SAMPLES)
    pressure = np.random.normal(loc=pres_loc, scale=pres_scale, size=N_SAMPLES)
    temperature = np.random.normal(loc=temp_loc, scale=temp_scale, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    start , end = get_fault_window(mode)
    fault_len = end - start

     # Intensity of noise
    if intensity == "low":
        noise_loc, noise_scale = 50, 5
    else:
        noise_loc, noise_scale = 65, 15

    # Apply fault 
    temperature[start:end]  = np.random.normal(loc=noise_loc, scale=noise_scale, size=fault_len)
    label[start:end] = "temperature_noise"

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
def generate_vibration_noise(sample_id, mode="random" , intensity="high"):
    time = np.arange(N_SAMPLES)

    # Scale variation
    vib_loc = np.random.uniform(0.9, 1.1)
    vib_scale = 0.05
    pres_loc = np.random.uniform(48, 52)
    pres_scale = 1.0
    temp_loc = np.random.uniform(38, 42)
    temp_scale = 0.5
    
    # Generate  values for each sensor
    vibration = np.random.normal(loc=vib_loc, scale=vib_scale, size=N_SAMPLES)
    pressure = np.random.normal(loc=pres_loc, scale=pres_scale, size=N_SAMPLES)
    temperature = np.random.normal(loc=temp_loc, scale=temp_scale, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    start , end = get_fault_window(mode)
    fault_len = end - start

    # Intensity of noise
    if intensity == "low":
        noise_loc, noise_scale = 2, 1
    else:
        noise_loc, noise_scale = 5, 2.5

    # Inject noise
    vibration[start:end] = np.random.normal(loc= noise_loc , scale= noise_scale , size= fault_len)
    label[start:end] = "vibration_noise"

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
    for mode in ["start" , "random", "recover"]:
            for intensity in ["low", "high"]:
                for _ in range(n_samples):
                    generate_pressure_noise(wid, mode=mode , intensity = intensity)   ; wid += 1
                    generate_temperature_noise(wid, mode=mode , intensity = intensity) ; wid += 1
                    generate_vibration_noise(wid, mode=mode , intensity = intensity) ; wid += 1
    print(f" All sensor faults generated. Metadata saved at {METADATA_FILE}")    
# Run the generation
if __name__ == "__main__":
    generate_sensor_fault_noise_dataset(n_samples=25)                        
