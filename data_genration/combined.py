import numpy as np
import pandas as pd
import os
import random

# Constants for synthetic data
DURATION = 600  # 10 minutes
SAMPLE_RATE = 1 # 1Hz
N_SAMPLES = DURATION * SAMPLE_RATE
random.seed(42)
np.random.seed(42)

ROOT_DIR = "synthetic_data"
METADATA_FILE = os.path.join(ROOT_DIR, "combined_metadata.csv")
os.makedirs(ROOT_DIR, exist_ok=True)            
if not os.path.exists(METADATA_FILE):
    pd.DataFrame(columns=["sample_id", "fault_type", "intensity" , "mode", "file_path"]).to_csv(METADATA_FILE, index=False)

# save data with metadata
def save_with_metadata(df, fault_type, mode, intensity ,uid):
    out_dir = os.path.join(ROOT_DIR, "combined_faults")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{fault_type}_{mode}_{intensity}_{uid:04d}.csv"
    file_path = os.path.join(out_dir, filename)
    df.to_csv(file_path, index=False)
    row = {
        "sample_id": f"{fault_type}_{mode}_{uid:04d}",
        "fault_type": fault_type,
        "mode": mode,
        "intensity": intensity,
        "file_path": file_path
    }
    pd.DataFrame([row]).to_csv(METADATA_FILE, mode="a", header=False, index=False)

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
    
# Function to generate combined leak and blockage fault    
def generate_combined_leak_blockage(sample_id, intensity = "high ", mode="random"):
    time = np.arange(N_SAMPLES)

    # scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=0.05, size=N_SAMPLES)
    pressure  = np.random.normal(loc=np.random.uniform(48, 52), scale=1.0, size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=0.5, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    start, end = get_fault_window(mode)
    fault_len = end - start

    # Intensity levels
    if intensity == "low":
        leak_drop = np.random.uniform(5, 12)
        vib_rise = np.random.uniform(1, 2)
        blockage_spike = np.random.uniform(5, 12)
    else: 
        leak_drop = np.random.uniform(15, 25)
        vib_rise = np.random.uniform(2, 5)
        blockage_spike = np.random.uniform(15, 25)

    # Apply the fault
    pressure[start:end] -= leak_drop
    vibration[start:end] += vib_rise
    pressure[start:end] += blockage_spike
    label[start:end] = "leak+blockage"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
    "time": time,
    "vibration": vibration,
    "pressure": pressure,
    "temperature": temperature,
    "label": label
    })

    #save data
    save_with_metadata( df , "leak+blockage", mode , intensity , sample_id)


# Function to generate combined blockage and pressure noise
def generate_blockage_pressure_noise_fault(sample_id, mode="random", intensity="high"):
    time = np.arange(N_SAMPLES)

    # scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=0.05, size=N_SAMPLES)
    pressure  = np.random.normal(loc=np.random.uniform(48, 52), scale=1.0, size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=0.5, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    start, end = get_fault_window(mode)
    fault_len = end - start

   # Intensity levels
    if intensity == "low":
        spike_val = np.random.uniform(5, 12)
        noise_scale = 5
    else:  # high
        spike_val = np.random.uniform(15, 25)
        noise_scale = 12

    # Apply the fault
    pressure[start:end] += spike_val
    noise_loc = pressure[start:end].mean()
    pressure[start:end] += np.random.normal(loc=noise_loc, scale=noise_scale, size=fault_len)
    label[start:end] = "blockage+pressure_noise"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    #save data
    save_with_metadata(df , " blockage+pressure" , intensity , mode , sample_id)


# Function to generate leak, temperature, and vibration fault
def generate_leak_temp_vibration_fault(sample_id, mode="random", intensity="high"):
    time = np.arange(N_SAMPLES)

    # scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=0.05, size=N_SAMPLES)
    pressure  = np.random.normal(loc=np.random.uniform(48, 52), scale=1.0, size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=0.5, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    start, end = get_fault_window(mode)
    fault_len = end - start

     # Intensity levels
    if intensity == "low":
        leak_drop = np.random.uniform(5, 12)
        vib_rise = np.random.uniform(0.5, 1.5)
        temp_rise = np.random.uniform(5, 10)
    else:  
        leak_drop = np.random.uniform(15, 25)
        vib_rise = np.random.uniform(2, 4)
        temp_rise = np.random.uniform(15, 25)

    # Apply the faults
    pressure[start:end] -= leak_drop
    vibration[start:end] += vib_rise
    temperature[start:end] += temp_rise
    label[start:end] = "leak+temp+vibration"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save data
    save_with_metadata(df , "leak+temp+vib" , intensity , mode , sample_id)


# Function to generate all combinations of leak and blockage faults
def generate_all_leak_blockage():
    wid = 1
    for mode in [ "start" ,"random", "recover"]:
            for intensity in ["low", "high"]:
                for i in range(25):
                    generate_combined_leak_blockage(wid, mode=mode , intensity = intensity) ; wid += 1
                    generate_leak_temp_vibration_fault(wid , mode =mode , intensity= intensity) ; wid +=1
                    generate_blockage_pressure_noise_fault(wid, mode=mode , intensity = intensity) ; wid += 1

#  generate all combinations
if __name__ == "__main__":
    generate_all_leak_blockage()
    
    