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

ROOT_DIR = os.path.join("..", "data", "problems", "combined")
METADATA_FILE = os.path.join("..", "data", "metadata.csv")
os.makedirs(ROOT_DIR, exist_ok=True)            
if os.path.exists(METADATA_FILE):
    os.remove(METADATA_FILE)
pd.DataFrame(columns=["sample_id", "category" , "fault_type", "intensity" , "mode", "file_path"]).to_csv(METADATA_FILE, index=False)

# save data with metadata
def save_with_metadata(df, fault_type, mode, intensity ,uid):
    out_dir = os.path.join(ROOT_DIR, fault_type, f"{intensity}", mode)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{fault_type}_{mode}_{intensity}_{uid:04d}.csv"
    file_path = os.path.join(out_dir, filename)
    df.to_csv(file_path, index=False)
    row = {
        "sample_id": f"{fault_type}_{mode}_{uid:04d}",
        "category": "combined",
        "fault_type": fault_type,
        "intensity": intensity,
        "mode": mode,
        "file_path": file_path
    }
    pd.DataFrame([row]).to_csv(METADATA_FILE, mode="a", header=False, index=False)

# Function to get the fault window based on mode
def get_fault_window(mode):
    if mode == "start":
        return 0, N_SAMPLES
    elif mode == "recover":
        start = random.randint(N_SAMPLES // 4, N_SAMPLES // 3)
        end = min(start + N_SAMPLES // 3, N_SAMPLES)
        return start, end
    else:
        raise ValueError("Invalid mode")
    
# Function to generate baseline data
def generate_baseline():
    return (
        np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=0.02, size=N_SAMPLES),  # vibration
        np.random.normal(loc=np.random.uniform(48, 52), scale=0.5, size=N_SAMPLES),     # pressure
        np.random.normal(loc=np.random.uniform(38, 42), scale=0.3, size=N_SAMPLES),     # temperature
        np.array(["normal"]*N_SAMPLES)       # label
    )    
    
# Function to generate combined leak and blockage fault    
def generate_combined_leak_blockage(sample_id, intensity = "high", mode="start"):
    time = np.arange(N_SAMPLES)
    vibration , pressure , temperature , label = generate_baseline()
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
    if mode == "start":
    # gradually apply fault from start to end
       pressure[start:end] -=  np.linspace(0, leak_drop, fault_len)  # leak drop
       pressure[start:end] += np.linspace(0, blockage_spike, fault_len) * 0.3  # blockage spike rises
       pressure[start:end] += np.random.normal(0, 0.2, fault_len)
       vibration[start:end] += np.linspace(0, vib_rise, fault_len) + np.random.normal(0, 0.1, fault_len)       # vibration rise
    elif mode == "recover":
        rise_ratio = 0.4
        peak_idx = int(fault_len * rise_ratio)
        fall_len = fault_len - peak_idx

        # Rise to peak
        pressure[start:start+peak_idx] -= np.linspace(0, leak_drop, peak_idx)
        pressure[start:start+peak_idx] += np.linspace(0, blockage_spike, peak_idx) * 0.3
        vibration[start:start+peak_idx] += np.linspace(0, vib_rise, peak_idx) + np.random.normal(0, 0.1, peak_idx)

        # Gradual recovery after peak back to normal, with noise
        pressure[start+peak_idx:end] -= np.linspace(leak_drop, 0, fall_len)
        pressure[start+peak_idx:end] += np.linspace(blockage_spike*0.3, 0, fall_len)
        vibration[start+peak_idx:end] -= np.linspace(vib_rise, 0, fall_len) + np.random.normal(0, 0.1, fall_len)

    label[start:end] = "leak+blockage"

    # After applying all fault changes
    pressure = np.clip(pressure, 0, None)
    vibration = np.clip(vibration, 0, None)
    temperature = np.clip(temperature, 0, None)


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
def generate_blockage_pressure_noise_fault(sample_id, mode="start", intensity="high"):
    time = np.arange(N_SAMPLES)
    vibration , pressure , temperature , label = generate_baseline()
    start, end = get_fault_window(mode)
    fault_len = end - start

   # Intensity levels
    if intensity == "low":
        spike_val = np.random.uniform(8, 15)
        noise_scale = np.random.uniform(1, 3)
    else:  # high
        spike_val = np.random.uniform(20, 30)
        noise_scale = np.random.uniform(3, 16)


    # Apply the fault
    if mode == "start":
    # Gradually apply fault from start to end
       pressure[start:end] += np.linspace(0, spike_val, fault_len)
       pressure[start:end] += np.random.normal(0 , scale=noise_scale, size=fault_len)
    elif mode == "recover":
        rise_ratio = 0.4
        peak_idx = int(fault_len * rise_ratio)
        fall_len = fault_len - peak_idx

        # Rise to peak with noise
        pressure[start:start+peak_idx] += np.linspace(0, spike_val, peak_idx)  # deterministic trend
        pressure[start:start+peak_idx] += np.random.normal(0, noise_scale, peak_idx)  # added noise

        # Fall back to normal smoothly
        fall_trend = np.linspace(pressure[start+peak_idx-1], pressure[start-1], fall_len)
        pressure[start+peak_idx:end] = fall_trend + np.random.normal(0, noise_scale, fall_len)

    label[start:end] = "blockage+pressure noise"

    # After applying all fault changes
    pressure = np.clip(pressure, 0, None)
    vibration = np.clip(vibration, 0, None)
    temperature = np.clip(temperature, 0, None)


    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    #save data
    save_with_metadata(df , "blockage+pressure" , mode , intensity , sample_id)


# Function to generate leak, temperature, and vibration fault
def generate_leak_temp_vibration_fault(sample_id, mode="start", intensity="high"):
    time = np.arange(N_SAMPLES)

    vibration , pressure , temperature , label = generate_baseline()
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
    if mode == "start":
    # gradually apply fault from start to end
       pressure[start:end] -= np.linspace(0, leak_drop, fault_len)
       vibration[start:end] += np.linspace(0, vib_rise, fault_len)
       temperature[start:end] += np.linspace(0, temp_rise, fault_len)
    elif mode == "recover":

        rise_ratio = 0.4
        peak_idx = int(fault_len * rise_ratio)
        fall_len = fault_len - peak_idx

        # Rise to peak with small noise
        pressure[start:start+peak_idx] -= np.linspace(0, leak_drop, peak_idx) + np.random.normal(0, 0.1, peak_idx)
        vibration[start:start+peak_idx] += np.linspace(0, vib_rise, peak_idx) + np.random.normal(0, 0.1, peak_idx)
        temperature[start:start+peak_idx] += np.linspace(0, temp_rise, peak_idx) + np.random.normal(0, 0.1, peak_idx)

         # Fall back to normal smoothly with noise
        pressure[start+peak_idx:end] = np.linspace(pressure[start+peak_idx-1], pressure[start-1], fall_len) + np.random.normal(0, 0.2, fall_len)
        vibration[start+peak_idx:end] = np.linspace(vibration[start+peak_idx-1], vibration[start-1], fall_len) + np.random.normal(0, 0.2, fall_len)
        temperature[start+peak_idx:end] = np.linspace(temperature[start+peak_idx-1], temperature[start-1], fall_len) + np.random.normal(0, 0.2, fall_len)
        
    label[start:end] = "leak+temp+vibration"

    # After applying all fault changes
    pressure = np.clip(pressure, 0, None)
    vibration = np.clip(vibration, 0, None)
    temperature = np.clip(temperature, 0, None)


    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save data
    save_with_metadata(df , "leak+temp+vib" , mode , intensity , sample_id)


# Function to generate all combinations of leak and blockage faults
def generate_all_leak_blockage():
    wid = 1
    for mode in [ "start" , "recover"]:
            for intensity in ["low", "high"]:
                for i in range(25):
                    generate_combined_leak_blockage(wid, mode=mode , intensity = intensity) ; wid += 1
                    generate_leak_temp_vibration_fault(wid , mode =mode , intensity= intensity) ; wid +=1
                    generate_blockage_pressure_noise_fault(wid, mode=mode , intensity = intensity) ; wid += 1
    print(f" All faults generated. Metadata saved at {METADATA_FILE}")
#  generate all combinations
if __name__ == "__main__":
    generate_all_leak_blockage()
    
    