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

ROOT_DIR = os.path.join("..", "data", "problems", "faults")
METADATA_FILE = os.path.join("..", "data", "metadata.csv")
os.makedirs(ROOT_DIR, exist_ok=True)
if  os.path.exists(METADATA_FILE):
    os.remove(METADATA_FILE)
pd.DataFrame(columns=["sample_id","category" , "fault_type", "mode", "intensity" ,"file_path"]).to_csv(METADATA_FILE, index=False)

# save data with metadata
def save_with_metadata(df, fault_type,  mode, intensity , uid):
    out_dir = os.path.join(ROOT_DIR, fault_type,f"{intensity}",  mode)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{fault_type}_{mode}_{intensity}_{uid:04d}.csv"
    file_path = os.path.join(out_dir, filename)
    df.to_csv(file_path, index=False)
    row = {
        "sample_id": f"{fault_type}_{mode}_{intensity}_{uid:04d}",
        "category": "fault",
        "fault_type": fault_type,
        "mode": mode,
        "intensity": intensity,
        "file_path": file_path
    }
    pd.DataFrame([row]).to_csv(METADATA_FILE, mode="a", header=False, index=False)

# function to get the fault window based on mode
def get_fault_window(mode):
    if mode == "start":
        return 0, N_SAMPLES
    elif mode == "recover":
        start = random.randint(N_SAMPLES // 4, N_SAMPLES // 3)
        end = min(start + N_SAMPLES // 3, N_SAMPLES)
        return start, end
    else:
        raise ValueError("Invalid mode")
    
def generate_baseline():
    return (
        np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=0.02, size=N_SAMPLES),  # vibration
        np.random.normal(loc=np.random.uniform(48, 52), scale=0.5, size=N_SAMPLES),     # pressure
        np.random.normal(loc=np.random.uniform(38, 42), scale=0.3, size=N_SAMPLES),     # temperature
        np.array(["normal"]*N_SAMPLES)  # label
    )   
        
# Leak Fault
def generate_leak_fault(sample_id, mode="start", intensity="high"):
    time = np.arange(N_SAMPLES)

    vibration, pressure, temperature, label = generate_baseline()
    fault_start, fault_end = get_fault_window(mode)
    fault_len = fault_end - fault_start

    # Generate pressure_drop and vibration_rise based on intensity 
    if intensity == "low":
        max_pressure_drop = 4.0
        max_vibration_rise = 0.6
    else:  # high
        max_pressure_drop = 15.0
        max_vibration_rise = 3.0

    # Jitter
    pressure_jitter = np.random.normal(0, 0.1, fault_len)
    vibration_jitter = np.random.normal(0, 0.05, fault_len)

    if mode == "start":
        # simple ramp up until end
        pressure[fault_start:fault_end] -= np.linspace(0, max_pressure_drop, fault_len) + pressure_jitter
        vibration[fault_start:fault_end] += np.linspace(0, max_vibration_rise, fault_len) + vibration_jitter
        label[fault_start:fault_end] = "leak"

    elif mode == "recover":
       up_len = int(fault_len * 0.4)
       down_len = fault_len - up_len

    # 1. Ramp up (baseline → peak)
       pressure[fault_start:fault_start+up_len] -= np.linspace(0, max_pressure_drop, up_len) + pressure_jitter[:up_len]
       vibration[fault_start:fault_start+up_len] += np.linspace(0, max_vibration_rise, up_len ) + vibration_jitter[:up_len]

    # 2. Ramp down (peak → baseline)
       pressure[fault_start+up_len:fault_end] -= np.linspace(max_pressure_drop, 0, down_len) + pressure_jitter[up_len:]
       vibration[fault_start+up_len:fault_end] += np.linspace(max_vibration_rise, 0, down_len) + vibration_jitter[up_len:]

    # Labels
       label[fault_start:fault_end] = "leak"

    # DataFrame
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save
    save_with_metadata(df, "leak", mode, intensity, sample_id)

# Blockage Fault
def generate_blockage_fault(sample_id, mode="start", intensity="high"):
    time = np.arange(N_SAMPLES)

    vibration, pressure, temperature, label = generate_baseline()
    fault_start, fault_end = get_fault_window(mode)
    fault_len = fault_end - fault_start

    # Intensity-based spikes
    if intensity == "low":
        max_pressure_spike = 6.0
        max_vibration_spike = 1.0
    else:  # high
        max_pressure_spike = 18.0
        max_vibration_spike = 4.0

    # Jitter
    pressure_jitter = np.random.normal(0, 0.5, fault_len)
    vibration_jitter = np.random.normal(0, 0.2, fault_len)

    if mode == "start":
        # Ramp up continuously till end
        pressure[fault_start:fault_end] += np.linspace(0, max_pressure_spike, fault_len) + pressure_jitter
        vibration[fault_start:fault_end] += np.linspace(0, max_vibration_spike, fault_len) + vibration_jitter
        label[fault_start:fault_end] = "blockage"

    elif mode == "recover":
        up_len = int(fault_len * 0.4)
        down_len = fault_len - up_len

        pressure[fault_start:fault_start+up_len] += np.linspace(0, max_pressure_spike, up_len) + pressure_jitter[:up_len]
        vibration[fault_start:fault_start+up_len] += np.linspace(0, max_vibration_spike, up_len) + vibration_jitter[:up_len]

        pressure[fault_start+up_len:fault_end] += np.linspace(max_pressure_spike, 0, down_len) + pressure_jitter[up_len:]
        vibration[fault_start+up_len:fault_end] += np.linspace(max_vibration_spike, 0, down_len) + vibration_jitter[up_len:]

        label[fault_start:fault_end] = "blockage"

    # DataFrame
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save
    save_with_metadata(df, "blockage", mode, intensity, sample_id)


# Temperature Fault 
def generate_temperature_fault(sample_id, mode="start", intensity="high"):
    time = np.arange(N_SAMPLES)

    vibration, pressure, temperature, label = generate_baseline()
    fault_start, fault_end = get_fault_window(mode)
    fault_len = fault_end - fault_start

    # Intensity-based effect
    if intensity == "low":
        max_temp_rise = 6.0
        max_pressure_drop = 0.6
    else:  # high
        max_temp_rise = 30.0
        max_pressure_drop = 5.0 


    # Jitter
    temp_jitter = np.random.normal(0, 0.5, fault_len)
    pressure_jitter = np.random.normal(0, 0.2, fault_len)

    if mode == "start":
        # Ramp up continuously till end
        temperature[fault_start:fault_end] += np.linspace(0, max_temp_rise, fault_len) + temp_jitter
        pressure[fault_start:fault_end] -= np.linspace(0, max_pressure_drop, fault_len) + pressure_jitter
        label[fault_start:fault_end] = "temperature_fault"

    elif mode == "recover":
        up_len = int(fault_len * 0.4)
        down_len = fault_len - up_len

        temperature[fault_start:fault_start+up_len] += np.linspace(0, max_temp_rise, up_len) + temp_jitter[:up_len]
        pressure[fault_start:fault_start+up_len] -= np.linspace(0, max_pressure_drop, up_len) + pressure_jitter[:up_len]

        temperature[fault_start+up_len:fault_end] += np.linspace(max_temp_rise, 0, down_len) + temp_jitter[up_len:]
        pressure[fault_start+up_len:fault_end] -= np.linspace(max_pressure_drop, 0, down_len) + pressure_jitter[up_len:]

        label[fault_start:fault_end] = "temperature_fault"
    # DataFrame
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save
    save_with_metadata(df, "temperature_fault", mode, intensity, sample_id)


    # Run All Generators
def generate_all_faults(n_samples=25):
    wid = 1
    for mode in [ "start" , "recover" ]:
            for intensity in ["low", "high"]:
                for _ in range(n_samples):
                    generate_leak_fault(wid, mode=mode , intensity= intensity) ; wid += 1
                    generate_blockage_fault(wid , mode=mode , intensity = intensity) ; wid += 1
                    generate_temperature_fault(wid , mode=mode , intensity = intensity)   ; wid += 1
    print(f" All faults generated. Metadata saved at {METADATA_FILE}")


# Run the generation
if __name__ == "__main__":
    generate_all_faults(n_samples=25)
