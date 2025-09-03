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


ROOT_DIR = "synthetic_data"
METADATA_FILE = os.path.join(ROOT_DIR, "metadata.csv")
os.makedirs(ROOT_DIR, exist_ok=True)
if not os.path.exists(METADATA_FILE):
    pd.DataFrame(columns=["sample_id", "fault_type", "mode", "intensity" ,"file_path"]).to_csv(METADATA_FILE, index=False)

# save data with metadata
def save_with_metadata(df, fault_type,  mode, intensity , uid):
    out_dir = os.path.join(ROOT_DIR, "faults", fault_type,f"{intensity}",  mode)
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

# function to get the fault window based on mode
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

# Leak Fault
def generate_leak_fault(sample_id, mode="random" , intensity="high"):
    time = np.arange(N_SAMPLES)

    # scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=0.02, size=N_SAMPLES)
    pressure  = np.random.normal(loc=np.random.uniform(48, 52), scale=0.5, size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=0.3, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    fault_start, fault_end = get_fault_window(mode)
    fault_len = fault_end - fault_start
    
     # Fault effect depends on intensity
    if intensity == "low":
        pressure_drop = np.linspace(1.0, 5.0, fault_len)
        vibration_rise = np.linspace(0.3, 0.8, fault_len)
    else:  # high
        pressure_drop = np.linspace(5.0, 15.0, fault_len)
        vibration_rise = np.linspace(1.0, 3.0, fault_len)

    # Apply the fault
    pressure[fault_start:fault_end] -= pressure_drop
    vibration[fault_start:fault_end] += vibration_rise
    label[fault_start:fault_end] = "leak"

   # Create a DataFrame with all columns
    df =  pd.DataFrame({
        "time": time,
          "vibration": vibration, 
          "pressure": pressure, 
          "temperature": temperature, 
          "label": label
    })
    # Save the data
    save_with_metadata(df, "leak", mode, intensity , sample_id)


# Blockage Fault
def generate_blockage_fault(sample_id, mode="random" , intensity="high"):
    time = np.arange(N_SAMPLES)

    # scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=0.02, size=N_SAMPLES)
    pressure  = np.random.normal(loc=np.random.uniform(48, 52), scale=0.5, size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=0.3, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    fault_start, fault_end = get_fault_window(mode)
    fault_len = fault_end - fault_start

    # Generate pressure and vibration spike based on intensity 
    if intensity == "low":
      pressure_spike = np.linspace(0, np.random.uniform(5, 10), fault_len)
      vibration_spike = np.linspace(0, np.random.uniform(1, 2), fault_len)
    else:
       pressure_spike = np.linspace(0, np.random.uniform(15, 25), fault_len)
       vibration_spike = np.linspace(0, np.random.uniform(3, 6), fault_len)


    # Apply the fault
    pressure[fault_start:fault_end] += pressure_spike
    vibration[fault_start:fault_end] += vibration_spike
    label[fault_start:fault_end] = "blockage"

    # Create a DataFrame with all columns
    df= pd.DataFrame({
        "time": time, 
        "vibration": vibration, 
        "pressure": pressure, 
        "temperature": temperature, 
        "label": label
    })

    # Save the data
    save_with_metadata(df, "blockage", mode, intensity , sample_id)

# Temperature Fault 
def generate_temperature_fault(sample_id, mode="random" , intensity="high"):
    time = np.arange(N_SAMPLES)

    #scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=0.02, size=N_SAMPLES)
    pressure  = np.random.normal(loc=np.random.uniform(48, 52), scale=0.5, size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=0.3, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    fault_start, fault_end = get_fault_window(mode)
    fault_len = fault_end - fault_start

    # Generate temperature rise and pressure drop based on intensity 
    if intensity == "low":
       temp_rise = np.linspace(0, np.random.uniform(5, 15), fault_len)
       pressure_drop = np.linspace(1, 3, fault_len)
    else:
       temp_rise = np.linspace(0, np.random.uniform(20, 40), fault_len)
       pressure_drop = np.linspace(3, 6, fault_len)

    # Apply the fault
    temperature[fault_start:fault_end] += temp_rise
    pressure[fault_start:fault_end] -= pressure_drop
    label[fault_start:fault_end] = "temperature_fault"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
          "vibration": vibration, 
          "pressure": pressure, 
          "temperature": temperature, 
          "label": label
    })
    # Save the data
    save_with_metadata(df, "temperature_fault",  mode, intensity ,sample_id)

    # Run All Generators
def generate_all_faults(n_samples=25):
    wid = 1
    for mode in [ "start" , "random", "recover" ]:
            for intensity in ["low", "high"]:
                for _ in range(n_samples):
                    generate_leak_fault(wid, mode=mode , intensity= intensity) ; wid += 1
                    generate_blockage_fault(wid , mode=mode , intensity = intensity) ; wid += 1
                    generate_temperature_fault(wid , mode=mode , intensity = intensity)   ; wid += 1
    print(f"✅ All faults generated. Metadata saved at {METADATA_FILE}")


# Run the generation
if __name__ == "__main__":
    generate_all_faults(n_samples=25)
