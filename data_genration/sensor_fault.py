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
if  os.path.exists(METADATA_FILE):
    os.remove(METADATA_FILE)
pd.DataFrame(columns=["sample_id","category" , "sensor",  "mode" , "intensity",  "file_path"]).to_csv(METADATA_FILE, index=False)

# save data with metadata
def save_with_metadata(df, sensor, mode,intensity ,  uid):
    out_dir = os.path.join(ROOT_DIR, sensor, intensity, mode)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{sensor}_{mode}_{intensity}_{uid:04d}.csv"
    file_path = os.path.join(out_dir, filename)
    df.to_csv(file_path, index=False)
    row = {
        "sample_id": f"{sensor}_{mode}_{intensity}_{uid:04d}",
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
    elif mode == "recover":
        start = random.randint(N_SAMPLES // 4, N_SAMPLES // 3)
        end = min(start + N_SAMPLES // 3, N_SAMPLES)
        return start, end
    else:
        raise ValueError("Invalid mode")


# pressure Noise Fault
def generate_pressure_noise(sample_id, mode="start" , intensity="high"):
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
        noise_loc = np.random.uniform(58,62)
        noise_scale = np.random.uniform(4,6)
    else:
        noise_loc = np.random.uniform(90,120)
        noise_scale = np.random.uniform(13,17)

    # Inject noise
    if mode == "start":
        # Gradually rise to peak noise
        pressure[start:end] = np.linspace(pres_loc, noise_loc, fault_len) \
                              + np.random.normal(0, noise_scale, fault_len)
    elif mode == "recover":
        # Define rise, peak, fall segments
        rise_len = fault_len // 3
        peak_len = fault_len // 3
        fall_len = fault_len - rise_len - peak_len
    
        # Rise gradually from baseline to peak
        pressure[start:start+rise_len] = np.linspace(pres_loc, noise_loc, rise_len) + \
                                     np.random.normal(0, noise_scale*0.5, rise_len)
    
        # Slight fluctuation around peak (no flat hold)
        pressure[start+rise_len:start+rise_len+peak_len] = np.linspace(noise_loc*0.9, noise_loc, peak_len) + \
                                                       np.random.normal(0, noise_scale, peak_len)
    
        # Fall gradually back to baseline
        pressure[start+rise_len+peak_len:end] = np.linspace(noise_loc, pres_loc, fall_len) + \
                                            np.random.normal(0, noise_scale*0.5, fall_len)

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
def generate_temperature_noise(sample_id, mode="start" , intensity="high"):
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
        noise_loc = np.random.uniform(50,55)
        noise_scale = np.random.uniform(4,6)
    else:
        noise_loc = np.random.uniform(80,100)
        noise_scale = np.random.uniform(13,17)

    # Inject noise 
    if mode == "start":
        # Gradually rise to peak noise
        temperature[start:end] = np.linspace(temp_loc, noise_loc, fault_len) \
                              + np.random.normal(0, noise_scale, fault_len)
    elif mode == "recover":
        # Define rise, peak, fall segments
        rise_len = fault_len // 3
        peak_len = fault_len // 3
        fall_len = fault_len - rise_len - peak_len
    
        # Rise gradually from baseline to peak
        temperature[start:start+rise_len] = np.linspace(temp_loc, noise_loc, rise_len) + \
                                     np.random.normal(0, noise_scale*0.5, rise_len)
    
        # Slight fluctuation around peak (no flat hold)
        temperature[start+rise_len:start+rise_len+peak_len] = np.linspace(noise_loc*0.9, noise_loc, peak_len) + \
                                                       np.random.normal(0, noise_scale, peak_len)
    
        # Fall gradually back to baseline
        temperature[start+rise_len+peak_len:end] = np.linspace(noise_loc, temp_loc, fall_len) + \
                                            np.random.normal(0, noise_scale*0.5, fall_len)

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
def generate_vibration_noise(sample_id, mode="start" , intensity="high"):
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
        noise_loc = np.random.uniform(1,3)
        noise_scale = 1
    else:
        noise_loc = np.random.uniform(3,7)
        noise_scale = np.random.uniform(1,3)

    # Inject noise
    if mode == "start":
        # Gradually rise to peak noise
        vibration[start:end] = np.linspace(vib_loc, noise_loc, fault_len) \
                              + np.random.normal(0, noise_scale, fault_len)
    elif mode == "recover":
        # Define rise, peak, fall segments
        rise_len = fault_len // 3
        peak_len = fault_len // 3
        fall_len = fault_len - rise_len - peak_len

        # Rise gradually from baseline to peak
        vibration[start:start+rise_len] = np.linspace(vib_loc, noise_loc, rise_len) + \
                                     np.random.normal(0, noise_scale*0.5, rise_len)  
        
        # Slight fluctuation around peak (no flat hold)
        vibration[start+rise_len:start+rise_len+peak_len] = np.linspace(noise_loc*0.9, noise_loc, peak_len) + \
                                                       np.random.normal(0, noise_scale, peak_len)   
         
        # Fall gradually back to baseline
        vibration[start+rise_len+peak_len:end] = np.linspace(noise_loc, vib_loc, fall_len) + \
                                            np.random.normal(0, noise_scale*0.5, fall_len)
        
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
    for mode in ["start" , "recover"]:
            for intensity in ["low", "high"]:
                for _ in range(n_samples):
                    generate_pressure_noise(wid, mode=mode , intensity = intensity)   ; wid += 1
                    generate_temperature_noise(wid, mode=mode , intensity = intensity) ; wid += 1
                    generate_vibration_noise(wid, mode=mode , intensity = intensity) ; wid += 1
    print(f" All sensor faults generated. Metadata saved at {METADATA_FILE}")    
# Run the generation
if __name__ == "__main__":
    generate_sensor_fault_noise_dataset(n_samples=25)                        
