import numpy as np
import pandas as pd
import os
import random

# Constants for synthetic data
DURATION = 600  # 10 minutes
SAMPLE_RATE = 1 # 1Hz
N_SAMPLES = DURATION * SAMPLE_RATE

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
    elif mode == "snippet":
        return 0, N_SAMPLES // 5
    elif mode == "rear":
        start = random.randint(N_SAMPLES // 2, N_SAMPLES - N_SAMPLES // 3)
        return start, N_SAMPLES
    else:
        raise ValueError("Invalid mode")

# pressure Flat Fault
def generate_pressure_flat(sample_id, mode="random"):
    time = np.arange(N_SAMPLES)
    
    # Scale variation
    vib_loc = np.random.uniform(0.9, 1.1)
    vib_scale = np.random.uniform(0.03, 0.07)
    temp_loc = np.random.uniform(38, 42)
    temp_scale = np.random.uniform(0.3, 0.7)

    # Generate normally distributed values for each sensor
    vibration = np.random.normal(loc=vib_loc, scale=vib_scale, size=N_SAMPLES)
    pressure = np.random.normal(loc=50, scale=1.0, size=N_SAMPLES)
    temperature = np.random.normal(loc=temp_loc, scale=temp_scale, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    start, end = get_fault_window(mode)
    
    # Inject flat fault
    pressure[start:end] = 0.0
    label[start:end] = "pressure_flat"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/sensor_fault/pressure/flat", mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)


# pressure Noise Fault
def generate_pressure_noise(sample_id, mode="random" , intensity="high"):
    time = np.arange(N_SAMPLES)

    # Scale variation
    vib_loc = np.random.uniform(0.9, 1.1)
    vib_scale = np.random.uniform(0.03, 0.07)
    pres_loc = np.random.uniform(48, 52)
    pres_scale = np.random.uniform(0.8, 1.5)
    temp_loc = np.random.uniform(38, 42)
    temp_scale = np.random.uniform(0.3, 0.7)

    # Generate normally distributed values for each sensor
    vibration = np.random.normal(loc=vib_loc, scale=vib_scale, size=N_SAMPLES)
    pressure = np.random.normal(loc=pres_loc, scale=pres_scale, size=N_SAMPLES)
    temperature = np.random.normal(loc=temp_loc, scale=temp_scale, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    start , end = get_fault_window(mode)
    fault_len = end - start

     #Intensity Control 
    noise_loc = 75 if intensity == "low" else 100
    noise_scale = 5 if intensity == "low" else 15

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

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/sensor_fault/pressure/noise", intensity , mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)


# Temperature Flat Fault
def generate_temperature_flat(sample_id, mode="random"):
    time = np.arange(N_SAMPLES)

    # Scale variation
    vib_loc = np.random.uniform(0.9, 1.1)
    vib_scale = np.random.uniform(0.03, 0.07)
    pres_loc = np.random.uniform(48, 52)
    pres_scale = np.random.uniform(0.8, 1.5)

    # Generate normally distributed values for each sensor
    vibration = np.random.normal(loc=vib_loc, scale=vib_scale, size=N_SAMPLES)
    pressure = np.random.normal(loc=pres_loc, scale=pres_scale, size=N_SAMPLES)
    temperature = np.random.normal(loc=40, scale=0.5, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    start , end = get_fault_window(mode)

    # Inject flat fault
    temperature[start:end] = 0.0
    label[start:end] = "temperature_flat"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/sensor_fault/temperature/flat", mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)

# Temperature Noise Fault
def generate_temperature_noise(sample_id, mode="random" , intensity="high"):
    time = np.arange(N_SAMPLES)

    # Scale variation
    vib_loc = np.random.uniform(0.9, 1.1)
    vib_scale = np.random.uniform(0.03, 0.07)
    pres_loc = np.random.uniform(48, 52)
    pres_scale = np.random.uniform(0.8, 1.5)
    temp_loc = np.random.uniform(38, 42)
    temp_scale = np.random.uniform(0.3, 0.7)

    # Generate normally distributed values for each sensor
    vibration = np.random.normal(loc=vib_loc, scale=vib_scale, size=N_SAMPLES)
    pressure = np.random.normal(loc=pres_loc, scale=pres_scale, size=N_SAMPLES)
    temperature = np.random.normal(loc=temp_loc, scale=temp_scale, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    start , end = get_fault_window(mode)
    fault_len = end - start

     # Intensity of noise
    noise_loc = 60 if intensity == "low" else 100
    noise_scale = 5 if intensity == "low" else 15

    # Inject noise
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

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/sensor_fault/temperature/noise", intensity , mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)

# Vibration Flat Fault
def generate_vibration_flat(sample_id, mode="random" ):
    time = np.arange(N_SAMPLES)

    # Scale variation
    pres_loc = np.random.uniform(48, 52)
    pres_scale = np.random.uniform(0.8, 1.5)
    temp_loc = np.random.uniform(38, 42)
    temp_scale = np.random.uniform(0.3, 0.7)
    
    # Generate normally distributed values for each sensor
    vibration = np.random.normal(loc=1.0, scale=0.05, size=N_SAMPLES)
    pressure = np.random.normal(loc=pres_loc, scale=pres_scale, size=N_SAMPLES)
    temperature = np.random.normal(loc=temp_loc, scale=temp_scale, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    start , end = get_fault_window(mode)

    # Inject flat fault
    vibration[start:end] = 0.0
    label[start:end] = "vibration_flat"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/sensor_fault/vibration/flat", mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)

# Vibration Noise Fault
def generate_vibration_noise(sample_id, mode="random" , intensity="high"):
    time = np.arange(N_SAMPLES)

    # Scale variation
    vib_loc = np.random.uniform(0.9, 1.1)
    vib_scale = np.random.uniform(0.03, 0.07)
    pres_loc = np.random.uniform(48, 52)
    pres_scale = np.random.uniform(0.8, 1.5)
    temp_loc = np.random.uniform(38, 42)
    temp_scale = np.random.uniform(0.3, 0.7)
    
    # Generate normally distributed values for each sensor
    vibration = np.random.normal(loc=vib_loc, scale=vib_scale, size=N_SAMPLES)
    pressure = np.random.normal(loc=pres_loc, scale=pres_scale, size=N_SAMPLES)
    temperature = np.random.normal(loc=temp_loc, scale=temp_scale, size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    start , end = get_fault_window(mode)
    fault_len = end - start

    # Intensity of noise
    noise_loc = 3 if intensity == "low" else 6
    noise_scale = 1 if intensity == "low" else 2.5

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

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/sensor_fault/vibration/noise", intensity , mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)


# Generate both flat faults
def generate_sensor_fault_flat_dataset(n_samples=25):
    for mode in ["start", "random", "recover", "snippet" , "rear"]:
        for i in range(n_samples):
            generate_pressure_flat(i, mode=mode)
            generate_temperature_flat(i, mode=mode)
            generate_vibration_flat(i, mode=mode)

# Function to generate noise faults
def generate_sensor_fault_noise_dataset(n_samples=25):
    for mode in ["start", "random", "recover", "snippet", "rear"]:
        for intensity in ["low", "high"]:
            for i in range(n_samples):
                generate_pressure_noise(i, mode=mode, intensity=intensity)   
                generate_temperature_noise(i, mode=mode, intensity=intensity)
                generate_vibration_noise(i, mode=mode, intensity=intensity)
            
# Run the generation
if __name__ == "__main__":
    generate_sensor_fault_flat_dataset(n_samples=25)
    generate_sensor_fault_noise_dataset(n_samples=25)                        
