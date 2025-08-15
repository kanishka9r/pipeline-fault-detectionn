import numpy as np
import pandas as pd
import os
import random

# Constants for synthetic data
DURATION = 600  # 10 minutes
SAMPLE_RATE = 1 # 1Hz
N_SAMPLES = DURATION * SAMPLE_RATE

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
    elif mode == "snippet":
        return 0, N_SAMPLES // 5
    elif mode == "rear":
        start = random.randint(N_SAMPLES // 2, N_SAMPLES - N_SAMPLES // 3)
        return start, N_SAMPLES
    else:
        raise ValueError("Invalid mode")

# Leak Fault
def generate_leak_fault(sample_id, mode="random", intensity="high", speed="fast"):
    time = np.arange(N_SAMPLES)

    # scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
    pressure = np.random.normal(loc=np.random.uniform(48, 52), scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    fault_start, fault_end = get_fault_window(mode)
    fault_len = fault_end - fault_start
    
    # Generate pressure drop and vibration rise based on intensity and speed
    pressure_drop = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(15, 25)
    vibration_rise = np.random.uniform(0.5, 1.0) if intensity == "low" else np.random.uniform(2.0, 3.5)
    drop = np.linspace(0, pressure_drop, fault_len) if speed == "slow" else np.full(fault_len, pressure_drop)
    bump = np.linspace(0, vibration_rise, fault_len) if speed == "slow" else np.full(fault_len, vibration_rise)

    # Apply the fault
    pressure[fault_start:fault_end] -= drop
    vibration[fault_start:fault_end] += bump
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
    mode_dir = os.path.join("synthetic_data/faults/leak", f"{intensity}_{speed}" , mode)
    os.makedirs(mode_dir, exist_ok=True)
    filename = os.path.join(mode_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)


# Blockage Fault
def generate_blockage_fault(sample_id, mode="random", intensity="high", speed="fast"):
    time = np.arange(N_SAMPLES)

    # scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
    pressure = np.random.normal(loc=np.random.uniform(48, 52), scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    fault_start, fault_end = get_fault_window(mode)
    fault_len = fault_end - fault_start

    # Generate pressure spike based on intensity and speed
    pressure_spike = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(15, 25)
    spike = np.linspace(0, pressure_spike, fault_len) if speed == "slow" else np.full(fault_len, pressure_spike)
 
    # Apply the fault
    pressure[fault_start:fault_end] += spike
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
    mode_dir = os.path.join("synthetic_data/faults/blockage", f"{intensity}_{speed}" , mode)
    os.makedirs(mode_dir, exist_ok=True)
    filename = os.path.join(mode_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)


# Temperature Fault 
def generate_temperature_fault(sample_id, mode="random", intensity="high", speed="fast"):
    time = np.arange(N_SAMPLES)

    #scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
    pressure = np.random.normal(loc=np.random.uniform(48, 52), scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)
    fault_start, fault_end = get_fault_window(mode)
    fault_len = fault_end - fault_start

    # Generate temperature rise based on intensity and speed
    temp_rise = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(20, 30)
    temp_change = np.linspace(0, temp_rise, fault_len) if speed == "slow" else np.full(fault_len, temp_rise)

    # Apply the fault
    temperature[fault_start:fault_end] += temp_change
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
    mode_dir = os.path.join("synthetic_data/faults/temperature_fault", f"{intensity}_{speed}", mode)
    os.makedirs(mode_dir, exist_ok=True)
    filename = os.path.join(mode_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)

    # Run All Generators
def generate_all_faults(n_samples=25):
    for mode in ["start", "random", "recover", "snippet" , "rear"]:
        for intensity in ["low", "high"]:
            for speed in ["slow", "fast"]:
                for i in range(n_samples):
                    generate_leak_fault(i, mode=mode, intensity=intensity, speed=speed)
                    generate_blockage_fault(i, mode=mode, intensity=intensity, speed=speed)
                    generate_temperature_fault(i, mode=mode, intensity=intensity, speed=speed)
    print(f" All faults generated: {n_samples * 5 * 2 * 2 * 3} samples (mode × intensity × speed × 3 faults)")

# Run the generation
if __name__ == "__main__":
    generate_all_faults(n_samples=25)
