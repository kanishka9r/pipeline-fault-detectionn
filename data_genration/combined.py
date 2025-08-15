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
    
# Function to generate combined leak and blockage fault    
def generate_combined_leak_blockage(sample_id, fault_name="leak_blockage", mode="random", intensity="high", speed="fast"):
        time = np.arange(N_SAMPLES)

        # Baseline normal signal with scale variation
        vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
        pressure = np.random.normal(loc=np.random.uniform(48, 52), scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
        temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
        label = np.array(["normal"] * N_SAMPLES)
        start, end = get_fault_window(mode)
        fault_len = end - start

        # Leak effect: pressure drop + vibration bump
        pressure_drop = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(15, 25)
        vibration_rise = np.random.uniform(0.5, 1.0) if intensity == "low" else np.random.uniform(2.0, 3.5)
        drop = np.linspace(0, pressure_drop, fault_len) if speed == "slow" else np.full(fault_len, pressure_drop)
        bump = np.linspace(0, vibration_rise, fault_len) if speed == "slow" else np.full(fault_len, vibration_rise)

        # Apply the leak fault
        pressure[start:end] -= drop
        vibration[start:end] += bump

        # Blockage effect: pressure spike (superimposed)
        pressure_spike = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(15, 25)
        spike = np.linspace(0, pressure_spike, fault_len) if speed == "slow" else np.full(fault_len, pressure_spike)

        # Apply the blockage fault
        pressure[start:end] += spike  # spike overrides part of the leak drop
        label[start:end] = fault_name

        # Create a DataFrame with all columns
        df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
         })

         #save CSV with zero-padded sample ID
        save_dir = os.path.join("synthetic_data", "combined_faults", fault_name, f"{intensity}_{speed}", mode)
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
        df.to_csv(filename, index=False)

# Function to generate combined leak and temperature fault
def generate_combined_leak_temp_fault(sample_id, mode="random", intensity="high", speed="fast"):
    time = np.arange(N_SAMPLES)

    # Baseline normal signal with scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1),scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
    pressure = np.random.normal(loc=np.random.uniform(48, 52),scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42),scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)

    # Fault window
    start, end = get_fault_window(mode)
    fault_len = end - start

    # Intensity levels
    pressure_drop = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(15, 25)
    vibration_rise = np.random.uniform(0.5, 1.0) if intensity == "low" else np.random.uniform(2.0, 3.5)
    temp_rise = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(20, 30)

    # Speed-based profile
    pressure_change = np.linspace(0, pressure_drop, fault_len) if speed == "slow" else np.full(fault_len, pressure_drop)
    vibration_change = np.linspace(0, vibration_rise, fault_len) if speed == "slow" else np.full(fault_len, vibration_rise)
    temp_change = np.linspace(0, temp_rise, fault_len) if speed == "slow" else np.full(fault_len, temp_rise)

    # Inject faults
    pressure[start:end] -= pressure_change
    vibration[start:end] += vibration_change
    temperature[start:end] += temp_change
    label[start:end] = "leak+temp_fault"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })
  
    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/combined_faults/leak_temp_fault", f"{intensity}_{speed}", mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)        

# Function to generate combined blockage and temperature fault
def generate_combined_blockage_temp_fault(sample_id, mode="random", intensity="high", speed="fast"):
    time = np.arange(N_SAMPLES)

    # Baseline normal signal with scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
    pressure = np.random.normal(loc=np.random.uniform(48, 52) , scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)

    # Fault window
    start, end = get_fault_window(mode)
    fault_len = end - start

    # Intensity levels
    pressure_spike = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(15, 25)
    temp_rise = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(20, 30)

    # Speed-based profile
    pressure_change = np.linspace(0, pressure_spike, fault_len) if speed == "slow" else np.full(fault_len, pressure_spike)
    temp_change = np.linspace(0, temp_rise, fault_len) if speed == "slow" else np.full(fault_len, temp_rise)

    # Inject both faults
    pressure[start:end] += pressure_change
    temperature[start:end] += temp_change
    label[start:end] = "blockage+temp_fault"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/combined_faults/blockage_temp_fault", f"{intensity}_{speed}", mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)    

# Function to generate combined leak and vibration noise fault
def generate_combined_leak_vibration_noise(sample_id, mode="random", intensity="high", speed="fast"):
    time = np.arange(N_SAMPLES)

    # Baseline normal signal with scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
    pressure = np.random.normal(loc=np.random.uniform(48, 52), scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)

    # Fault window
    start, end = get_fault_window(mode)
    fault_len = end - start

    # Leak fault: pressure drop
    pressure_drop = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(15, 25)
    pressure_change = np.linspace(0, pressure_drop, fault_len) if speed == "slow" else np.full(fault_len, pressure_drop)

    # Vibration noise: sudden high fluctuation
    vib_loc = 3 if intensity == "low" else 6
    vib_scale = 1 if intensity == "low" else 2.5
    vib_noise = np.random.normal(loc=vib_loc, scale=vib_scale, size=fault_len)

    #inject both faults
    pressure[start:end] -= pressure_change
    vibration[start:end] = vib_noise
    label[start:end] = "leak+vibration_noise"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/combined_faults/leak_vibration_noise", f"{intensity}_{speed}", mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)

# Function to generate combined blockage and pressure noise
def generate_blockage_pressure_noise_fault(sample_id, mode="random", intensity="high", speed="fast"):
    time = np.arange(N_SAMPLES)

    # Baseline normal signal with scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
    pressure = np.random.normal(loc=np.random.uniform(48, 52), scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)

    # Fault window
    start, end = get_fault_window(mode)
    fault_len = end - start

    # Blockage (pressure spike)
    spike_val = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(15, 25)
    spike = np.linspace(0, spike_val, fault_len) if speed == "slow" else np.full(fault_len, spike_val)
    pressure[start:end] += spike

    # Add pressure noise over spike
    noise_loc = 75 if intensity == "low" else 100
    noise_scale = 5 if intensity == "low" else 15

    # Inject pressure noise
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

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/combined_faults/blockage_pressure_noise", f"{intensity}_{speed}", mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)

# Function to generate leak temperature flat fault
def generate_leak_temperature_flat_fault(sample_id, mode="random", intensity="high", speed="fast"):
    time = np.arange(N_SAMPLES)
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
    pressure = np.random.normal(loc=np.random.uniform(48, 52), scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)

    # Fault window
    start, end = get_fault_window(mode)
    fault_len = end - start

    # Leak: pressure drop + vibration bump
    pressure_drop = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(15, 25)
    vib_rise = np.random.uniform(0.5, 1.0) if intensity == "low" else np.random.uniform(2.0, 3.5)
    pressure[start:end] -= np.linspace(0, pressure_drop, fault_len) if speed == "slow" else pressure_drop
    vibration[start:end] += np.linspace(0, vib_rise, fault_len) if speed == "slow" else vib_rise

    # Temperature flat
    temperature[start:end] = 0.0
    # Label the fault
    label[start:end] = "leak+temperature_flat"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/combined_faults/leak_temperature_flat", f"{intensity}_{speed}", mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)

# Function to generate blockage vibration flat fault
def generate_blockage_vibration_flat_fault(sample_id, mode="random", intensity="high", speed="fast"):
    time = np.arange(N_SAMPLES)
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
    pressure = np.random.normal(loc=np.random.uniform(48, 52), scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)

    # Fault window
    start, end = get_fault_window(mode)
    fault_len = end - start

    # Blockage: pressure spike
    pressure_spike = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(15, 25)
    spike = np.linspace(0, pressure_spike, fault_len) if speed == "slow" else np.full(fault_len, pressure_spike)
    pressure[start:end] += spike

    # Vibration flat
    vibration[start:end] = 0.0

    # Label the fault
    label[start:end] = "blockage+vibration_flat"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/combined_faults/blockage_vibration_flat", f"{intensity}_{speed}", mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)

# Function to generate leak, temperature, and vibration fault
def generate_leak_temp_vibration_fault(sample_id, mode="random", intensity="high", speed="fast"):
    time = np.arange(N_SAMPLES)
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
    pressure = np.random.normal(loc=np.random.uniform(48, 52), scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)

    # Fault window
    start, end = get_fault_window(mode)
    fault_len = end - start

    # Leak (pressure drop + vibration rise)
    pressure_drop = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(15, 25)
    vib_rise = np.random.uniform(0.5, 1.0) if intensity == "low" else np.random.uniform(2.0, 3.5)
    pressure[start:end] -= np.linspace(0, pressure_drop, fault_len) if speed == "slow" else pressure_drop
    vibration[start:end] += np.linspace(0, vib_rise, fault_len) if speed == "slow" else vib_rise

    # Temperature rise
    temp_rise = np.random.uniform(5, 10) if intensity == "low" else np.random.uniform(20, 30)
    temperature[start:end] += np.linspace(0, temp_rise, fault_len) if speed == "slow" else temp_rise

    # Label the fault
    label[start:end] = "leak+temp+vibration_fault"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/combined_faults/leak_temp_vibration_fault", f"{intensity}_{speed}", mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)

# Function to generate temperature noise and pressure flat fault
def generate_temp_noise_pressure_flat(sample_id, mode="random", intensity="high", speed="fast"):
    time = np.arange(N_SAMPLES)

    # Normal signal generation with scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
    pressure = np.random.normal(loc=np.random.uniform(48, 52), scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)

    # Fault window
    start, end = get_fault_window(mode)
    fault_len = end - start

    # Fault parameters
    temp_noise_loc = 60 if intensity == "low" else 100
    temp_noise_scale = 5 if intensity == "low" else 15

    # Inject temperature and pressure faults
    temperature[start:end] = np.random.normal(loc=temp_noise_loc, scale=temp_noise_scale, size=fault_len)
    pressure[start:end] = 0.0

    # Label the fault
    label[start:end] = "temp_noise+pressure_flat"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/combined_faults/temp_noise+pressure_flat", f"{intensity}_{speed}", mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)

# Function to generate vibration flat and temperature flat fault
def generate_vib_flat_temp_flat(sample_id, mode="random", intensity="high", speed="fast"):
    time = np.arange(N_SAMPLES)

    # Normal signal generation with scale variation
    vibration = np.random.normal(loc=np.random.uniform(0.9, 1.1), scale=np.random.uniform(0.03, 0.07), size=N_SAMPLES)
    pressure = np.random.normal(loc=np.random.uniform(48, 52), scale=np.random.uniform(0.8, 1.5), size=N_SAMPLES)
    temperature = np.random.normal(loc=np.random.uniform(38, 42), scale=np.random.uniform(0.3, 0.7), size=N_SAMPLES)
    label = np.array(["normal"] * N_SAMPLES)

    # Fault window
    start, end = get_fault_window(mode)

    # Inject both flat faults
    vibration[start:end] = 0.0
    temperature[start:end] = 0.0

    # Label the fault
    label[start:end] = "vibration_flat+temp_flat"

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        "time": time,
        "vibration": vibration,
        "pressure": pressure,
        "temperature": temperature,
        "label": label
    })

    # Save CSV with zero-padded sample ID
    save_dir = os.path.join("synthetic_data/combined_faults/vibration_flat+temp_flat", f"{intensity}_{speed}", mode)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"sample_{sample_id:03d}.csv")
    df.to_csv(filename, index=False)

# Function to generate all combinations of leak and blockage faults
def generate_all_leak_blockage():
    for mode in ["start", "random", "recover", "snippet", "rear"]:
        for intensity in ["low", "high"]:
            for speed in ["slow", "fast"]:
                for i in range(50):
                    generate_combined_leak_blockage(i, mode=mode, intensity=intensity, speed=speed)
                    generate_combined_leak_temp_fault(i, mode=mode, intensity=intensity, speed=speed)
                    generate_combined_blockage_temp_fault(i, mode=mode, intensity=intensity, speed=speed)
                    generate_combined_leak_vibration_noise(i, mode=mode, intensity=intensity, speed=speed)
                    generate_blockage_pressure_noise_fault(i, mode=mode, intensity=intensity, speed=speed)
                    generate_leak_temperature_flat_fault(i, mode=mode, intensity=intensity, speed=speed)
                    generate_blockage_vibration_flat_fault(i, mode=mode, intensity=intensity, speed=speed)
                    generate_leak_temp_vibration_fault(i, mode=mode, intensity=intensity, speed=speed)
                    generate_temp_noise_pressure_flat(i, mode=mode, intensity=intensity, speed=speed)
                    generate_vib_flat_temp_flat(i, mode=mode, intensity=intensity, speed=speed)

#  generate all combinations
if __name__ == "__main__":
    generate_all_leak_blockage()
    
    