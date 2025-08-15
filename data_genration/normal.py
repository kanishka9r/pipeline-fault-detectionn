import numpy as np
import pandas as pd
import os

# Constants for synthetic data
DURATION = 600  # 10 minutes
SAMPLE_RATE = 1  # 1Hz
N_SAMPLES = DURATION * SAMPLE_RATE

# Directory setup
SAVE_DIR = "synthetic_data/normal" 
os.makedirs(os.path.join(SAVE_DIR, "normal_1"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "normal_2"), exist_ok=True)

# Function to generate normal_1 (stationary normal data)
def generate_normal_1(sample_id):
    time = np.arange(N_SAMPLES)

    # Randomized mean (loc) and standard deviation (scale) for each sensor
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

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        'time': time,
        'vibration': vibration,
        'pressure': pressure,
        'temperature': temperature,
        'label': 'normal_1'
    })

    # Save CSV with zero-padded sample ID
    filename = f"{SAVE_DIR}/normal_1/normal_1_sample_{sample_id:03d}.csv"
    df.to_csv(filename, index=False)


# Function to generate normal_2 (fluctuating normal data)
def generate_normal_2(sample_id):
    time = np.arange(N_SAMPLES)
    
    # Drift in pressure over time (simulates slow fluctuations)
    drift = np.linspace(0, np.random.uniform(-2, 2), N_SAMPLES)

    # Randomized mean (loc) and standard deviation (scale) for each sensor
    vib_loc = np.random.uniform(0.9, 1.1)
    vib_scale = np.random.uniform(0.1, 0.2)
    pres_loc = np.random.uniform(48, 52)
    pres_scale = np.random.uniform(2.0, 4.0)
    temp_loc = np.random.uniform(38, 42)
    temp_scale = np.random.uniform(1.0, 2.5)

    # Generate normally distributed values for each sensor with fluctuations
    vibration = np.random.normal(loc=vib_loc, scale=vib_scale, size=N_SAMPLES) + 0.1 * np.sin(np.linspace(0, 10*np.pi, N_SAMPLES))
    pressure = np.random.normal(loc=pres_loc, scale=pres_scale, size=N_SAMPLES) + drift
    temperature = np.random.normal(loc=temp_loc , scale=temp_scale, size=N_SAMPLES) + 0.2 * np.sin(np.linspace(0, 4*np.pi, N_SAMPLES))

    # Create a DataFrame with all columns
    df = pd.DataFrame({
        'time': time,
        'vibration': vibration,
        'pressure': pressure,
        'temperature': temperature,
        'label': 'normal_2'
    })

    # Save CSV with zero-padded sample ID
    filename = f"{SAVE_DIR}/normal_2/normal_2_sample_{sample_id:03d}.csv"
    df.to_csv(filename, index=False)


# Function to generate both normal_1 and normal_2 samples
def generate_normal_data(n_samples=100):
    for i in range(n_samples):
        generate_normal_1(i)
        generate_normal_2(i)
    print(f" Generated {n_samples} samples each for normal_1 and normal_2")

#Run 
if __name__ == "__main__":
    generate_normal_data(n_samples=100)