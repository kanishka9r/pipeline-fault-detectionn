import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from Aautoencoder import LSTMAutoencoder

df = pd.read_csv("reconstruction_metrics.csv")
low = df[df["case_name"].str.contains("_low", case=False)]["optimal_threshold"].mean()
print(low)
high = df[df["case_name"].str.contains("_high", case=False)]["optimal_threshold"].mean()
print(high)

# Function to get structured label from file path
def get_structured_label(file_path):
    p = Path(file_path).resolve()
    parts = p.parts
    if "normalized_data" not in parts:
        raise ValueError("Path does not contain 'normalized_data' directory")
    idx = parts.index("normalized_data")
    category = parts[idx + 1]
    if category == "faults":
        fault_type = parts[idx + 2]
        return f"fault_{fault_type}"
    elif category ==  "sensor_fault":
        sensor = parts[idx + 2]
        return f"sensor_{sensor}"
    elif category == "combined":
        combo = parts[idx + 2]
        return f"combined_{combo}"
    else:
        raise ValueError(f"Unknown category in path: {category}")

# Function to compute reconstruction error
def compute_reconstruction_error(model, data):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1,600,3)
        out = model(x)
        seq_error = torch.mean((out - x) ** 2).item()
    return seq_error
    
def normalize_name(s):
    return s.replace("+", "_").replace(" ", "_").lower()
                
# Function to extract anomaly segments 
def extract_anomaly_segments(model, base_dir, save_dir, low_th, high_th , segment_length=60):
    os.makedirs(save_dir, exist_ok=True)
    low_save_dir = os.path.join(save_dir, "low_faults")
    high_save_dir = os.path.join(save_dir, "high_faults")
    os.makedirs(low_save_dir, exist_ok=True)
    os.makedirs(high_save_dir, exist_ok=True)
    count_low, count_high = 0, 0

    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            path = os.path.join(root, file)
            df = pd.read_csv(path)
            if df.shape[0] != 600 :
                continue
            if df.shape[1] < 4:
                continue
            data = df.iloc[:, 1:4].values  # (600,3)
            label_name = get_structured_label(path)
            src_name = Path(path).stem
            num_segments = data.shape[0] 
            for i in range(num_segments):
                seg = data[i*segment_length:(i+1)*segment_length]
                err = compute_reconstruction_error(model, seg)

                if err <= low_th:
                    continue
                level = "high" if err > high_th else "low"
                save_root = high_save_dir if level == "high" else low_save_dir
                name = f"{label_name}_{src_name}_{level}_seg{i+1}.csv"
                out = os.path.join(save_root, name)
                pd.DataFrame(seg, columns=["vibration", "pressure", "temperature"]).to_csv(out, index=False)

                if level == "low":
                    count_low += 1
                else:
                    count_high += 1

    print(f"\nTotal LOW segments saved: {count_low}")
    print(f"Total HIGH segments saved: {count_high}")
            

#  Main 
if __name__ == "__main__":
    model = LSTMAutoencoder(input_size=3, hidden_size=64, latent_size=32)
    model.load_state_dict(torch.load("data/models/autoencoder.pt"))
    model.eval()
    extract_anomaly_segments(model=model, base_dir="data/problem2/normalized_data", save_dir="data/problem2/extracted_faults", low_th=low, high_th=high , segment_length=60)
    

   