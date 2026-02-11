import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from Aautoencoder import LSTMAutoencoder
import joblib

# Function to get structured label from file path
def get_structured_label(file_path):
    p = Path(file_path).resolve()
    parts = p.parts
    if "normalized_data" not in parts:
        raise ValueError("Path does not contain 'normalized_data' directory")
    idx = parts.index("normalized_data")
    category = parts[idx + 1].lower()
    if "sensor_fault" in category:
        return "sensor_fault"
    elif "fault" in category:
        return "fault"
    elif "combined" in category:
        return "combined"
    else:
        return category

# Function to compute reconstruction error
def compute_reconstruction_error(model, data , device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device) # (1,600,3)
        out = model(x)
        seq_error = torch.mean((out - x) ** 2).item()
    return seq_error

def load_thresholds(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    threshold_map = dict(zip(df['case_name'], df['optimal_threshold']))
    print(f"Successfully loaded {len(threshold_map)} thresholds.")
    return threshold_map
   
# Function to extract anomaly segments 
def extract_anomaly_segments(model, base_dir, save_dir, device , threshold_map, segment_length=60):
    low_save_dir = os.path.join(save_dir, "low")
    high_save_dir = os.path.join(save_dir, "high")
    os.makedirs(low_save_dir, exist_ok=True)
    os.makedirs(high_save_dir, exist_ok=True)

    count_low, count_high = 0, 0

    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            path = os.path.join(root, file)
            df = pd.read_csv(path)
            if df.shape[0] != 600 or df.shape[1] < 4:
               continue
            p = Path(path)
            parts = p.parts
            idx = parts.index("normalized_data")
            fault_name = parts[idx + 2]
            level = parts[idx + 3] 
            matched_key = f"{fault_name}_{level}"
            if matched_key not in threshold_map:
                print("No threshold for:", matched_key)
                continue
            current_threshold = threshold_map[matched_key]
            case_name = matched_key
            data = df[['vibration', 'pressure', 'temperature']].values # (600,3)
            # sequence-level error 
            err = compute_reconstruction_error(model, data , device)
            if err < (current_threshold): # 20% margin for safety
                continue
            level = "high" if "_high" in case_name else "low"
            save_root = high_save_dir if level == "high" else low_save_dir
            label_name = get_structured_label(path)
            num_segments = data.shape[0] // segment_length  # 10

            for i in range(num_segments):
                seg = data[i*segment_length:(i+1)*segment_length]
                if seg.shape[0] != segment_length:
                    continue
                base = Path(path).stem
                name = f"{label_name}_{base}_seg{i+1}_{level}.csv"
                out = os.path.join(save_root, name)
                pd.DataFrame( seg, columns=["vibration", "pressure", "temperature"]).to_csv(out, index=False)

                if level == "low":
                    count_low += 1
                else:
                    count_high += 1

    print(f"\nTotal LOW segments saved: {count_low}")
    print(f"Total HIGH segments saved: {count_high}")

#  Main 
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold_file = "reconstruction_metrics.csv"
    threshold_map = load_thresholds(threshold_file)
    model = LSTMAutoencoder(input_size=3, hidden_size=64, latent_size=32)
    model.load_state_dict(torch.load("data/models/autoencoder.pt" ,  map_location=device))
    model.to(device)
    model.eval()
    extract_anomaly_segments(model=model, base_dir="data/problem2/normalized_data", save_dir="data/problem2/extracted_faults", device = device , threshold_map=threshold_map , segment_length=60 )
    

   