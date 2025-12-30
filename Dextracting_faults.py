import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from Aautoencoder import LSTMAutoencoder
from scipy.ndimage import label
import matplotlib.pyplot as plt


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
        if data.ndim == 2:  # single sequence
            inputs = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        else:
            inputs = torch.tensor(data, dtype=torch.float32)
        outputs = model(inputs)
        errors = torch.mean((outputs - inputs)**2, dim=2).cpu().numpy()  # shape: (600,)
    return errors.squeeze()

# Function to detect anomalies in all fault types    
def detect_anomalies_in_all_fault_types(model, threshold, base_dir , seq_len=600):
      print("\n--- Anomaly Detection on All Fault Types ---")
      for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv') :
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path ,  usecols=[1,2,3]) #columns are vibration, pressure, temperature
                data = df.values 
                if data.shape[0] < seq_len:
                    print(f"[{file_path}] Skipped: too short ({data.shape[0]} rows)")
                    continue
                num_seq = data.shape[0] // seq_len
                data_trimmed = data[:num_seq * seq_len]
                data_seq = data_trimmed.reshape(num_seq, seq_len, 3) # (num_seq, 600, 3)
                errors = compute_reconstruction_error(model, data_seq)
                anomalies = errors > threshold
                count = np.sum(anomalies)
                print(f"[{file_path}]  {count}/{num_seq*seq_len} anomalous sequences")    

                
# Load thresholds from CSV
def load_thresholds(csv_file):
    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]
    thresholds = {}
    for _, row in df.iterrows():
        # Convert Case Type + Case Name to folder path
        case_type = row['case_type'].lower()  # e.g., 'sensor_fault'
        case_name = row['case_name'].replace('+','_').replace(' ','_').lower()  # e.g., 'leak_blockage'
        key = f"{case_type}/{case_name}"
        thresholds[key] = float(row['optimal_threshold'])
    return thresholds

def get_threshold_for_file(file_path, base_dir, thresholds_dict):
    rel_path = os.path.relpath(file_path, base_dir).replace("\\", "/").lower()
    for key in thresholds_dict:
        if rel_path.startswith(key):
            return thresholds_dict[key]
    # fallback to autoencoder threshold
    return default_threshold

# Function to extract anomaly segments 
def extract_anomaly_segments(model, base_dir, save_dir, segment_length=60 , low_th=0.01, high_th=0.04):
    os.makedirs(save_dir, exist_ok=True)
    low_save_dir = os.path.join(save_dir, "low_faults")
    high_save_dir = os.path.join(save_dir, "high_faults")
    os.makedirs(low_save_dir, exist_ok=True)
    os.makedirs(high_save_dir, exist_ok=True)
    count_low, count_high = 0, 0
    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith('.csv'):
                continue
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)

            if df.shape[0] != 600 or df.shape[1] < 4:
                print(f"[{file_path}] Skipped: Invalid shape {df.shape}")
                continue

            data = df.iloc[:, 1:4].values  # Vibration, Temperature, Pressure
            low_threshold =  low_th
            high_threshold = high_th
            errors = compute_reconstruction_error(model, data)
            # Create two anomaly masks
            low_anomaly_mask = (errors > low_threshold) & (errors <= high_threshold)
            high_anomaly_mask = (errors>high_threshold) 
 
            for mask, level, save_root, in [
                (low_anomaly_mask, "LOW", low_save_dir),
                (high_anomaly_mask, "HIGH", high_save_dir)
            ]:
                if not np.any(mask):
                    continue

                labeled, num_segments = label(mask)
                for seg_id in range(1, num_segments + 1):
                    indices = np.where(labeled == seg_id)[0]
                    start_idx, end_idx = indices[0], indices[-1] + 1
                    cluster_length = end_idx - start_idx
                    num_subsegments = int(np.ceil(cluster_length / segment_length))

                    for sub_id in range(num_subsegments):
                        sub_start = start_idx + sub_id * segment_length
                        sub_end = min(sub_start + segment_length, end_idx)
                        segment = data[sub_start:sub_end]
                        if segment.shape[0] < segment_length:
                            pad_len = segment_length - segment.shape[0]
                            segment = np.pad(segment, ((0, pad_len), (0, 0)), mode='edge')

                        label_name = get_structured_label(file_path)
                        src_name = Path(file_path).stem
                        save_name = f"{label_name}_{src_name}_{level.lower()}_seg{seg_id}_{sub_id+1}.csv"
                        save_path = os.path.join(save_root, save_name)
                        pd.DataFrame(segment, columns=["Vibration", "Pressure", "Temperature"]).to_csv(save_path, index=False)

                        if level == "LOW":
                            count_low += 1
                        else:
                            count_high += 1

    print(f"\n total low-intensity segments saved: {count_low}")
    print(f" total high-intensity segments saved: {count_high}")


#  Main 
if __name__ == "__main__":
    model = LSTMAutoencoder(input_size=3, hidden_size=64, latent_size=32)
    model.load_state_dict(torch.load("data/models/autoencoder.pt"))
    model.eval()

    _, _, _, default_threshold = np.load("data/scalers/autoencoder_error_stats.npy", allow_pickle=True)
    extract_anomaly_segments(model=model,
                             base_dir="data/problem2/normalized_data",
                             save_dir="data/problem2/extracted_faults",
                             segment_length=60,
                             low_th=low,
                             high_th=high)
    

   