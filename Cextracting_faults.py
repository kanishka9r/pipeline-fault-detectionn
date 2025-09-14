import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from Aautoencoder import LSTMAutoencoder
from scipy.ndimage import label

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
        subtype = parts[idx + 3]
        return f"sensor_{sensor}_{subtype}"
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

# Function to extract anomaly segments                 
def extract_anomaly_segments(model, threshold, base_dir, save_dir , segment_length=60):
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
 
                # Ensure valid shape
                if df.shape[0] != 600 or df.shape[1] < 4:
                    print(f"[{file_path}] Skipped: Invalid shape {df.shape}")
                    continue
                data = df.iloc[:, 1:4].values  # Vibration, Temp, Pressure
                if data.ndim != 2 or data.shape[1] != 3:
                       print(f"[{file_path}] Skipped: bad shape {data.shape}")
                       continue

                error = compute_reconstruction_error(model, data)
                # Create binary mask for anomalies
                anomaly_mask = error > threshold
                if not np.any(anomaly_mask):
                    print(f"[{file_path}]  Normal (no anomaly segment)")
                    continue

                labeled, num_segments = label(anomaly_mask)

                for seg_id in range(1, num_segments + 1):
                    indices = np.where(labeled == seg_id)[0]
                    start_idx, end_idx = indices[0], indices[-1] + 1
    
                     # Extract fixed-length segment
                    cluster_length = end_idx - start_idx
                    num_subsegments = int(np.ceil(cluster_length / segment_length))

                    for sub_id in range(num_subsegments):
                        sub_start = start_idx + sub_id * segment_length
                        sub_end = min(sub_start + segment_length, end_idx)
                        segment = data[sub_start:sub_end]

                         # Pad segment if shorter than required length
                        if segment.shape[0] < segment_length:
                           pad_len = segment_length - segment.shape[0]
                           segment = np.pad(segment, ((0, pad_len), (0, 0)), mode='edge')
                   
               
                    # Save as CSV with proper header
                    segment_df = pd.DataFrame(segment, columns=['Vibration', 'Temp erature', 'Pressure'])                   
                    # Rebuild relative path under normalized_data
                    rel_path = Path(file_path).relative_to(base_dir).with_suffix("")  # remove .csv
                    save_subdir = os.path.join(save_dir, rel_path.parent)            # keep folder hierarchy
                    os.makedirs(save_subdir, exist_ok=True)

                   # Save with structured name inside the subfolder
                    save_name = f"{get_structured_label(file_path)}_seg{seg_id+1}.csv"  # add segment number
                    save_path = os.path.join(save_subdir, save_name)
                    segment_df.to_csv(save_path, index=False)
                    count += 1

    print(f"\n Total anomaly segments saved as CSV: {count}")


# Main execution
if __name__ == "__main__":
    # Load trained model
    model = LSTMAutoencoder(input_size=3, hidden_size=64, latent_size=32)
    model.load_state_dict(torch.load("data/models/autoencoder.pt"))
    model.eval()

    # Run extractor
    min_error, max_error, mean_error, threshold = np.load("data/scalers/autoencoder_error_stats.npy", allow_pickle=True)
    extract_anomaly_segments(model=model,threshold=threshold , base_dir="data/problems/normalized_data",save_dir="data/problems/extracted_faults",segment_length=60)

    # Detect anomalies in all fault types
    #detect_anomalies_in_all_fault_types(model, threshold = 0.014 , base_dir='data/problems/normalized_data')   