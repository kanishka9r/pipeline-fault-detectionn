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
    if "problems" not in parts:
        raise ValueError("Path does not contain 'processed' directory")
    idx = parts.index("problems")
    category = parts[idx + 1]
    if category == "faults":
        fault_type = parts[idx + 2]
        return f"fault_{fault_type}"
    elif category ==  "sensor_fault":
        sensor = parts[idx + 2]
        subtype = parts[idx + 3]
        return f"sensor_{sensor}_{subtype}"
    elif category == "combined_faults":
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
    min_error_possible, max_error_possible, _, _ = np.load("autoencoder_error_stats.npy")
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
                    end_idx = min(start_idx + segment_length, data.shape[0])
                    segment = data[start_idx:end_idx]

                    # Pad segment if shorter than required length
                    if segment.shape[0] < segment_length:
                        pad_len = segment_length - segment.shape[0]
                        segment = np.pad(segment, ((0, pad_len), (0, 0)), mode='edge')


                # Calculate intensity based on error
                    segment_error = error[start_idx:end_idx]
                    denom = max_error_possible - min_error_possible
                    if denom == 0: 
                        denom = 1e-8  # avoid division by zero
                    intensity_per_step = (segment_error - min_error_possible) / (max_error_possible - min_error_possible)
                    intensity_per_step = np.clip(intensity_per_step, 0.0, 1.0)

                # Pad intensity vector if segment was padded
                    if len(intensity_per_step) < segment_length:
                        pad_len = segment_length - len(intensity_per_step)
                        intensity_per_step = np.pad(intensity_per_step, (0, pad_len), mode='edge')
               
                # Save as CSV with proper header
                    structured_label = get_structured_label(file_path)
                    segment_df = pd.DataFrame(segment, columns=['Vibration', 'Temperature', 'Pressure'])
                    segment_df['intensity'] = intensity_per_step
                    base_name = Path(file_path).stem
                    save_name = f"{structured_label}_{base_name}_seg{seg_id}_{count}.csv"
                    save_path = os.path.join(save_dir, save_name)
                    segment_df.to_csv(save_path, index=False)

                    print(f"[{file_path}] Segment saved as {os.path.basename(save_path)}")
                    count += 1
    print(f"\n Total anomaly segments saved as CSV: {count}")


# Main execution
if __name__ == "__main__":
    # Load trained model
    model = LSTMAutoencoder(input_size=3, hidden_size=64, latent_size=32)
    model.load_state_dict(torch.load("autoencoder.pt"))
    model.eval()

    # Run extractor
    min_error, max_error, mean_error, threshold = np.load("autoencoder_error_stats.npy", allow_pickle=True)
    extract_anomaly_segments(model=model,threshold=threshold , base_dir="data/processed/problems",save_dir="data/processed/anomaly_segments",segment_length=60)

    # Detect anomalies in all fault types
    #detect_anomalies_in_all_fault_types(model, threshold = 0.014 , base_dir='data/processed/anomilies')   