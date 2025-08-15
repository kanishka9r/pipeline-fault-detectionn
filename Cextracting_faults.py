import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from Aautoencoder import LSTMAutoencoder

# Function to get structured label from file path
def get_structured_label(file_path):
    p = Path(file_path).resolve()
    parts = p.parts
    if "processed" not in parts:
        raise ValueError("Path does not contain 'processed' directory")
    idx = parts.index("processed")
    category = parts[idx + 2]
    if category == "faults":
        fault_type = parts[idx + 3]
        return f"fault_{fault_type}"
    elif category in ("sensor_faults", "sensor_fault"):
        sensor = parts[idx + 3]
        subtype = parts[idx + 4]
        return f"sensor_{sensor}_{subtype}"
    elif category == "combined_faults":
        combo = parts[idx + 3]
        return f"combined_{combo}"
    elif category == "normal":
        return "normal"
    else:
        raise ValueError(f"Unknown category in path: {category}")

# Function to compute reconstruction error
def compute_reconstruction_error(model, data):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(data, dtype=torch.float32).unsqueeze(0) # (1, 600, 3)
        outputs = model(inputs)
        errors = torch.mean((outputs - inputs)**2, dim=2).squeeze(0).numpy()  # shape: (600,)
    return errors

# Function to detect anomalies in all fault types    
def detect_anomalies_in_all_fault_types(model, threshold, base_dir , seq_len=600):
      print("\n--- Anomaly Detection on All Fault Types ---")
      for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv') and 'normal' not in root.lower():
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
                print(f"[{file_path}]  {count}/{seq_len} anomalous sequences")    

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
                error = compute_reconstruction_error(model, data)
                # Create binary mask for anomalies
                anomaly_mask = error > threshold
                if not np.any(anomaly_mask):
                    print(f"[{file_path}]  Normal (no anomaly segment)")
                    continue
                # Find start of first anomaly segment
                start_idx = np.argmax(anomaly_mask)
                end_idx = min(start_idx + segment_length, data.shape[0])
                segment = data[start_idx:end_idx]

                # Calculate intensity based on error
                min_error_possible = 0.0
                max_error_possible = 0.11
                segment_error = error[start_idx:end_idx]
                intensity_raw = float(np.mean(segment_error))
                intensity = (intensity_raw - min_error_possible) / (max_error_possible - min_error_possible)
                intensity = max(0.0, min(1.0, intensity))  # clip to range

                # Pad segment if shorter than required length
                if segment.shape[0] < segment_length:
                    pad_len = segment_length - segment.shape[0]
                    segment = np.pad(segment, ((0, pad_len), (0, 0)), mode='edge')

                segment_df = pd.DataFrame(segment, columns=['Vibration', 'Temperature', 'Pressure'])
                # Save as CSV with proper header
                label = get_structured_label(file_path)
                # Build and save CSV
                save_name = f"{label}_{count}.csv"
                save_path = os.path.join(save_dir, save_name)
                segment_df.to_csv(save_path, index=False)
                segment_df['intensity'] = intensity
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
    extract_anomaly_segments(model=model,threshold=0.014 , base_dir="data/processed/problems",save_dir="data/processed/anomaly_segments",segment_length=60)

    # Detect anomalies in all fault types
    #detect_anomalies_in_all_fault_types(model, threshold = 0.014 , base_dir='data/processed/anomilies')   