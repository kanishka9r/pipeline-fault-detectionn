import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, auc
import csv
import random

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

torch.backends.cudnn.enabled = False  

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, latent_size=32):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.bottleneck = nn.Linear(hidden_size, latent_size)
        self.decoder_input = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        enc_out, _ = self.encoder_lstm(x)
        bottleneck = self.bottleneck(enc_out[:, -1, :])
        bottleneck = bottleneck.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_input = self.decoder_input(bottleneck)
        dec_out, _ = self.decoder_lstm(dec_input)
        return self.output_layer(dec_out)

# Compute reconstruction error
def reconstruction_error(model, data, device):
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device).contiguous()
    with torch.no_grad():
        output = model(data_tensor.unsqueeze(0))
        errors = ((output - data_tensor.unsqueeze(0)) ** 2).mean(dim=2).cpu().numpy().flatten()
    return errors

#  folder scan
def get_errors_from_folder(model, folder, device):
    all_errors = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if not f.endswith(".csv"):
                continue
            path = os.path.join(root, f)
            try:
                # Take only vibration, pressure, temperature
                data = pd.read_csv(path).iloc[:, 1:4].values.astype(np.float32)
                if data.shape[0] == 0:
                    continue
                errs = reconstruction_error(model, data, device)
                if len(errs) > 0:
                    all_errors.append(errs)
            except:
                continue
    if all_errors:
        return np.concatenate(all_errors)
    return np.array([])

#  Metrics + histogram 
def metrics_and_plot(normal_errors, problem_errors, case_type, case_name):
    if len(problem_errors) == 0:
        print(f"Skipping {case_type}/{case_name}, no data.")
        return None

    ks_stat, p_value = ks_2samp(normal_errors, problem_errors)
    labels = np.concatenate([np.zeros_like(normal_errors), np.ones_like(problem_errors)])
    scores = np.concatenate([normal_errors, problem_errors])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\n{case_type}: {case_name}")
    print(f"KS={ks_stat:.4f}, p={p_value:.4f}, AUROC={auroc:.4f}, Threshold={optimal_threshold:.4f}")
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal', color='skyblue')
    plt.hist(problem_errors, bins=50, alpha=0.6, label=case_name, color='salmon')
    plt.axvline(optimal_threshold, color='black', linestyle='--', label=f'Threshold={optimal_threshold:.2f}')
    plt.legend()
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title(f'Reconstruction Error: Normal vs {case_name}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'hist_{case_name.replace("+","_")}.png')
    plt.close()
    
    return {
        'case_type': case_type,
        'case_name': case_name,
        'ks_stat': ks_stat,
        'p_value': p_value,
        'auroc': auroc,
        'optimal_threshold': optimal_threshold,
        'mean_normal_error': np.mean(normal_errors),
        'mean_problem_error': np.mean(problem_errors)
    }

def plot_cdf(normal_errors, all_fault_errors, T_low=0.0085, T_high=0.042):
    # Sort values for CDF
    norm_sorted = np.sort(normal_errors)
    fault_sorted = np.sort(all_fault_errors)
    # Compute CDF values
    norm_cdf = np.arange(len(norm_sorted)) / float(len(norm_sorted))
    fault_cdf = np.arange(len(fault_sorted)) / float(len(fault_sorted))
    plt.figure(figsize=(12,7))
    # Plot Normal CDF
    plt.plot(norm_sorted, norm_cdf, label="Normal CDF", linewidth=2, color='blue')
    # Plot Fault CDF
    plt.plot(fault_sorted, fault_cdf, label="All Faults CDF", linewidth=2, color='red')
    # Threshold Lines
    plt.axvline(T_low, color='green', linestyle='--', linewidth=2,label=f"Low Threshold = {T_low}")
    plt.axvline(T_high, color='black', linestyle='--', linewidth=2,label=f"High Threshold = {T_high}")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("CDF")
    plt.xscale('log')
    plt.title("CDF Comparison: Normal vs All Faults")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.savefig("cdf_normal_vs_faults.png", dpi=300)
    plt.close()
    print("Saved: cdf_normal_vs_faults.png")

# Main 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMAutoencoder(input_size=3).to(device)
model.load_state_dict(torch.load('./data/models/autoencoder.pt', map_location=device))
model.eval()
print("Model loaded successfully.")

# Normal data
normal_data = pd.read_csv('./data/normal/normal_combined.csv').iloc[:, 0:3].values.astype(np.float32)
normal_errors = reconstruction_error(model, normal_data, device)

# Problem folders
base_path = './data/problems/normalized_data'
groups = ['sensor_fault', 'faults', 'combined']

all_results = []
all_fault_errors = []


for group in groups:
    group_path = os.path.join(base_path, group)
    if not os.path.exists(group_path):
        continue

    for fault_type in os.listdir(group_path):
        fault_path = os.path.join(group_path, fault_type)
        if not os.path.isdir(fault_path):
            continue

        for intensity in os.listdir(fault_path):  
            intensity_path = os.path.join(fault_path, intensity)
            if not os.path.isdir(intensity_path):
                continue

            errors = get_errors_from_folder(model, intensity_path, device)
            if errors.ndim > 1:
               errors = errors.flatten()
            if len(errors) == 0:
                print(f"No errors found for {group}/{fault_type}/{intensity}, skipping.")
                continue
        
            case_name = f"{fault_type}_{intensity}"
            result = metrics_and_plot(normal_errors, errors, group , case_name)
            all_fault_errors.extend(errors.tolist())
            if result:
                all_results.append(result)  


# Save CSV
csv_file = 'reconstruction_metrics.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

print(f"\nAll metrics saved to {csv_file}")
plot_cdf(normal_errors, all_fault_errors)
