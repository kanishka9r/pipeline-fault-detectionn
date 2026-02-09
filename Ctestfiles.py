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
import joblib

scaler = joblib.load("data/scalers/minmax_scaler.pkl")
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
        pooled = torch.mean(enc_out, dim=1)      
        z = self.bottleneck(pooled)
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_in = self.decoder_input(z)
        dec_out, _ = self.decoder_lstm(dec_in)
        return self.output_layer(dec_out)

# Compute reconstruction error
def reconstruction_error(model, data, device):
    x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        seq_error = torch.mean((out - x) ** 2).item()
    return np.array([seq_error])

def get_normal_errors(model, normal_base, device):
    errors = []
    for sub in ["normal_1", "normal_2"]:
        folder = os.path.join(normal_base, sub)
        for f in os.listdir(folder):
            if f.endswith(".csv"):
                raw = pd.read_csv(os.path.join(folder, f)).iloc[:, 1:4].values
                if raw.shape[0] == 600:
                    data = scaler.transform(raw)
                    err = reconstruction_error(model, data, device)
                    errors.append(err.item())
    return np.array(errors)

def get_errors_from_folder(model, folder, device):
    errors = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".csv"):
                data = pd.read_csv(os.path.join(root, f)).iloc[:, 1:4].values
                if data.shape[0] == 600:
                    err = reconstruction_error(model, data, device)
                    errors.append(err.item())
    return np.array(errors)

#  Metrics + histogram 
def metrics_and_plot(normal_errors, problem_errors, case_type, case_name):
    if len(problem_errors) == 0:
        print(f"Skipping {case_type}/{case_name}, no data.")
        return None

    ks_stat, p_value = ks_2samp(normal_errors, problem_errors)
    labels = np.concatenate([np.zeros(len(normal_errors)),np.ones(len(problem_errors))])
    scores = np.concatenate([normal_errors, problem_errors])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\n{case_type}: {case_name}")
    print(f"KS={ks_stat:.4f}, p={p_value:.4f}, AUROC={auroc:.4f}, Threshold={optimal_threshold:.4f}")
    
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

# Main 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMAutoencoder(input_size=3).to(device)
model.load_state_dict(torch.load('./data/models/autoencoder.pt', map_location=device))
model.eval()
print("Model loaded successfully.")

normal_errors = get_normal_errors(model, "./data/normal", device)
print("\nNormal Errors Stats:")
print("Min:", normal_errors.min())
print("Max:", normal_errors.max())
print("Mean:", normal_errors.mean())
print("Count:", len(normal_errors))

# Problem folders
base_path = './data/problem2/normalized_data'
groups = ['faults', 'combined']

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