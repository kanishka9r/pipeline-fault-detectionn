import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score , roc_curve
from paderborn_loader import load_dataset_with_files, to_fft
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define CNN Autoencoder
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), 
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(64 * 256, 128), 
            nn.ReLU()
        )

        self.decoder_input = nn.Linear(128, 64 * 256)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 2, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        z = self.encoder(x)
        out = self.decoder_input(z)
        out = out.view(-1, 64, 256)
        out = self.decoder(out)
        return out.permute(0, 2, 1) 
    
loss_history=[]
# Train the autoencoder on the normal data
def train_autoencoder(model, data, epochs=50, batch_size=32, lr=1e-4):
    dataset = TensorDataset(torch.from_numpy(data).float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    loss_history.clear()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            output = model(x)
            loss = criterion(output, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(dataloader)
        loss_history.append(epoch_loss);    
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")    

# Compute reconstruction error on normal data
def compute_sequence_error(model, data_seq, batch_size=128):
    model.eval()
    dataset = TensorDataset(torch.from_numpy(data_seq).float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    errors = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            output = model(x)
            mae = torch.mean(torch.abs(output - x), dim=(1,2))
            errors.append(mae.cpu().numpy())
    return np.concatenate(errors)

# main process
if __name__ == "__main__":
    X, y, file_ids = load_dataset_with_files("data_genration/pipelinedataset",window_size=2048 )
    # Boolean mask for healthy windows
    healthy_mask = (y == 0)
    # Unique healthy files
    healthy_files = np.unique(file_ids[healthy_mask])
    np.random.shuffle(healthy_files)
    split_idx = int(0.8 * len(healthy_files))
    train_files = healthy_files[:split_idx]
    val_files   = healthy_files[split_idx:]
    # File-level masks
    train_mask = np.isin(file_ids, train_files) & healthy_mask
    val_mask   = np.isin(file_ids, val_files) & healthy_mask
    train_raw = X[train_mask]
    val_raw   = X[val_mask]
    # Convert to FFT
    train_data = np.array([to_fft(w) for w in X[train_mask]])
    val_data   = np.array([to_fft(w) for w in X[val_mask]])
     
    print("Train shape:", train_data.shape)
    print("Val shape:", val_data.shape)

    # ===== NORMALIZATION (HEALTHY TRAIN ONLY) =====
    mean = train_data.mean(axis=(0,1))
    std  = train_data.std(axis=(0,1)) + 1e-8
    train_data = (train_data - mean) / std
    val_data   = (val_data - mean) / std
    print("Number of healthy files:", len(healthy_files))

    # ===== TRAIN =====
    model = CNNAutoencoder()
    train_autoencoder(model, train_data, epochs=50, lr=1e-3)
    val_errors = compute_sequence_error(model, val_data)
    threshold_unsup = np.percentile(val_errors, 95)
    # ===== EVALUATE ON FULL DATASET =====
    X_fft = np.array([to_fft(w) for w in X])
    X_norm = (X_fft - mean) / std
    all_errors = compute_sequence_error(model, X_norm)
    true_anomaly = (y != 0).astype(int)

    # ===== ROC-based threshold selection =====
    fpr, tpr, thresholds = roc_curve(true_anomaly, all_errors)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    threshold = thresholds[best_idx]
    print("ROC Threshold:", threshold)
    print("Unsupervised Threshold:", threshold_unsup)


    pred_anomaly = (all_errors > threshold).astype(int)
    true_anomaly = (y != 0).astype(int)

    tp = np.sum((pred_anomaly == 1) & (true_anomaly == 1))
    fp = np.sum((pred_anomaly == 1) & (true_anomaly == 0))
    fn = np.sum((pred_anomaly == 0) & (true_anomaly == 1))
    tn = np.sum((pred_anomaly == 0) & (true_anomaly == 0))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    auc = roc_auc_score(true_anomaly, all_errors)
    healthy_errors = all_errors[y == 0]
    fault_errors   = all_errors[y != 0]
    ks_stat, p_value = ks_2samp(healthy_errors, fault_errors)

    print("KS Statistic:", ks_stat)
    print("KS p-value:", p_value)
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)
    print("Precision:", precision)
    print("Recall:", recall)
    print("ROC-AUC:", auc)
    print("Mean Healthy Error:", healthy_errors.mean())
    print("Mean Fault Error:", fault_errors.mean())
    print("Error Ratio (Fault/Healthy):", fault_errors.mean() / healthy_errors.mean())


     
    plt.hist(healthy_errors, bins=50, alpha=0.6, label="Healthy")
    plt.hist(fault_errors, bins=50, alpha=0.6, label="Fault")
    plt.axvline(threshold, color='r', linestyle='--', label="Threshold")
    plt.legend()
    plt.title("Reconstruction Error Distribution")
    plt.show()