import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score , roc_curve
from paderborn_loader import load_dataset_with_files, to_fft
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define CNN Autoencoder
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),  #extract info
            nn.BatchNorm1d(32),  #normalize all channels
            nn.ReLU(),  # add non -linearity
            nn.Dropout(0.1),  #randomly switch off neurons

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),   #2nd layers more info
            nn.BatchNorm1d(64),  
            nn.ReLU(),
  
            nn.Flatten(), #(batch size , 64*256)
            nn.Linear(64 * 256, 128),  #compress 64*256 to 128
            nn.ReLU()
        )

        self.decoder_input = nn.Linear(128, 64 * 256) #expand again

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 2, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  #change shape
        z = self.encoder(x)
        out = self.decoder_input(z)
        out = out.view(-1, 64, 256)  #change shape
        out = self.decoder(out)
        return out.permute(0, 2, 1) 
    
loss_history=[]  # to store loss

# Train the autoencoder on the normal data
def train_autoencoder(model, data, epochs=50, batch_size=32, lr=1e-4):
    dataset = TensorDataset(torch.from_numpy(data).float())  #covert to tensor and then to dataset & returnd tuple
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #break dataset into batches
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #smart gradient adjust
    criterion = nn.MSELoss() #loss function
    loss_history.clear() # clear loss before train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device) # to access data tensordataset return tuple
            noise = torch.randn_like(x) * 0.01
            noisy_x = x + noise
            output = model(noisy_x)
            loss = criterion(output, x) #calculate loss
            optimizer.zero_grad()  #remove old gradient 
            loss.backward() #backprogation to calculate new grad
            optimizer.step() #weights are updated
            total_loss += loss.item() #store loss for each epouch
        epoch_loss = total_loss / len(dataloader) #avg loss per epoch
        loss_history.append(epoch_loss);     
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")    

# Compute reconstruction error on normal data
def compute_sequence_error(model, data_seq, batch_size=128):
    model.eval()
    dataset = TensorDataset(torch.from_numpy(data_seq).float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #to preserve seq
    errors = []
    with torch.no_grad():  #no gradient calculation
        for batch in dataloader: 
            x = batch[0].to(device)
            output = model(x)
            mse = torch.mean((output - x) ** 2, dim=(1,2)) # per batch loss shape [batch_size ,]
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)

# main process
if __name__ == "__main__":

    x, y, file_ids = load_dataset_with_files("data_genration/pipelinedataset",window_size=2048 )
    # Boolean mask for healthy windows
    healthy_mask = (y == 0)
    # Unique healthy files
    healthy_files = np.unique(file_ids[healthy_mask])
    #shuffle files
    np.random.shuffle(healthy_files)
    # split the data
    split_idx = int(0.8 * len(healthy_files))
    train_files = healthy_files[:split_idx]
    val_files   = healthy_files[split_idx:]
    # File-level masks
    train_mask = np.isin(file_ids, train_files) & healthy_mask
    val_mask   = np.isin(file_ids, val_files) & healthy_mask
    train_raw = x[train_mask]
    val_raw   = x[val_mask]
    # Convert to FFT
    train_data = np.array([to_fft(w) for w in x[train_mask]])
    val_data   = np.array([to_fft(w) for w in x[val_mask]])
     
    print("Train shape:", train_data.shape)
    print("Val shape:", val_data.shape)

    # normalize the train data
    mean = train_data.mean(axis=(0,1))
    std  = train_data.std(axis=(0,1)) + 1e-8 # to avoid divide by 0
    train_data = (train_data - mean) / std # standardization
    val_data   = (val_data - mean) / std #prevent data leakage
    print("Number of healthy files:", len(healthy_files))

    # train on training data & threshold for validating dataset
    model = CNNAutoencoder()
    train_autoencoder(model, train_data, epochs=50, lr=1e-3)
    val_errors = compute_sequence_error(model, val_data)
    val_errors = np.log(1 + val_errors)
    threshold_unsup = np.percentile(val_errors, 95)

    # ===== EVALUATE ON FULL DATASET =====
    X_fft = np.array([to_fft(w) for w in x])
    X_norm = (X_fft - mean) / std
    all_errors = compute_sequence_error(model, X_norm)
    all_errors = np.log1p(all_errors)
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