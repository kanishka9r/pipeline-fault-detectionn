import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib.pyplot as plt

# Load normal CSVs as sequences
def load_normal_sequences(path, seq_len=600):
    sequences = []
    for subfolder in ["normal_1", "normal_2"]:
        folder = os.path.join(path, subfolder)
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(folder, file))
                seq = df.iloc[:, 1:4].values
                if seq.shape[0] == seq_len:
                    sequences.append(seq)
    return np.array(sequences)

# Per-sequence normalization
def normalize_sequences(seqs):
    N, T, C = seqs.shape
    flat = seqs.reshape(-1, C)        # (N*600,3)
    scaler = MinMaxScaler()
    flat_norm = scaler.fit_transform(flat)
    os.makedirs("data/scalers", exist_ok=True)
    joblib.dump(scaler, "data/scalers/minmax_scaler.pkl")
    norm_seqs = flat_norm.reshape(N, T, C)
    return norm_seqs

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, latent_size=32):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.bottleneck = nn.Linear(hidden_size, latent_size)
        self.decoder_input = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Encode
        enc_out, _ = self.encoder_lstm(x)
        pooled = torch.mean(enc_out, dim=1)          
        z = self.bottleneck(pooled)
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_in = self.decoder_input(z)
        dec_out, _ = self.decoder_lstm(dec_in)
        return self.output_layer(dec_out)    

loss_history=[]
# Train the autoencoder on the normal data
def train_autoencoder(model, data, epochs=50, batch_size=32, lr=1e-4):
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")
    
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
def compute_sequence_error(model, data_seq):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data_seq, dtype=torch.float32).to(device)
        out = model(x)
        seq_errors = torch.mean((out - x) ** 2, dim=(1,2))  # (num_seq,)
    return seq_errors.cpu().numpy()


# Determine threshold for anomaly detection
def determine_threshold(errors, method='percentile', value=97):
    if method == 'percentile':
        return np.percentile(errors, value)
    elif method == 'max':
        return np.max(errors)
    else:
        raise ValueError("Method should be 'percentile' or 'max'")

# run the main process
if __name__ == "__main__":

    # Load and normalize normal data
    data_seq = load_normal_sequences("data/normal")
    data_seq = normalize_sequences(data_seq)

    # Reshape into sequences of 600 (10 min segments at 1Hz)
    split_idx = int(0.8 * len(data_seq))  # 80% train, 20% val
    train_data, val_data = data_seq[:split_idx], data_seq[split_idx:]
    # Initialize and train autoencoder
    model = LSTMAutoencoder()
    train_autoencoder(model, train_data, epochs=50)

    plt.figure(figsize=(10,5))
    plt.plot(loss_history, label="Training Loss", color='blue')
    plt.title("Autoencoder Training Loss Curve ")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save model
    os.makedirs("data/models", exist_ok=True)
    torch.save(model.state_dict(), "data/models/autoencoder.pt")
  
    # Compute reconstruction error on normal data
    errors = compute_sequence_error(model, val_data)

    # Set threshold
    threshold = np.percentile(errors, 97)
    print("Min seq error:", errors.min())
    print("Max seq error:", errors.max())
    print("Mean seq error:", errors.mean())
    print("97th percentile:", threshold)

    # Evaluate false positives on normal validation set
    num_fp = np.sum(errors > threshold)
    total = len(errors)
    print(f"False Positives: {num_fp}/{total} ({100*num_fp/total:.2f}%)")
    print(f"Accuracy on Normal Data: {100*(1-num_fp/total):.2f}%")
