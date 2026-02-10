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
        if not os.path.exists(folder): continue 
        for file in sorted(os.listdir(folder)): 
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(folder, file))
                seq = df[['vibration', 'pressure', 'temperature']].values
                if seq.shape[0] == seq_len:
                    sequences.append(seq)
    return np.array(sequences)

# Per-sequence normalization
def normalize_sequences(train_seqs, val_seqs):
    # Train data shape: (N, T, C)
    N, T, C = train_seqs.shape
    flat_train = train_seqs.reshape(-1, C)
    scaler = MinMaxScaler()
    flat_train_norm = scaler.fit_transform(flat_train)
    os.makedirs("data/scalers", exist_ok=True) 
    joblib.dump(scaler, "data/scalers/minmax_scaler.pkl")
    train_norm = flat_train_norm.reshape(train_seqs.shape)
    flat_val = val_seqs.reshape(-1, C)
    flat_val_norm = scaler.transform(flat_val)
    val_norm = flat_val_norm.reshape(val_seqs.shape)
    return train_norm, val_norm

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
        _, (hidden, _) = self.encoder_lstm(x) 
        z = self.bottleneck(hidden[-1]) 
        
        # Repeat vector for sequence reconstruction
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        dec_in = self.decoder_input(z)
        dec_out, _ = self.decoder_lstm(dec_in)
        return self.output_layer(dec_out)
    
loss_history=[]
# Train the autoencoder on the normal data
def train_autoencoder(model, data, epochs=50, batch_size=32, lr=1e-4):
    dataset = TensorDataset(torch.from_numpy(data).float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="mean")
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
def compute_sequence_error(model, data_seq):
    model.eval()
    errors = []
    with torch.no_grad():
        # Processing in batches to be safe
        x_tensor = torch.from_numpy(data_seq).float().to(device)
        output = model(x_tensor)
        mse = torch.mean((output - x_tensor)**2, dim=(1, 2))
        errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


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
    np.random.shuffle(data_seq)
    split_idx = int(0.8 * len(data_seq))
    train_data, val_data = data_seq[:split_idx], data_seq[split_idx:]
    train_data, val_data = normalize_sequences(train_data, val_data)
    # Initialize and train autoencoder
    model = LSTMAutoencoder()
    train_autoencoder(model, train_data, epochs=50 , lr=1e-3)

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
    train_errors = compute_sequence_error(model, train_data)
    threshold = determine_threshold(train_errors, method='percentile', value=97)
    # Evaluate false positives on normal validation set
    val_errors = compute_sequence_error(model, val_data)
    num_fp = np.sum(val_errors > threshold)
    total = len(val_errors)
    print(f"Calculated Threshold: {threshold:.6f}")
    print(f"False Positives: {num_fp}/{total} ({100*num_fp/total:.2f}%)")
    print(f"Accuracy on Normal Data: {100*(1-num_fp/total):.2f}%")
