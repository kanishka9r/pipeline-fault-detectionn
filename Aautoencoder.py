import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.model_selection import train_test_split


# Load and combine normal data
def load_and_combine_normal_data(path):
    all_data = []
    for subfolder in ['normal_1', 'normal_2']:
        folder_path = os.path.join(path, subfolder)
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, file))
                 # Only keep vibration, pressure, and temperature columns
                all_data.append(df.iloc[:, 1:4].values)
    # Stack vertically to get a single array of shape (total_samples, 3)
    combined_data = np.vstack(all_data)
    return combined_data

# Normalize the combined data and save as CSV
def normalize_and_save(data, out_path):
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data)
    df_normalized = pd.DataFrame(normalized, columns=['vibration', 'pressure', 'temperature'])
    df_normalized.to_csv(out_path, index=False)
    print(f"Saved normalized data to: {out_path}")
    joblib.dump(scaler, 'data/processed/minmax_scaler.pkl')
    print("Saved MinMaxScaler to: data/processed/minmax_scaler.pkl")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, latent_size=32):
        super(LSTMAutoencoder, self).__init__()
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.bottleneck = nn.Linear(hidden_size, latent_size)
        self.decoder_input = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        # Encode
        enc_out, _ = self.encoder_lstm(x)
        bottleneck = self.bottleneck(enc_out[:, -1, :])  # last time step only
        bottleneck = bottleneck.unsqueeze(1).repeat(1, x.size(1), 1)
        # Decode
        dec_input = self.decoder_input(bottleneck)
        dec_out, _ = self.decoder_lstm(dec_input)
        return dec_out    


# Train the autoencoder on the normal data
def train_autoencoder(model, data, epochs=20, batch_size=32, lr=1e-3):
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")    

# Compute reconstruction error on normal data
def compute_reconstruction_error(model, data):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(data, dtype=torch.float32).to(device)
        outputs = model(inputs)
        errors = torch.mean((outputs - inputs) ** 2, dim= 2 ).cpu().numpy()
    return errors

# Determine threshold for anomaly detection
def determine_threshold(errors, method='percentile', value=95):
    if method == 'percentile':
        return np.percentile(errors, value)
    elif method == 'max':
        return np.max(errors)
    else:
        raise ValueError("Method should be 'percentile' or 'max'")

# run the main process
if __name__ == "__main__":

    data_path = 'data/synthetic/normal'
    out_csv = 'data/processed/normal_combined.csv'

    # Load and normalize normal data
    normal_data = load_and_combine_normal_data(data_path)
    normalize_and_save(normal_data, out_csv)   
    # Load normalized data
    df = pd.read_csv('data/processed/normal_combined.csv')
    full_data = df.values

    # Reshape into sequences of 600 (10 min segments at 1Hz)
    SEQ_LEN = 600
    num_seqs = full_data.shape[0] // SEQ_LEN
    data_seq = full_data[:num_seqs * SEQ_LEN].reshape(num_seqs, SEQ_LEN, 3)
    train_data, val_data = train_test_split(data_seq, test_size=0.2, random_state=42)

    # Initialize and train autoencoder
    model = LSTMAutoencoder()
    train_autoencoder(model, train_data, epochs=20)

    # Save model
    torch.save(model.state_dict(), "autoencoder.pt")

    # Compute reconstruction error on normal data
    errors = compute_reconstruction_error(model, val_data)

    # Set threshold
    flat_errors = errors.flatten()
    threshold = determine_threshold(flat_errors, method='percentile', value=95)
    print("Anomaly threshold set to:", threshold)
    print("Min error:", np.min(errors))
    print("Max error:", np.max(errors))
    print("Mean error:", np.mean(errors))
    print("95th percentile:", np.percentile(errors, 95))

    # Evaluate false positives on normal validation set
    anomalies = flat_errors > threshold
    num_anomalies = np.sum(anomalies)
    total_errors = len(flat_errors)
    print(f"\n False Positives on Normal Validation: {num_anomalies}/{total_errors} ({(num_anomalies / total_errors) * 100:.2f}%)")
    print(f"Accuracy on Clean Data: {(1 - num_anomalies / total_errors) * 100:.2f}%")
    
  


