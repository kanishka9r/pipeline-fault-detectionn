import torch.nn as nn
import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from torch.utils.data import  DataLoader , TensorDataset , Subset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random


# Constants
data_dir= "data/problems/extracted_faults"  
batch_size = 32
lr = 1e-4
num_epoch = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
patience = 8  # early stopping patience on val loss
seed = 42

# Set random seeds for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# Function to get label from filename
def get_label_from_filename(filename):
    name = Path(filename).stem.lower()
    # Remove trailing _segX_Y if present
    if "_seg" in name:
        name = name.split("_seg")[0]
    return name  # e.g. "fault_leak_low"


# Function to build dataset from folder structure
def build_dataset_from_folder(data_dir):
    data_dir = Path(data_dir)
    filenames = list(data_dir.rglob("*.csv"))
    if len(filenames) == 0:
        raise ValueError(f"No CSV files found in {data_dir}")    
    segments = []
    raw_labels = []


    for path in filenames:
        df = pd.read_csv(path)
        feat = df.iloc[:, :3].values.astype(np.float32)
        if feat.shape != (60, 3):
            continue
        segments.append(feat)
        raw_labels.append(get_label_from_filename(path))

    if len(segments) == 0:
        raise ValueError("No valid segments found ")

    # Stack as before
    X = np.stack(segments)
    encoder = LabelEncoder()
    y_class = np.array(encoder.fit_transform(raw_labels), dtype=np.int64)


    # Convert to tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    y_class_t = torch.tensor(y_class, dtype=torch.long)

    return TensorDataset(X_t, y_class_t,), encoder


  
# Define the CNN-LSTM model with multi-task learning
class CNN_LSTM_MTL(nn.Module):
    def __init__(self, input_size=3, cnn_channels=64, lstm_hidden=256, num_classes=18):
        super().__init__()

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.MaxPool1d(kernel_size=2)
        )

        # LSTM layers
        self.lstm = nn.LSTM(input_size=cnn_channels*2, hidden_size=lstm_hidden,
        num_layers=2 , batch_first=True , bidirectional=True)
        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)
        self.attn = nn.Linear(lstm_hidden * 2, 1)

         
        # Multi-task output layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden*2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, seq_len=60, features=3)
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.cnn(x)         # (B, F_out, T//2)
        x = x.permute(0, 2, 1)  # (B, T//2, F_out)
        lstm_out, _ = self.lstm(x)   # (B, T//2, hidden)
        attn_logits = self.attn(lstm_out)             # (B, T, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, T, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, hidden*2)
        context = self.layer_norm(context)
        return self.classifier(context)

# Dataset loading and splitting
dataset, label_encoder = build_dataset_from_folder(data_dir)
N = len(dataset)
train_idx, val_idx = train_test_split(list(range(N)), test_size=0.2, random_state=42, stratify=[dataset[i][1].item() for i in range(N)])
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
num_classes = len(label_encoder.classes_)
print("Num samples:", N, "Num classes:", num_classes)
labels_list = [dataset[i][1].item() for i in range(len(dataset))]
class_weights_np = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=labels_list
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
print("\nClass Weights (for balanced training):", class_weights_np)

#Model preparation
model = CNN_LSTM_MTL(input_size=3, cnn_channels=64, lstm_hidden=256, num_classes=num_classes).to(device)
cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

# Training loop with early stopping on val combined loss
low_loss = float('inf')
patience_cnt = 0
train_acc_history = []
val_acc_history = []

for epoch in range(1, num_epoch + 1):
    #train model
    model.train()
    train_loss_sum = 0.0
    correct = 0
    total = 0
    
    for X, y_cls, in train_loader:
        X = X.to(device)
        if model.training:
            noise = torch.randn_like(X) * 0.01  # small Gaussian noise
            scale = 1.0 + 0.02 * torch.randn(X.shape[0], 1, X.shape[2], device=device)  # small scale jitter
            X = X * scale + noise
        y_cls = y_cls.to(device)

        optimizer.zero_grad() #setting zero gradient
        logits = model(X) #output from model

        # Compute losses
        loss_cls = cls_loss_fn(logits, y_cls)
        loss_cls.backward() #loss backpropagation
        optimizer.step() #update model parameters

        # Accumulate losses 
        train_loss_sum += loss_cls.item() * X.size(0)


        preds = logits.argmax(dim=1)
        correct += (preds == y_cls).sum().item()
        total += X.size(0)

    # printable parameters
    train_loss_avg = train_loss_sum / len(train_set)
    train_acc = correct / total
    train_acc_history.append(train_acc)
    
    # validate
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad(): # no gradient calculation during validation

        for X, y_cls in val_loader:
            X = X.to(device)
            y_cls = y_cls.to(device)

            logits = model(X) 

            # Compute losses
            loss_cls = cls_loss_fn(logits, y_cls)

            # Accumulate validation losses
            val_loss_sum += loss_cls.item() * X.size(0)
    

            preds = logits.argmax(dim=1)
            val_correct += (preds == y_cls).sum().item()
            val_total += X.size(0)

    # printable parameters
    val_loss_avg = val_loss_sum / len(val_set)
    val_acc = val_correct / val_total
    val_acc_history.append(val_acc)

    #setup learning rate scheduler
    scheduler.step(val_loss_avg)


    # Print epoch results
    print(f"Epoch {epoch}/{num_epoch} | Train loss {train_loss_avg:.4f}  "
          f"| Train acc {train_acc:.4f} | Val loss {val_loss_avg:.4f} | Val acc {val_acc:.4f}")

    # early stopping & save best
    if val_loss_avg < low_loss:
        low_loss = val_loss_avg
        patience_cnt = 0
        os.makedirs("data/models", exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_classes': label_encoder.classes_
        },  'data/models/best_cnn_lstm_mtl.pt')
        print("  Saved best model to data/models/best_cnn_lstm_mtl.pt")

    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print("Early stopping triggered.")
            break

# Final evaluation (load best and print classification report + regression MAE)
saved_model = torch.load('data/models/best_cnn_lstm_mtl.pt', map_location=device , weights_only = False)
model.load_state_dict(saved_model['model_state_dict'])
class_names = list(saved_model['label_classes'])

# Evaluate on validation set
model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for X, y_cls in val_loader:
        X = X.to(device)
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        all_pred.extend(preds)
        all_true.extend(y_cls.cpu().numpy().tolist())

#print classification report and regression MAE
print("\nClassification Report (validation):")
print(classification_report(all_true, all_pred, target_names=class_names))



# Compute confusion matrix
cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Validation)')
plt.show()

# Plot training and validation accuracy
epochs = range(1, len(train_acc_history) + 1)
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_acc_history, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc_history, 'ro-', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
