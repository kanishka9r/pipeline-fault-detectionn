import torch.nn as nn
import torch
import time
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import  DataLoader , TensorDataset , Subset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
from paderborn_loader import load_dataset_with_files, to_fft

# Constants
batch_size = 32
lr = 3e-4
num_epoch = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
patience = 10  # early stopping patience on val loss
seed = 42

# Set random seeds for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Train / Validation split 
x, y, file_ids = load_dataset_with_files("data_genration/pipelinedataset", window_size=2048) 
print("Total windows:", len(x))
x = np.array([to_fft(w) for w in x])
unique_files = np.unique(file_ids)
train_files, val_files = train_test_split(unique_files, test_size=0.2, random_state=42)

# true labels and data
train_idx = np.isin(file_ids, train_files)
val_idx  = np.isin(file_ids, val_files)
x_train = x[train_idx]
x_val   = x[val_idx]
y_train = y[train_idx]
y_val   = y[val_idx]

# file leakage check
file_overlap = np.intersect1d(train_files, val_files)
print("Common files between Train and Val:", len(file_overlap))
if len(file_overlap) == 0:
    print("No file-level leakage detected.")
else:
    print("File leakage detected")
    print(len(file_overlap))

# normalization (train data only)
mean = x_train.mean(axis=(0,1))
std  = x_train.std(axis=(0,1)) + 1e-8
x_train = (x_train - mean) / std
x_val   = (x_val - mean) / std

# Convert to tensors and dataloader
x_train = torch.tensor(x_train, dtype=torch.float32)
x_val   = torch.tensor(x_val, dtype=torch.float32) #weights can be in decimal
y_train = torch.tensor(y_train, dtype=torch.long) #nn.crossentrophyloss() requires int64 which is equal to long
y_val   = torch.tensor(y_val, dtype=torch.long)
train_set = TensorDataset(x_train, y_train)
val_set   = TensorDataset(x_val, y_val)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=batch_size)
num_classes = len(np.unique(y))
print("Num classes:", num_classes)

# Define the CNN model for classification
class CNNClassifier(nn.Module):
    def __init__(self,  num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1) ) # take avg of each channel on time (batch size , 256 , 1) 
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(64, num_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x.squeeze(-1) # (batch size , 256)
        return self.classifier(x)

#move model to device
model = CNNClassifier(num_classes).to(device)

# Class weights (important for imbalance)
class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(y_train.numpy()) ,y=y_train.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# defined criterion , optimizer , scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=5 ,factor=0.5)

# Training loop with early stopping on val loss
best_val_acc = 0.0
patience_cnt = 0
train_acc_history = []
val_acc_history = []
start_time = time.time()

for epoch in range(1, num_epoch + 1):
    epoch_start = time.time()

    #train model
    model.train()
    train_loss_sum = 0.0
    correct = 0
    total = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad() #setting zero gradient
        output = model(x_batch) #output from model
        loss = criterion(output, y_batch) # Compute losses
        loss.backward() #loss backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #prevent gradient exploding
        optimizer.step() #update model parameters
        train_loss_sum += loss.item() * x_batch.size(0) # Accumulate losses 
        preds = output.argmax(dim=1) # maxi value per row
        correct += (preds == y_batch).sum().item() # sum of correct output
        total += x_batch.size(0)

    # printable parameters
    train_loss_avg = train_loss_sum / len(train_set)
    train_acc = correct / total
    train_acc_history.append(train_acc)
    
    # validate
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            val_loss_sum += loss.item() * x_batch.size(0)
            preds = output.argmax(dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total += x_batch.size(0)

    # printable parameters
    val_loss_avg = val_loss_sum / len(val_set)
    val_acc = val_correct / val_total
    val_acc_history.append(val_acc)

    #setup learning rate scheduler
    scheduler.step(val_acc)

    # Print epoch results
    print(f"Epoch {epoch}/{num_epoch} | Train loss {train_loss_avg:.4f}  "
          f"| Train acc {train_acc:.4f} | Val loss {val_loss_avg:.4f} | Val acc {val_acc:.4f}")

    # early stopping & save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_cnt = 0
        torch.save(model.state_dict(), "data_genration/model/best_paderborn_cnn.pt")
        print(" Saved Best Model")
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print("Early stopping triggered.")
            break
    epoch_end = time.time()
    print(f"[EPOCH {epoch}] Total Time: {epoch_end - epoch_start:.2f} seconds\n")    
end_time = time.time()
total_time = end_time - start_time
print(f"Total Training Time: {total_time:.2f} seconds")
print(f"Average Time per Epoch: {total_time / epoch:.2f} seconds")

# Final evaluation (load best and print classification report + regression MAE)
model.load_state_dict(torch.load("data_genration/model/best_paderborn_cnn.pt" , map_location=device))
model.eval()
all_pred = []
all_true = []
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        output = model(x_batch)
        preds = output.argmax(dim=1).cpu().numpy()
        all_pred.extend(preds)
        y_batch = y_batch.cpu().numpy()
        all_true.extend(y_batch)
class_names = [ "Healthy", "Outer_fault", "Inner_fault", "Ball_fault"]

#classification report
print("\nClassification Report (Validation):")
print(classification_report(all_true, all_pred, target_names=class_names))

# Compute confusion matrix
cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(15, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
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

print(np.unique(y_train.numpy()))
print(class_names)
for i, name in enumerate(class_names):
    print(i, "->", name)