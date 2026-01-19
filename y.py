import torch
import torch.nn as nn
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset, Subset
import random
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# =======================
# CONFIG
# =======================
data_dir = "data/problem2/extracted_faults"
batch_size = 32
lr = 1e-4
num_epoch = 50
patience = 8
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# =======================
# LABEL PARSING
# =======================
def get_stage_labels(filename):
    name = Path(filename).stem.lower()
    if "_seg" in name:
        name = name.split("_seg")[0]

    if name.startswith("sensor_"):
        return "sensor_fault", name.replace("sensor_", "")
    elif name.startswith("fault_"):
        return "fault", name.replace("fault_", "")
    elif name.startswith("combined_"):
        return "combined", name.replace("combined_", "")
    else:
        raise ValueError(f"Unknown label: {name}")


# =======================
# DATASET BUILDER
# =======================
def build_dataset_from_folder(data_dir):
    paths = list(Path(data_dir).rglob("*.csv"))
    segments, stage1_labels, stage2_labels = [], [], []

    for p in paths:
        df = pd.read_csv(p)
        x = df.iloc[:, :3].values.astype(np.float32)
        if x.shape != (60, 3):
            continue

        s1, s2 = get_stage_labels(p)
        segments.append(x)
        stage1_labels.append(s1)
        stage2_labels.append(s2)

    X = np.stack(segments)

    enc_stage1 = LabelEncoder()
    enc_stage2 = LabelEncoder()

    y1 = enc_stage1.fit_transform(stage1_labels)
    y2 = enc_stage2.fit_transform(stage2_labels)

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y1, dtype=torch.long),
        torch.tensor(y2, dtype=torch.long)
    )

    print("\nDataset size:", len(dataset))
    print("Stage-1 classes:", enc_stage1.classes_)
    print("Stage-2 classes:", len(enc_stage2.classes_))
    return dataset, enc_stage1, enc_stage2


# =======================
# MODEL
# =======================
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(3, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.attn = nn.Linear(512, 1)
        self.norm = nn.LayerNorm(512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        h, _ = self.lstm(x)
        w = torch.softmax(self.attn(h), dim=1)
        context = torch.sum(w * h, dim=1)
        context = self.norm(context)

        return self.classifier(context)


# =======================
# LOAD DATA
# =======================
dataset, enc_stage1, enc_stage2 = build_dataset_from_folder(data_dir)
N = len(dataset)

train_idx, val_idx = train_test_split(
    np.arange(N),
    test_size=0.2,
    random_state=42,
    stratify=[dataset[i][1].item() for i in range(N)]
)

train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)

train_loader_stage1 = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader_stage1 = DataLoader(val_set, batch_size=batch_size)

# =======================
# LOSS & MODEL
# =======================
num_classes = len(enc_stage1.classes_)
train_labels = [dataset[i][1].item() for i in train_idx]


weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels
)
weights = torch.tensor(weights, dtype=torch.float32).to(device)

model = CNN_LSTM(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=4, factor=0.5
)


# =======================
# TRAIN STAGE-1
# =======================
best_val_loss = float("inf")
patience_cnt = 0

for epoch in range(1, num_epoch + 1):
    model.train()
    train_loss, correct, total = 0, 0, 0

    for X, y1, _ in train_loader_stage1:
        X, y1 = X.to(device), y1.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y1)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y1).sum().item()
        total += X.size(0)

    train_acc = correct / total
    train_loss /= len(train_set)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for X, y1, _ in val_loader_stage1:
            X, y1 = X.to(device), y1.to(device)

            logits = model(X)
            loss = criterion(logits, y1)

            val_loss += loss.item() * X.size(0)
            val_correct += (logits.argmax(1) == y1).sum().item()
            val_total += X.size(0)

    val_acc = val_correct / val_total
    val_loss /= len(val_set)

    scheduler.step(val_loss)

    print(
    f"Epoch {epoch:02d} | "
    f"Train loss {train_loss:.4f} | Train acc {train_acc:.4f} | "
    f"Val loss {val_loss:.4f} | Val acc {val_acc:.4f}"
)


    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_cnt = 0
        os.makedirs("data/models", exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "stage1_classes": enc_stage1.classes_
        }, "data/models/stage1_model.pt")
        print("Saved best Stage-1 model")
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print("Early stopping")
            break


# =======================
# STAGE-2: SENSOR FAULT DATASET
# =======================
SENSOR_CLASS_NAME = "sensor_fault"
sensor_class_id = list(enc_stage1.classes_).index(SENSOR_CLASS_NAME)

sensor_indices = [
    i for i in range(len(dataset))
    if dataset[i][1].item() == sensor_class_id
]

sensor_dataset = Subset(dataset, sensor_indices)

print("\nStage-2 (sensor_fault) samples:", len(sensor_dataset))

# Extract original labels
sensor_labels_raw = [dataset[i][2].item() for i in sensor_indices]

# Get unique class ids used by sensor_fault
sensor_classes = np.unique(sensor_labels_raw)

# Build remap: global → local
stage2_remap = {old: new for new, old in enumerate(sensor_classes)}

# Remap labels
sensor_labels = [stage2_remap[y] for y in sensor_labels_raw]
num_classes_stage2 = len(sensor_classes)
train_idx, val_idx = train_test_split(
    np.arange(len(sensor_indices)),
    test_size=0.2,
    random_state=42,
    stratify=sensor_labels
)

train_set = Subset(sensor_dataset, train_idx)
val_set = Subset(sensor_dataset, val_idx)

train_loader_stage2 = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader_stage2 = DataLoader(val_set, batch_size=batch_size)

sensor_classes = np.unique(sensor_labels)
num_classes_stage2 = len(sensor_classes)

train_stage2_labels = [sensor_labels[i] for i in train_idx]

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes_stage2),
    y=train_stage2_labels
)

weights = torch.tensor(weights, dtype=torch.float32).to(device)

model_stage2 = CNN_LSTM(num_classes=num_classes_stage2).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model_stage2.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=4, factor=0.5
)
# =======================
# TRAIN STAGE-2 (sensor_fault)
# =======================
best_val_loss = float("inf")
patience_cnt = 0

for epoch in range(1, num_epoch + 1):
    model_stage2.train()
    train_loss, correct, total = 0, 0, 0

    for X, _, y_stage2 in train_loader_stage2:
        X, y_stage2 = X.to(device), y_stage2.to(device)

        optimizer.zero_grad()
        logits = model_stage2(X)
        loss = criterion(logits, y_stage2)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y_stage2).sum().item()
        total += X.size(0)

    train_acc = correct / total
    train_loss /= len(train_set)

    model_stage2.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for X, _, y_stage2 in val_loader_stage2:
            X, y_stage2 = X.to(device), y_stage2.to(device)
            logits = model_stage2(X)
            loss = criterion(logits, y_stage2)

            val_loss += loss.item() * X.size(0)
            val_correct += (logits.argmax(1) == y_stage2).sum().item()
            val_total += X.size(0)

    val_acc = val_correct / val_total
    val_loss /= len(val_set)

    scheduler.step(val_loss)
    print(
    f"[Stage-2 Sensor] Epoch {epoch} | "
    f"Train loss {train_loss:.4f} | Train acc {train_acc:.4f} | "
    f"Val loss {val_loss:.4f} | Val acc {val_acc:.4f}"
)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_cnt = 0
        os.makedirs("data/models", exist_ok=True)
        sensor_class_names = enc_stage2.classes_[sensor_classes]

        torch.save({
    "model_state_dict": model_stage2.state_dict(),
    "stage2_classes": sensor_class_names
}, "data/models/stage2_sensor_model.pt")

        print("Saved best Stage-2 sensor model")
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print("Early stopping Stage-2")
            break

# =======================
# STAGE-1 EVALUATION
# =======================
saved = torch.load("data/models/stage1_model.pt", map_location=device)
model.load_state_dict(saved["model_state_dict"])
stage1_classes = list(saved["stage1_classes"])

model.eval()
stage1_preds, stage1_true = [], []

with torch.no_grad():
    for X, y_stage1, _ in val_loader_stage1:
        X = X.to(device)
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().numpy()

        stage1_preds.extend(preds)
        stage1_true.extend(y_stage1.numpy())

print("\nStage-1 Classification Report:")
print(classification_report(stage1_true, stage1_preds, target_names=stage1_classes))

cm = confusion_matrix(stage1_true, stage1_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=stage1_classes,
            yticklabels=stage1_classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Stage-1 Confusion Matrix")
plt.show()

stage1_acc = np.mean(np.array(stage1_preds) == np.array(stage1_true))
print(f"Stage-1 Accuracy: {stage1_acc:.4f}")

# =======================
# STAGE-2 EVALUATION (Sensor Faults)
# =======================
saved = torch.load("data/models/stage2_sensor_model.pt", map_location=device)
model_stage2.load_state_dict(saved["model_state_dict"])
stage2_classes = list(saved["stage2_classes"])

model_stage2.eval()
stage2_preds, stage2_true = [], []

with torch.no_grad():
    for X, _, y_stage2 in val_loader_stage2:
        X = X.to(device)
        logits = model_stage2(X)
        preds = logits.argmax(dim=1).cpu().numpy()

        stage2_preds.extend(preds)
        stage2_true.extend(y_stage2.numpy())

print("\nStage-2 (Sensor) Classification Report:")
print(classification_report(stage2_true, stage2_preds, target_names=stage2_classes))

stage2_acc = np.mean(np.array(stage2_preds) == np.array(stage2_true))
print(f"Stage-2 Accuracy: {stage2_acc:.4f}")

sensor_id = stage1_classes.index("sensor_fault")

correct_both = 0
total_sensor = 0

for s1_t, s1_p, s2_t, s2_p in zip(
        stage1_true, stage1_preds,
        stage2_true, stage2_preds):
    if s1_t == sensor_id:
        total_sensor += 1
        if s1_p == sensor_id and s2_t == s2_p:
            correct_both += 1

end_to_end_acc = correct_both / total_sensor if total_sensor > 0 else 0
print(f"\nEnd-to-End Sensor Fault Accuracy: {end_to_end_acc:.4f}")
