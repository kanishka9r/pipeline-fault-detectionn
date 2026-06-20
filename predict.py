import torch
import numpy as np
from paderborn_loader import (load_dataset_with_files,to_fft)
import random
from model import CNNClassifier
from gradCam import GradCAM
import matplotlib.pyplot as plt
from scipy.signal import resample

# CONFIG
device = torch.device( "cuda" if torch.cuda.is_available()
    else "cpu"
)

CLASS_NAMES = [
    "Healthy",
    "Outer Fault",
    "Inner Fault",
    "Ball Fault"
]

# LOAD MODEL
model = CNNClassifier(
    num_classes=4
).to(device)

model.load_state_dict(
    torch.load(
        "data_genration/model/best_paderborn_cnn.pt",
        map_location=device
    )
)

model.eval()
print("Model Loaded")


# LOAD DATA
x, y, file_ids = load_dataset_with_files(
    "data_genration/pipelinedataset",
    window_size=2048
)
x = np.array([
    to_fft(w)
    for w in x
])
print("Total Samples:", len(x))


# NORMALIZATION
mean = np.load(
        "data_genration/model/mean.npy"
    )
std = np.load(
        "data_genration/model/std.npy"
    )
x = (x - mean) / std
print("Loaded Saved Mean/Std")

# PICK SAMPLE
sample_idx = random.randint(0,len(x)-1)
sample = torch.tensor(
    x[sample_idx],
    dtype=torch.float32
)
sample = sample.unsqueeze(0)
sample = sample.to(device)

# PREDICTION
with torch.no_grad():
    output = model(sample)
    probs = torch.softmax(
        output,
        dim=1
    )
    confidence, pred = torch.max(
        probs,
        dim=1
    )
pred_class = pred.item()
confidence = (
    confidence.item() * 100
)

# OUTPUT
print("Sample Index:", sample_idx)
print("\nPrediction Result")
print(
    "True Label :",
    CLASS_NAMES[y[sample_idx]]
)
print(
    "Predicted  :",
    CLASS_NAMES[pred_class]
)
print(
    f"Confidence : {confidence:.2f}%"
)
print("\nClass Probabilities")
for i, p in enumerate(
    probs[0].cpu().numpy()
):
    print(
        f"{CLASS_NAMES[i]} : {p*100:.2f}%"
    )

# GRADCAM
gradcam = GradCAM(
    model,
    model.features[11]
)
cam, pred = gradcam.generate(
    sample
)
print(
    "Predicted:",
    CLASS_NAMES[pred]
)
print(
    "CAM Shape:",
    cam.shape
)
print(
    "CAM Max:",
    cam.max()
)
print(
    "CAM Min:",
    cam.min()
)

# GRADCAM IMPORTANCE PLOT
plt.figure(figsize=(12,4))
plt.plot(cam)
plt.title(
    "GradCAM Importance"
)
plt.xlabel(
    "Feature Position"
)
plt.ylabel(
    "Importance"
)
plt.grid(True)
plt.show()    


# FFT + GRADCAM OVERLAY
fft_signal = sample.cpu().numpy()[0, :, 0]
cam_resized = resample(
    cam,
    len(fft_signal)
)
cam_resized = (
    cam_resized - cam_resized.min()
) / (
    cam_resized.max()
    - cam_resized.min()
    + 1e-8
)

plt.figure(figsize=(14,5))

plt.plot(
    fft_signal,
    label="FFT Spectrum"
)

plt.plot(
    cam_resized * np.max(fft_signal),
    label="GradCAM Importance"
)

plt.title(
    f"Prediction: {CLASS_NAMES[pred]}"
)

plt.xlabel("FFT Bin")

plt.ylabel("Magnitude")

plt.legend()
plt.grid(True)
plt.show()

# IMPORTANT FFT REGION
threshold = 0.5
important_idx = np.where(
    cam_resized > threshold
)[0]
start_bin = int(
    important_idx.min()
)
end_bin = int(
    important_idx.max()
)
print(
    f"Important FFT Region: "
    f"{start_bin}-{end_bin}"
)