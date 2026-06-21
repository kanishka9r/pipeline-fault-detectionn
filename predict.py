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

class_name = [
    "Healthy",
    "Outer Fault",
    "Inner Fault",
    "Ball Fault"
]

# LOAD MODEL
model = CNNClassifier(num_classes=4).to(device)
model.load_state_dict(
    torch.load(
        "data_genration/model/best_paderborn_cnn.pt",
        map_location=device
    )
)
model.eval()

# LOAD DATA
x, y, file_ids = load_dataset_with_files(
    "data_genration/pipelinedataset",
    window_size=2048
)
x = np.array([
    to_fft(w)
    for w in x
])

# NORMALIZATION
mean = np.load(
        "data_genration/model/mean.npy"
    )
std = np.load(
        "data_genration/model/std.npy"
    )
x = (x - mean) / std

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
    probs = torch.softmax( output, dim=1
    )
    confidence, pred = torch.max(probs,dim=1
    )
pred_class = pred.item()
confidence = (confidence.item() * 100)

# OUTPUT
print("\nPrediction Result")
print("\nFault:")
print(class_name[pred_class])
print(f"\nConfidence : {confidence:.2f}%")
print("\nClass Probabilities")
for i, p in enumerate(
    probs[0].cpu().numpy()
):
    print(
        f"{class_name[i]} : {p*100:.2f}%"
    )

# GRADCAM
gradcam = GradCAM(
    model,
    model.features[11]
)
cam, pred = gradcam.generate(
    sample
)

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
    f"Prediction: {class_name[pred]}"
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

region_center = (start_bin + end_bin) / 2
if region_center < 100:
    region_text = "Low Frequency Region"
elif region_center < 400:
    region_text = "Mid Frequency Region"
else:
    region_text = "High Frequency Region"

#Output
print(
    f"\nImportant FFT Region: "
    f"{start_bin}-{end_bin}"
)
print("\nExplanation:")
if confidence > 95:
    print(
        "Model is highly confident."
    )
elif confidence > 80:
    print(
        "Model confidence is moderate."
    )
else:
    print(
        "Prediction should be reviewed manually."
    )
print(
    f"CNN focused on FFT region "
    f"{start_bin}-{end_bin}"
    f"({region_text})."
)
if pred_class == 0:
    print(
        "Spectral pattern resembles healthy bearings."
    )
elif pred_class == 1:
    print(
        "Frequency distribution matches Outer Race fault patterns."
    )
elif pred_class == 2:
    print(
        "Frequency distribution matches Inner Race fault patterns."
    )
elif pred_class == 3:
    print(
        " Frequency distribution matches Ball Fault patterns."
    )