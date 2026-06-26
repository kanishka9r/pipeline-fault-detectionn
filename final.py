import torch
import numpy as np
from paderborn_loader import (load_dataset_with_files,to_fft)
from cnnlstm import CNNLSTMClassifier
from gradcam import GradCAM
import matplotlib.pyplot as plt
from scipy.signal import resample
from cnn import CNNGradCAM

# CONFIG
device = torch.device( "cuda" if torch.cuda.is_available()
    else "cpu"
)
test_files = np.load("data_genration/reqdata/test_files.npy",allow_pickle=True)

class_name = [
    "Healthy",
    "Outer Fault",
    "Inner Fault",
    "Ball Fault"
]

# LOAD MODEL
model = CNNLSTMClassifier(num_classes=4).to(device)
model.load_state_dict(
    torch.load(
        "data_genration/model/best_paderborn_cnn.pt",
        map_location=device
    )
)
model.eval()

# LOAD DATA
x, y, file_ids = load_dataset_with_files("data_genration/pipelinedataset",window_size=2048)
test_idx = np.isin(file_ids,test_files)
x = x[test_idx]
y = y[test_idx]
file_ids = file_ids[test_idx]

new_x = []
for w in x:
    fft_window = to_fft(w)
    new_x.append(fft_window)
x = np.array(new_x)

# NORMALIZATION
mean = np.load(
        "data_genration/reqdata/mean.npy"
    )
std = np.load(
        "data_genration/reqdata/std.npy"
    )
x = (x - mean) / std

# PICK SAMPLE
ball_idx = np.where(y == 3)[0]
sample_idx = ball_idx[0]
actual_class = y[sample_idx]
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

gradcam_model = CNNGradCAM(4).to(device)
gradcam_model.load_state_dict(
    torch.load(
        "data_genration/model/best_gradcam_cnn.pt",
        map_location=device
    )
)
gradcam_model.eval()

# GRADCAM
gradcam = GradCAM(gradcam_model, gradcam_model.features[11])
cam, gradcam_pred = gradcam.generate(sample)

# FFT + GRADCAM OVERLAY
fft_x = sample.cpu().numpy()[0, :, 0]
fft_y = sample.cpu().numpy()[0, :, 1]
cam_resized = resample(cam,len(fft_x))
cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max()- cam_resized.min()+ 1e-8)

plt.figure(figsize=(14,5))
plt.plot(fft_x, label="FFT X")
plt.plot(fft_y, label="FFT Y")
scale = max(np.max(fft_x),np.max(fft_y))
plt.plot(cam_resized * scale , label="GradCAM Importance")
plt.title(f"Prediction: {class_name[pred_class]}")
plt.xlabel("FFT Bin")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()

# IMPORTANT FFT REGION
threshold = 0.75
important_idx = np.where(
    cam_resized > threshold
)[0]
start_bin = int(
    important_idx.min()
)
end_bin = int(
    important_idx.max()
)

 #important freq range   
Fs = 64000
freq_resolution = Fs / 2048
start_freq = start_bin * freq_resolution
end_freq = end_bin * freq_resolution

center_freq_hz = (start_freq + end_freq) / 2
if center_freq_hz < 1000:
    region_text = "Low Frequency Region"
elif center_freq_hz < 5000:
    region_text = "Mid Frequency Region"
else:
    region_text = "High Frequency Region "
print(
    f"Important Frequency Region: "
    f"{start_freq:.0f} Hz - {end_freq:.0f} Hz"
)
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

print("Model Prediction :", class_name[pred_class])
print("GradCAM Prediction:", class_name[gradcam_pred])
print("Actual" , class_name[actual_class])
