# Pipeline Bearing Fault Detection

A machine learning pipeline and interactive dashboard for industrial bearing health monitoring.

This project implements an end-to-end intelligent fault diagnosis pipeline that processes vibration signals, extracts frequency-domain information, and applies deep learning models to learn complex degradation patterns in rotating machinery. The framework is designed to support accurate and interpretable machine condition monitoring.

---

## Features

- **Interactive Cloud Dashboard:** A fully responsive, modern Streamlit web application for real-time visualization of vibration signals and fault predictions.
- **Deep CNN-LSTM Architecture:** Uses a Convolutional Neural Network combined with Long Short-Term Memory (LSTM) layers for highly accurate time-series classification.
- **Explainable AI (Grad-CAM):** Uses a parallel pure CNN model to visualize which specific frequencies the AI is looking at to make its predictions, ensuring transparent decision-making.
- **Zero Data Leakage:** Strict file-level train/validation/test splitting guarantees the model generalizes reliably to unseen machinery.
- **Lightweight Deployment:** Uses a compressed `.npz` demo dataset to bypass GitHub storage limits while maintaining full functionality on Streamlit Cloud.

---

## Dataset Used

The system is optimized for the **Paderborn University (PU) Dataset**, focusing on high-frequency vibration data:
- **Healthy State:** Baseline operating conditions without defects.
- **Artificial & Real Damage:** Classifies defects into **Outer Fault**, **Inner Fault**, and **Ball Fault**.
- **Data Format:** Raw `.mat` (MATLAB) files, which are dynamically sliced into 2048-sample windows.

---

##  Model Architecture

The project processes raw vibration streams through a two-stage pipeline:

### 1. Data Engineering & Preprocessing
- **Segmentation:** Continuous streams are partitioned into discrete 2048-sample windows.
- **Envelope Analysis (Hilbert Transform):** Applied to the raw signals to extract the amplitude envelope, isolating fault-related impulses from the high-frequency carrier signal.
- **Fast Fourier Transform (FFT):** Converts raw time-domain vibrations into the frequency domain, where fault harmonics are most prominent.
- **Log-Scaling & Normalization:** Normalizes the dynamic range using standard scaling (`mean.npy` / `std.npy`) for stable neural network convergence.

### 2. Fault Classification (Deep 1D-CNN)
- **Architecture:** Hybrid 1D-CNN and LSTM model with three convolutional layers, Batch Normalization, and Dropout (0.4) for robust feature learning and regularization.
- **Optimization:** Trained using Adam optimizer, ReduceLROnPlateau scheduler, balanced class weights, and gradient clipping for stable learning.
- **Outputs:** Softmax probability distribution across four bearing conditions: Healthy, Outer Fault, Inner Fault, and Ball Fault.

---

##  Tech Stack

- **Python 3.10+** (Core programming language)
- **PyTorch** (Deep Learning framework for CNNs and Grad-CAM)
- **Streamlit** (Frontend dashboard and UI)
- **NumPy & Pandas** (Numerical operations and data manipulation)
- **SciPy & H5py** (MATLAB file processing and FFT transforms)
- **Matplotlib & Seaborn** (Data visualization)

---

##  How to Run

### 1. Local Setup
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/kanishka9r/pipeline-fault-detectionn.git
cd pipeline-fault-detectionn
pip install -r requirements.txt
```

### 2. Launch the Dashboard
To start the interactive web application, simply run:
```bash
streamlit run dashboard.py
```
This will open the dashboard in your default web browser.

### 3. Training the Model (Optional)
If you have the full 3.3 GB `.mat` dataset and want to retrain the models from scratch:
1. Place the dataset in `data_generation/pipelinedataset/`
2. Run the training script:
   ```bash
   python prediction_train.py
   ```

---

##  Evaluation and Metrics

The model's performance is rigorously evaluated exclusively on the isolated **Test Set**:
- **Accuracy:** Achieves >98% accuracy on unseen test machines.
- **Precision & Recall:** Maintains 98% weighted precision and 98% weighted recall, indicating highly reliable fault identification with minimal false positives and missed detections.
- **Confusion Matrix:** Tracks precise false-positive and false-negative rates across all fault types.
- **Grad-CAM Visualization:** Generates class-specific activation maps and highlights important frequency bands associated with the predicted bearing condition.

---

## 📄 License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file  for details.
