# Pipeline Bearing Fault Detection

An enterprise-grade, end-to-end machine learning pipeline and interactive dashboard for industrial bearing health monitoring. 

This project utilizes the **Paderborn University Bearing Dataset** to perform supervised Fault Classification. By analyzing high-frequency vibration signals, it identifies early-stage degradation and provides precise diagnostics of failure modes, significantly reducing industrial downtime.

---

## ✨ Features

- **Interactive Cloud Dashboard:** A fully responsive, modern Streamlit web application for real-time visualization of vibration signals and fault predictions.
- **Deep 1D-CNN Architecture:** Custom-built Convolutional Neural Network optimized specifically for time-series frequency data.
- **Explainable AI (Grad-CAM):** Visualizes which specific frequencies the AI is looking at to make its predictions, ensuring transparent decision-making.
- **Zero Data Leakage:** Strict file-level train/validation/test splitting guarantees the model generalizes reliably to unseen machinery.
- **Lightweight Deployment:** Uses a compressed `.npz` demo dataset to bypass GitHub storage limits while maintaining full functionality on Streamlit Cloud.

---

## 📊 Dataset Used

The system is optimized for the **Paderborn University (PU) Dataset**, focusing on high-frequency vibration data:
- **Healthy State:** Baseline operating conditions without defects.
- **Artificial & Real Damage:** Classifies defects into **Outer Fault**, **Inner Fault**, and **Ball Fault**.
- **Data Format:** Raw `.mat` (MATLAB) files, which are dynamically sliced into 2048-sample windows.

---

## 🧠 Model Architecture

The project processes raw vibration streams through a two-stage pipeline:

### 1. Data Engineering & Preprocessing
- **Segmentation:** Continuous streams are partitioned into discrete 2048-sample windows.
- **Fast Fourier Transform (FFT):** Converts raw time-domain vibrations into the frequency domain, where fault harmonics are most prominent.
- **Log-Scaling & Normalization:** Normalizes the dynamic range using standard scaling (`mean.npy` / `std.npy`) for stable neural network convergence.

### 2. Fault Classification (Deep 1D-CNN)
- **Architecture:** 4-layer Deep 1D Convolutional Network with Batch Normalization and Dropout (0.4) for robust regularization.
- **Optimization:** Trained using Adam optimizer, `ReduceLROnPlateau` scheduler, and balanced class weights to handle natural dataset imbalances.
- **Outputs:** Softmax probability distribution across 4 states (Healthy, Outer Fault, Inner Fault, Ball Fault).

---

##  Tech Stack

- **Python 3.9+** (Core programming language)
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
This will open the dashboard in your default web browser (usually at `http://localhost:8501`).

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
- **Confusion Matrix:** Tracks precise false-positive and false-negative rates across all fault types.
- **Grad-CAM Heatmaps:** Validates that the model is detecting legitimate mechanical frequencies rather than memorizing background noise.

---

## 📄 License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
