This project implements a multi-stage machine learning system for detecting, classifying, and estimating the intensity of faults in a simulated industrial pipeline. The system operates on time-series sensor data from vibration, pressure, and temperature sensors.

##Features
Anomalous Behavior Detection: Utilizes an unsupervised learning approach to identify deviations from normal operating conditions.
Fault Classification: Classifies detected anomalies into specific fault types, such as leaks, blockages, or sensor failures.
Fault Intensity Estimation: Quantifies the severity of a fault using a regression model, providing a measure of its intensity.
Comprehensive Fault Coverage: Trained on a diverse synthetic dataset that includes single, combined, and sensor-specific faults.

##Dataset Used
The system is trained and evaluated on a synthetically generated dataset to simulate a wide range of pipeline conditions. The dataset includes:

Normal Data: Stable and fluctuating operational data to establish a baseline.
Process Faults: Simulated events like leaks (pressure drop, vibration rise) and blockages (pressure spike), created with varying intensity (low, high) and speed (slow, fast).
Sensor Faults: Simulated sensor malfunctions, including flat-line and high-noise faults.
Combined Faults: Complex scenarios where multiple process and sensor faults occur simultaneously.

##Model Architecture
The project uses a two-stage machine learning pipeline.

Stage 1: Unsupervised Anomaly Detection
An LSTM Autoencoder is trained exclusively on normal pipeline data. The model learns to reconstruct normal time-series patterns. During inference, a high reconstruction error on a new data point indicates an anomaly. This stage's primary function is to identify when a fault occurs.

Stage 2: Supervised Fault Classification & Intensity Estimation
A CNN-LSTM Multi-Task Learning model is trained on extracted segments of anomalous data from Stage 1.
A 1D CNN extracts local, spatial features from the time-series segments.
A Bi-directional LSTM processes these features to learn temporal dependencies.
Two output heads perform simultaneous tasks:
Classification Head: A classifier predicts the specific fault type.
Regression Head: A regressor estimates the fault intensity.

##Tech Stack
Python: The core programming language.
PyTorch: The deep learning framework used for building and training the models.
NumPy: Used for numerical operations and data manipulation.
Pandas: Used for handling and processing the dataset.
scikit-learn: Used for data splitting, normalization, and evaluation metrics.
Matplotlib & Seaborn: Used for data visualization and plotting model performance.

##How to Run
Clone the repository: git clone [repository URL]
Install dependencies: pip install -r requirements.txt (or manually install the libraries listed in the Tech Stack section).
Generate the dataset: Run the data generation scripts to create the synthetic data.
Train the autoencoder: Run the script that trains the LSTM Autoencoder on normal data to establish a baseline and threshold.
Extract anomaly segments: Run the script that uses the autoencoder to find and extract anomalous segments from the fault data.
Train the multi-task model: Run the main script to train the CNN-LSTM model on the extracted anomaly segments.

##Future Enhancements
Real-Time Monitoring: Implement a system to process incoming data streams in real time.
Hyperparameter Tuning: Use automated tools to optimize the model's hyperparameters for better performance.
Explainable AI (XAI): Add methods to interpret model predictions, showing which sensor features are most indicative of a fault.
Real-World Data Validation: Test the model on real-world pipeline data to validate its robustness.
