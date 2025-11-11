# Industrial Pipeline Fault Detection Using LSTM  Autoencoder and CNN-LSTM Hybrid Model


This project implements a multi-stage machine learning system for detecting and classifying in a simulated industrial pipeline. The system operates on time-series sensor data from vibration, pressure, and temperature sensors.

## Features

- **Anomalous Behavior Detection:** Utilizes an unsupervised learning approach to identify deviations from normal operating conditions.  
- **Fault Classification:** Classifies detected anomalies into specific fault types, such as leaks, blockages, or sensor failures.    
- **Comprehensive Fault Coverage:** Trained on a diverse synthetic dataset that includes single, combined, and sensor-specific faults.  

## Dataset Used

The system is trained and evaluated on a synthetically generated dataset to simulate a wide range of pipeline conditions, including:

- **Normal Data:** Stable and fluctuating operational data to establish a baseline.  
- **Process Faults:** Simulated events like leaks (pressure drop, vibration rise) and blockages (pressure spike), created with varying intensity (low, high) .  
- **Sensor Faults:** Simulated sensor malfunctions, including low/high-noise faults.  
- **Combined Faults:** Complex scenarios where multiple faults occur simultaneously.  

## Model Architecture

The project uses a two-stage machine learning pipeline:

### Stage 1: Unsupervised Anomaly Detection
- An **LSTM Autoencoder** is trained exclusively on normal pipeline data.  
- The model learns to reconstruct normal time-series patterns.  
- During inference, a high reconstruction error on a new data point indicates an anomaly.  
- **Purpose:** Identify when a fault occurs.

### Stage 2: Supervised Fault Classification & Intensity Estimation
- A **CNN-LSTM Multi-Task Learning model** is trained on extracted segments of anomalous data from Stage 1.  
- **1D CNN:** Extracts local, spatial features from the time-series segments.  
- **Bi-directional LSTM:** Processes these features to learn temporal dependencies.  
- **Output Heads:**  
  - **Classification Head:** Predicts the specific fault type and intensity(low or high)

## Tech Stack

- **Python**: Core programming language  
- **PyTorch**: Deep learning framework  
- **NumPy**: Numerical operations and data manipulation  
- **Pandas**: Dataset handling and processing  
- **scikit-learn**: Data splitting, normalization, and evaluation metrics  
- **Matplotlib & Seaborn**: Data visualization and plotting  

## How to Run

1. Clone the repository:  
   ```bash
   git clone https://github.com/kanishka9r/pipeline-fault-detectionn.git
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt
3. Generate the dataset  
Run the data generation scripts to create the synthetic data.  
4. Train the autoencoder
Run the script to train the LSTM Autoencoder on normal data.
5. Normalize the dataset  
Run the script to normalize the dataset on 0-1 scale.
6. Extract anomaly segments  
Use the autoencoder to find and extract anomalous segments from the fault data.  
7. Train the multi-task model
Run the script to train the CNN-LSTM model on the extracted anomaly segments.

## Future Enhancements

- **Real-Time Monitoring:** Implement a system to process incoming data streams in real time.  
- **Hyperparameter Tuning:** Use automated tools to optimize model hyperparameters for better performance.  
- **Intensity Calculation.** Can use to check intensity of the fault on scale of (0-1).
- **Real-World Data Validation:** Test the model on real-world pipeline data to validate its robustness.  
  
## License
This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
See the [LICENSE](LICENSE) file for details.
