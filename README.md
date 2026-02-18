# Bearing Fault Detection using CNN Autoencoder + CNN Classifier

This project implements a robust pipeline for industrial bearing health monitoring. It utilizes the Paderborn University Bearing Dataset to perform both unsupervised Anomaly Detection (detecting if a fault exists) and supervised Fault Classification (identifying the specific type and severity of the fault). It is designed to provide automated predictive maintenance. By analyzing vibration signals, it reduces downtime by identifying early-stage degradation and providing precise diagnostics of the failure mode.

## Features

- **Dual-Stage Analytics:** Combines an Autoencoder for novelty detection and a CNN for multi-class classification.
- **Robust Signal Processing:** Implements Hilbert Transform for envelope extraction and Fast Fourier Transform (FFT) for spectral analysis.
- **Leakage-Free Validation:** Strict file-level data splitting ensures the model generalizes to new, unseen machinery.
- **Automated Labeling:** Dynamic mapping of complex MATLAB folder structures into human-readable fault categories.


## Dataset Used

The system is optimized for the Paderborn University (PU) Dataset:

- **Healthy State:** Baseline operating conditions.
- **Artificial & Real Damage:** Includes Outer Race and Inner Race damages. 
- **Severity Levels:** Categorized into Low and High severity based on the damage extent.
- **Format:** High-frequency vibration data stored in .mat (MATLAB) files.

## Model Architecture

The project uses a two-stage machine learning pipeline:

### Stage 1:  Data Engineering & Preprocessing

The raw time-series data undergoes a rigorous transformation pipeline before reaching the models:

1) Segmentation: Continuous streams are partitioned into discrete windows of 2048 samples.
2) Envelope Analysis: The Hilbert Transform is applied to isolate the fault-related impulses from the carrier signal.
3) Spectral Conversion: FFT converts the envelope into the frequency domain, where fault frequencies are most prominent.
4) Log-Scaling: Normalizes the dynamic range of the spectral peaks for stable neural network training.
- **Purpose:** Extract optimized features for model.

### Stage 2: Anomaly Detection (CNN-Autoencoder)

An unsupervised Convolutional Autoencoder serves as the first line of defense:

1) Training: Learned exclusively on "Healthy" data.
2) Detection Mechanism: The model attempts to reconstruct the input. When a faulty signal is encountered, the Reconstruction Error (MSE) spikes, triggering an anomaly alert.
3) Thresholding: Uses ROC-curve optimization to define the boundary between normal and abnormal states.

### Stage 2: Fault Classification (Deep 1D-CNN)
Once an anomaly is detected, a Supervised 1D-CNN classifies the failure:

1) Architecture: 4-layer Deep Convolutional Network with Batch Normalization and Dropout (0.4) for regularization.
2) Categories: Classified into 5 states (Healthy, Outer_Low, Outer_High, Inner_Low, Inner_High).
3) Optimization: Utilizes a ReduceLROnPlateau scheduler and balanced class weights to handle dataset imbalances.

## Tech Stack

- **Python**: Core programming language  
- **PyTorch**: Deep learning framework  
- **NumPy**: Numerical operations and data manipulation  
- **Pandas**: Dataset handling and processing  
- **scikit-learn**: Data splitting, normalization, and evaluation metrics  
- **Matplotlib & Seaborn**: Data visualization and plotting  
- **H5py**: for MATLAB compatiblility

## How to Run

1. Clone the repository:  
   ```bash
   git clone https://github.com/kanishka9r/pipeline-fault-detectionn.git
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt
3. Data Preparation
Ensure your dataset is organized in the following directory structure:
    ```bash
   data_generation/pipelinedataset/[Folder_Names]/[Files].mat
4. Training the Pipeline
-  **Run Preprocessing & Anomaly Detection:**
Execute the Autoencoder script to establish the healthy baseline and detection threshold.
- **Run Fault Classification:**
Execute the CNN Classifier script to train the diagnostic model. The best weights will be saved to:
data_generation/model/best_paderborn_cnn.pt

## Evaluation and metrics

The system outputs comprehensive performance reports:

- **Confusion Matrix:** To visualize classification accuracy across all fault types.
- **ROC-AUC:** To measure the reliability of the anomaly detection stage.
- **Error Distribution:** Histograms comparing healthy vs. faulty reconstruction errors.

## Future Enhancements

- **Cross-operating condition generalization:** Evaluate anomaly detection performance when trained on one operating condition and tested on unseen operating conditions
- **Real-Time Monitoring:** Develop a dashboard to process live vibration data from sensors, allowing maintenance teams to see the health of the machinery in real-time rather than processing static files. 
- **Multi-Sensor Integration:** Incorporate data from other sensors, such as Temperature and Acoustic Emission, to improve the accuracy of the fault detection system and reduce false alarms. 
- **Remaining Useful Life (RUL) Prediction** Extend the model’s capabilities to not only detect faults but also estimate how many days or hours the bearing can safely operate before it completely fails.  
  
## License
This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
