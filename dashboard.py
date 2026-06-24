import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import resample
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Pipeline Bearing Diagnostic Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — enterprise warm-neutral palette with high contrast text
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Top Header Strip */
    header[data-testid="stHeader"] {
        background-color: #EFECE5 !important;
        border-bottom: 1px solid #E4E0D7 !important;
    }


    /* Gradient Title */
    .dashboard-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F7A65 0%, #5B8C5A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .dashboard-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #5A5650;
        margin-bottom: 1.8rem;
    }

    /* Stats Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E4E0D7;
        border-radius: 14px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        min-height: 120px;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6F6A63;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.4rem;
        color: #2D2A26;
    }

    /* Status classes */
    .status-healthy { color: #5B8C5A !important; }
    .status-outer { color: #B75D5D !important; }
    .status-inner { color: #C98B47 !important; }
    .status-ball { color: #7A6F9B !important; }

    .status-match { color: #5B8C5A !important; font-weight: bold; }
    .status-mismatch { color: #B75D5D !important; font-weight: bold; }

    /* Buttons */
    .stButton > button {
        background-color: #4F7A65 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background-color: #456D5A !important;
    }
</style>
""", unsafe_allow_html=True)

# Make sure workspace path is in system path to import existing scripts
workspace_dir = os.path.dirname(os.path.abspath(__file__))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

try:
    from paderborn_loader import extract_signals, create_windows, get_label_from_folder, to_fft
    from cnnlstm import CNNLSTMClassifier
    from cnn import CNNGradCAM
    from gradcam import GradCAM
except ImportError as e:
    st.error(f"Failed to import core scripts: {e}. Please ensure that paderborn_loader.py, cnnlstm.py, cnn.py, and gradcam.py are in the workspace.")
    st.stop()

# Helper Constants
class_name = [
    "Healthy",
    "Outer Fault",
    "Inner Fault",
    "Ball Fault"
]

class_colors = {
    "Healthy": "#5B8C5A",
    "Outer Fault": "#B75D5D",
    "Inner Fault": "#C98B47",
    "Ball Fault": "#7A6F9B"
}

class_status_css = {
    "Healthy": "status-healthy",
    "Outer Fault": "status-outer",
    "Inner Fault": "status-inner",
    "Ball Fault": "status-ball"
}

# Chart palette
CHART_BG    = "#FAFAF8"
CHART_GRID  = "#E5E5E0"
CHART_TEXT  = "#2D2A26"
CHART_SEC   = "#5A5650"
CHART_SPINE = "#D8D3CA"

# --- Caching Loaders ---
@st.cache_resource
def load_models():
    """Load models once and cache them in memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. CNN-LSTM Classifier
    classifier = CNNLSTMClassifier(num_classes=4).to(device)
    classifier.load_state_dict(
        torch.load(
            os.path.join("data_genration", "model", "best_paderborn_cnn.pt"),
            map_location=device
        )
    )
    classifier.eval()

    # 2. GradCAM Model
    gc_model = CNNGradCAM(4).to(device)
    gc_model.load_state_dict(
        torch.load(
            os.path.join("data_genration", "model", "best_gradcam_cnn.pt"),
            map_location=device
        )
    )
    gc_model.eval()

    # 3. GradCAM generator
    gc_generator = GradCAM(gc_model, gc_model.features[11])

    return classifier, gc_generator, device

@st.cache_resource
def load_normalization_params():
    """Load and cache normalisation parameters (mean, std)."""
    if os.path.exists(os.path.join("data_genration", "reqdata", "mean.npy")):
        mean = np.load(os.path.join("data_genration", "reqdata", "mean.npy"))
        std = np.load(os.path.join("data_genration", "reqdata", "std.npy"))
    elif os.path.exists(os.path.join("data_genration", "traineddata", "mean.npy")):
        mean = np.load(os.path.join("data_genration", "traineddata", "mean.npy"))
        std = np.load(os.path.join("data_genration", "traineddata", "std.npy"))
    else:
        mean = np.load(os.path.join("data_genration", "model", "mean.npy"))
        std = np.load(os.path.join("data_genration", "model", "std.npy"))
    return mean, std

@st.cache_data
def get_categorized_test_files():
    """Categorize test files by actual class for easier exploration."""
    demo_path = os.path.join("data_genration", "model", "demo_signals.npz")
    has_full_data = os.path.exists(os.path.join("data_genration", "pipelinedataset"))
    
    if not has_full_data and os.path.exists(demo_path):
        # We are on the cloud demo, load only the demo files
        demo_data = np.load(demo_path)
        test_files = [k.replace('_x', '') for k in demo_data.files if k.endswith('_x')]
    else:
        req_path = os.path.join("data_genration", "reqdata", "test_files.npy")
        trained_path = os.path.join("data_genration", "traineddata", "test_files.npy")
        model_path = os.path.join("data_genration", "model", "test_files.npy")
        
        if os.path.exists(req_path):
            load_path = req_path
        elif os.path.exists(trained_path):
            load_path = trained_path
        else:
            load_path = model_path
            
        test_files = np.load(load_path, allow_pickle=True)

    categorized = {
        "Healthy": [],
        "Outer Fault": [],
        "Inner Fault": [],
        "Ball Fault": []
    }
    for f in test_files:
        parts = f.split('_')
        if len(parts) >= 4:
            folder = parts[3]
            if folder.startswith("K00"):
                categorized["Healthy"].append(f)
            elif folder.startswith("KA"):
                categorized["Outer Fault"].append(f)
            elif folder.startswith("KI"):
                categorized["Inner Fault"].append(f)
            elif folder.startswith("KB"):
                categorized["Ball Fault"].append(f)
    return categorized

# --- Single File Processor ---
def load_and_process_file(filename, window_size=2048):
    """Loads a single MATLAB file and slices it into windows."""
    demo_path = os.path.join("data_genration", "model", "demo_signals.npz")
    has_full_data = os.path.exists(os.path.join("data_genration", "pipelinedataset"))
    
    parts = filename.split('_')
    if len(parts) >= 4:
        folder = parts[3]
    else:
        folder = "Unknown"
        
    label = get_label_from_folder(folder) if folder != "Unknown" else 0

    if not has_full_data and os.path.exists(demo_path):
        # Load from demo data
        demo_data = np.load(demo_path)
        x_signal = demo_data[f"{filename}_x"]
        y_signal = demo_data[f"{filename}_y"]
    else:
        # Load from full dataset
        if folder == "Unknown":
            root_dir = os.path.join("data_genration", "pipelinedataset")
            for fld in os.listdir(root_dir):
                if os.path.isdir(os.path.join(root_dir, fld)):
                    if filename in os.listdir(os.path.join(root_dir, fld)):
                        folder = fld
                        label = get_label_from_folder(folder)
                        break
        
        file_path = os.path.join("data_genration", "pipelinedataset", folder, filename)
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Could not locate {file_path}")
        x_signal, y_signal = extract_signals(file_path)

    # Create windows
    windows = create_windows(x_signal, y_signal, window_size)

    return x_signal, y_signal, windows, label

# Helper: make a clean light-background chart
def make_figure(w=10, h=4.8):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=CHART_BG)
    ax.set_facecolor(CHART_BG)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(CHART_SPINE)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=CHART_SEC, labelsize=10)
    ax.grid(True, color=CHART_GRID, alpha=0.6, linestyle='-', linewidth=0.5)
    return fig, ax

# --- Main App Layout ---
st.markdown("<div class='dashboard-title'>Pipeline Bearing Diagnostic Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='dashboard-subtitle'>Vibration-Based Bearing Fault Classification with CNN-LSTM and GradCAM Explainability</div>", unsafe_allow_html=True)

# Load cached items
with st.spinner("Loading models and files list..."):
    try:
        model, gradcam_gen, device = load_models()
        mean, std = load_normalization_params()
        categorized_files = get_categorized_test_files()
    except Exception as ex:
        st.error(f"Error loading models or parameters: {ex}. Ensure you are running from the project root directory.")
        st.stop()

# --- Sidebar Controls ---
st.sidebar.markdown("### Diagnostic Settings")

# 1. Filter by actual class
actual_class_sel = st.sidebar.selectbox(
    "Filter by Actual Bearing State",
    options=["Healthy", "Outer Fault", "Inner Fault", "Ball Fault"]
)

# 2. Select file based on filter
files_list = categorized_files[actual_class_sel]
if len(files_list) == 0:
    st.sidebar.warning(f"No test files found for class '{actual_class_sel}'")
    st.stop()

# Random file selector
if st.sidebar.button("Pick Random File"):
    selected_file = np.random.choice(files_list)
    # Save selection in session state so it updates the selectbox
    st.session_state["selected_file"] = selected_file
else:
    if "selected_file" not in st.session_state or st.session_state["selected_file"] not in files_list:
        st.session_state["selected_file"] = files_list[0]

selected_file = st.sidebar.selectbox(
    "Select Test File (.mat)",
    options=files_list,
    index=files_list.index(st.session_state["selected_file"])
)
st.session_state["selected_file"] = selected_file

# Parse bearing identifier from file name
bearing_id = selected_file.split("_")[3] if len(selected_file.split("_")) >= 4 else "Unknown"

# 3. Load file data
x_sig, y_sig, windows, actual_label = load_and_process_file(selected_file)
num_windows = len(windows)

# 4. Window Selector Slider
st.sidebar.markdown("---")
st.sidebar.markdown("### Window Analysis")
window_idx = st.sidebar.slider(
    "Select Signal Window",
    min_value=0,
    max_value=num_windows - 1,
    value=0,
    help=f"Vibration files contain continuous measurements sliced into windows of 2048 samples. This file has {num_windows} windows."
)

# 5. Model Explanation Parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Parameters")
gradcam_thresh = st.sidebar.slider(
    "GradCAM Importance Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.77,
    step=0.01,
    help="Threshold above which a frequency region is considered highly important for the classification."
)

confidence_thresh = st.sidebar.slider(
    "High Confidence Threshold (%)",
    min_value=70,
    max_value=100,
    value=90,
    step=5,
    help="If prediction confidence is above this, model is labeled highly confident."
)

# Tabs
tab1, tab2 = st.tabs(["Diagnostic Analytics & Waveforms", "Batch Performance Report"])

# --- TAB 1: Diagnostic Analytics & Waveforms ---
with tab1:
    # RUN PREDICTION ON SELECTED WINDOW
    selected_window = windows[window_idx]

    # Process FFT
    fft_vals = to_fft(selected_window)
    # Normalize FFT
    fft_norm = (fft_vals - mean) / std

    # Model inference
    sample = torch.tensor(fft_norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(sample)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    pred_class_idx = pred.item()
    pred_class_name = class_name[pred_class_idx]
    actual_class_name = class_name[actual_label]
    confidence_val = confidence.item() * 100

    # Run GradCAM
    cam, gradcam_pred_idx = gradcam_gen.generate(sample)
    gradcam_class_name = class_name[gradcam_pred_idx]

    # Calculate important frequencies
    fft_x = sample.cpu().numpy()[0, :, 0]
    fft_y = sample.cpu().numpy()[0, :, 1]

    cam_resized = resample(cam, len(fft_x))
    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)

    important_idx = np.where(cam_resized > gradcam_thresh)[0]

    # Top level KPIs in custom styled HTML
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Selected File</div>
            <div class="metric-value" style="font-size: 1.25rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{selected_file}">{selected_file}</div>
            <div style="font-size: 0.8rem; color: #6F6A63; margin-top: 0.2rem;">Bearing: {bearing_id}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        class_css = class_status_css[actual_class_name]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Actual State</div>
            <div class="metric-value {class_css}">{actual_class_name}</div>
            <div style="font-size: 0.8rem; color: #6F6A63; margin-top: 0.2rem;">True Label</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        class_css = class_status_css[pred_class_name]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CNN-LSTM Prediction</div>
            <div class="metric-value {class_css}">{pred_class_name}</div>
            <div style="font-size: 0.8rem; color: #6F6A63; margin-top: 0.2rem;">Classifier Output</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        # Green for correct, Red for misclassification
        match_text = "Match" if pred_class_idx == actual_label else "Mismatch"
        match_css = "status-match" if pred_class_idx == actual_label else "status-mismatch"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Match Status</div>
            <div class="metric-value {match_css}">{match_text}</div>
            <div style="font-size: 0.8rem; color: #6F6A63; margin-top: 0.2rem;">Prediction Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Confidence</div>
            <div class="metric-value" style="color: #4F7A65;">{confidence_val:.2f}%</div>
            <div style="font-size: 0.8rem; color: #6F6A63; margin-top: 0.2rem;">Softmax Probability</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Middle Section: Probabilities & Diagnostic Text
    mid_col1, mid_col2 = st.columns([2, 3])

    with mid_col1:
        st.subheader("Class Probabilities Breakdown")
        prob_df = pd.DataFrame({
            "State": class_name,
            "Probability (%)": probs[0].cpu().numpy() * 100
        })

        # Color palettes matching our diagnostic classes
        colors = [class_colors[c] for c in class_name]

        fig_prob, ax_prob = make_figure(6, 4.3)

        bars = ax_prob.barh(prob_df["State"], prob_df["Probability (%)"], color=colors, height=0.55)
        ax_prob.set_xlim(0, 105)
        ax_prob.set_xlabel("Probability (%)", color=CHART_SEC, fontweight="bold")

        # Add labels to bars
        for bar in bars:
            width = bar.get_width()
            ax_prob.text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.1f}%",
                         va='center', ha='left', color=CHART_TEXT, fontweight='semibold', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig_prob)

    with mid_col2:
        st.subheader("Expert Diagnostic Explanation")

        # Build explanation string dynamically
        exp_box = st.container()
        with exp_box:
            # 1. Confidence Evaluation
            if confidence_val >= confidence_thresh:
                st.success(f"**Model Status:** Highly Confident. (Confidence: {confidence_val:.2f}%)")
                conf_exp = "The model is highly confident in this prediction based on the distinct spectral signature of the vibration signal."
            elif confidence_val >= 80:
                st.warning(f"**Model Status:** Moderate Confidence. (Confidence: {confidence_val:.2f}%)")
                conf_exp = "The model is moderately confident. The signal contains features corresponding to multiple bearing states, or high background noise."
            else:
                st.error(f"**Model Status:** Low Confidence - Manual Review Recommended. (Confidence: {confidence_val:.2f}%)")
                conf_exp = "The model confidence is low. This suggests a highly complex or unusual spectral pattern that may require manual inspection."

            # 2. Freq regions info
            region_str = "None detected"
            freq_str = "N/A"
            start_bin = "N/A"
            end_bin = "N/A"
            peak_bin = "N/A"
            peak_freq = 0
            if len(important_idx) > 0:
                start_bin = int(important_idx.min())
                end_bin = int(important_idx.max())

                # Convert bins to frequencies in Hz
                Fs = 64000
                freq_resolution = Fs / 2048
                start_freq = start_bin * freq_resolution
                end_freq = end_bin * freq_resolution
                freq_str = f"{start_freq:.0f} Hz - {end_freq:.0f} Hz"

                region_center = (start_bin + end_bin) / 2
                if region_center < 100:
                    region_str = "Low Frequency Region"
                elif region_center < 400:
                    region_str = "Mid Frequency Region"
                else:
                    region_str = "High Frequency Region"

                peak_bin = int(np.argmax(cam_resized))
                peak_freq = peak_bin * freq_resolution

            # 3. Text diagnostics
            diagnostic_texts = {
                0: "The spectral patterns represent a healthy bearing operating under standard parameters. The energy is evenly distributed without sharp localized peaks.",
                1: "The frequency distribution matches an Outer Race Fault. Outer race faults generate cyclical impacts at the Ball Pass Frequency Outer (BPFO) and its harmonics.",
                2: "The frequency distribution matches an Inner Race Fault. Inner race faults exhibit modulation effects around the shaft speed, resulting in sidebands around the Ball Pass Frequency Inner (BPFI).",
                3: "The frequency distribution matches a Ball Fault. Ball faults are characterized by vibration frequencies showing up at the Ball Spin Frequency (BSF) and Fundamental Train Frequency (FTF)."
            }

            explanation_html = f"""
            <div style="background-color: #FFFFFF; border-left: 4px solid {class_colors[pred_class_name]}; padding: 1rem; border-radius: 4px; margin-top: 0.5rem; border: 1px solid #E4E0D7; border-left: 4px solid {class_colors[pred_class_name]};">
                <p style="font-size: 1.05rem; line-height: 1.5; color: #2D2A26; margin-bottom: 0.8rem;">
                    <strong>Vibration Signature Diagnostic:</strong> {diagnostic_texts[pred_class_idx]}
                </p>
                <hr style="border: 0; border-top: 1px solid #E4E0D7; margin: 0.8rem 0;">
                <p style="font-size: 0.95rem; color: #5A5650; margin: 0;">
                    <strong>Explainable AI (GradCAM) Analysis:</strong><br>
                    The neural network focused its decision-making attention on the <strong>{region_str}</strong> (FFT bins {start_bin}-{end_bin}).<br>
                    &bull; <strong>Critical Frequency Band:</strong> {freq_str}<br>
                    &bull; <strong>Peak Excitation Frequency:</strong> {peak_freq:.0f} Hz (Bin {peak_bin})<br>
                    &bull; <strong>Interpretability Match:</strong> GradCAM and CNN-LSTM models {'agree' if pred_class_idx == gradcam_pred_idx else 'disagree'} on classification.
                </p>
            </div>
            """
            st.markdown(explanation_html, unsafe_allow_html=True)

    # Waveform and Spectral Analysis Plots
    st.markdown("<h3 style='margin-top: 1.5rem; color: #2D2A26;'>Waveform & Spectral Signatures</h3>", unsafe_allow_html=True)

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.markdown("**1. Raw Vibration Signals (Time Domain)**")
        # Extract signal coordinates for the selected window
        start_samp = window_idx * 2048
        end_samp = start_samp + 2048
        window_x_sig = x_sig[start_samp:end_samp]
        window_y_sig = y_sig[start_samp:end_samp]
        time_axis = np.arange(2048) / 64.0  # ms scale at 64kHz

        fig_raw, ax_raw = make_figure()

        ax_raw.plot(time_axis, window_x_sig, color='#4F7A65', alpha=0.85, linewidth=1.0, label="Channel X (Radial Force)")
        ax_raw.plot(time_axis, window_y_sig, color='#C98B47', alpha=0.75, linewidth=1.0, label="Channel Y (Vibration)")

        ax_raw.set_title("2048-Sample Raw Signal Window", color=CHART_TEXT, fontsize=11, pad=10)
        ax_raw.set_xlabel("Time (ms)", color=CHART_SEC)
        ax_raw.set_ylabel("Amplitude", color=CHART_SEC)
        ax_raw.legend(loc="upper right", framealpha=0.7, facecolor=CHART_BG, edgecolor=CHART_SPINE, labelcolor=CHART_TEXT)

        plt.tight_layout()
        st.pyplot(fig_raw)

    with plot_col2:
        st.markdown("**2. FFT Envelope Spectrum & GradCAM Importance**")

        fig_fft, ax_fft = make_figure()

        # FFT lines
        ax_fft.plot(fft_x, color='#5B8C5A', alpha=0.75, linewidth=1.2, label="FFT Envelope X")
        ax_fft.plot(fft_y, color='#C98B47', alpha=0.65, linewidth=1.2, label="FFT Envelope Y")

        # GradCAM overlay (scaled to match FFT height)
        scale_val = max(np.max(fft_x), np.max(fft_y))
        cam_scaled = cam_resized * scale_val
        ax_fft.plot(cam_scaled, color='#B75D5D', linewidth=1.8, linestyle='--', label="GradCAM Importance")

        # Highlight important regions
        if len(important_idx) > 0:
            ax_fft.fill_between(
                range(len(fft_x)), 0, cam_scaled,
                where=(cam_resized > gradcam_thresh),
                color='#B75D5D', alpha=0.12, label=f"Critical Frequencies (>{gradcam_thresh:.2f})"
            )
            # Add vertical lines for bounds
            ax_fft.axvline(x=start_bin, color='#B75D5D', alpha=0.3, linestyle='-', linewidth=0.8)
            ax_fft.axvline(x=end_bin, color='#B75D5D', alpha=0.3, linestyle='-', linewidth=0.8)

        ax_fft.set_title(f"Spectral Explanation (GradCAM Pred: {gradcam_class_name})", color=CHART_TEXT, fontsize=11, pad=10)
        ax_fft.set_xlabel("FFT Bin", color=CHART_SEC)
        ax_fft.set_ylabel("Magnitude (Log)", color=CHART_SEC)
        ax_fft.legend(loc="upper right", framealpha=0.7, facecolor=CHART_BG, edgecolor=CHART_SPINE, labelcolor=CHART_TEXT)

        plt.tight_layout()
        st.pyplot(fig_fft)

# --- TAB 2: Batch Performance Evaluation Report ---
with tab2:
    st.subheader("Real-Time Model Validation & Confusion Matrix")
    st.markdown("Assess the performance of the models on a random subset of files from the test dataset. Select the evaluation size below and run the batch diagnostic.")

    # Controls
    val_size = st.slider("Evaluation Sample Size (Files)", min_value=5, max_value=50, value=20, step=5)

    if st.button("Run Batch Evaluation Report"):
        # Check if we are running on Streamlit Cloud (using demo data)
        has_full_data = os.path.exists(os.path.join("data_genration", "pipelinedataset"))
        if has_full_data:
            if os.path.exists(os.path.join("data_genration", "reqdata", "test_files.npy")):
                test_files_all = np.load(os.path.join("data_genration", "reqdata", "test_files.npy"), allow_pickle=True)
            elif os.path.exists(os.path.join("data_genration", "traineddata", "test_files.npy")):
                test_files_all = np.load(os.path.join("data_genration", "traineddata", "test_files.npy"), allow_pickle=True)
            else:
                test_files_all = np.load(os.path.join("data_genration", "model", "test_files.npy"), allow_pickle=True)
        else:
            # If on cloud, only evaluate on the files included in the lightweight demo dataset
            demo_data = np.load(os.path.join("data_genration", "model", "demo_signals.npz"))
            test_files_all = [k.replace('_x', '') for k in demo_data.files if k.endswith('_x')]

        # Sample randomly
        sampled_files = np.random.choice(test_files_all, size=min(val_size, len(test_files_all)), replace=False)

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        results = []

        for i, fname in enumerate(sampled_files):
            status_text.text(f"Processing file {i+1}/{len(sampled_files)}: {fname}")
            progress_bar.progress((i + 1) / len(sampled_files))

            try:
                # Load file
                _, _, file_windows, file_label = load_and_process_file(fname)
                if len(file_windows) == 0:
                    continue

                # Pick a random window from the file
                w_idx = np.random.randint(0, len(file_windows))
                w = file_windows[w_idx]

                # FFT & Normalize
                f_vals = to_fft(w)
                f_norm = (f_vals - mean) / std

                # Predict
                w_sample = torch.tensor(f_norm, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    w_out = model(w_sample)
                    w_probs = torch.softmax(w_out, dim=1)
                    w_conf, w_pred = torch.max(w_probs, dim=1)

                results.append({
                    "Filename": fname,
                    "Actual Class": class_name[file_label],
                    "Predicted Class": class_name[w_pred.item()],
                    "Confidence (%)": w_conf.item() * 100,
                    "Status": "Correct" if w_pred.item() == file_label else "Incorrect",
                    "Actual_Idx": file_label,
                    "Pred_Idx": w_pred.item()
                })
            except Exception as e:
                st.warning(f"Failed to process {fname}: {e}")

        status_text.text("Evaluation completed!")

        if len(results) > 0:
            df_res = pd.DataFrame(results)

            # Compute accuracy
            correct_count = sum(df_res["Status"] == "Correct")
            acc = (correct_count / len(df_res)) * 100

            # Displays
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 1.5rem; border-color: #4F7A65;">
                    <div class="metric-label">Batch Accuracy</div>
                    <div class="metric-value" style="font-size: 3.5rem; color: #5B8C5A;">{acc:.1f}%</div>
                    <div style="font-size: 0.9rem; color: #6F6A63; margin-top: 0.3rem;">({correct_count}/{len(df_res)} files classified correctly)</div>
                </div>
                """, unsafe_allow_html=True)

                # Show results table
                st.markdown("### Evaluated Samples Details")
                st.dataframe(
                    df_res[["Filename", "Actual Class", "Predicted Class", "Confidence (%)", "Status"]],
                    use_container_width=True
                )

            with res_col2:
                st.markdown("### Confusion Matrix")

                y_true = df_res["Actual_Idx"].values
                y_pred = df_res["Pred_Idx"].values

                # Build complete 4x4 matrix
                cm = np.zeros((4, 4), dtype=int)
                for t, p in zip(y_true, y_pred):
                    cm[t, p] += 1

                fig_cm, ax_cm = make_figure(6.5, 5.5)

                sns.heatmap(
                    cm, annot=True, fmt='d',
                    xticklabels=class_name, yticklabels=class_name,
                    cbar=False, ax=ax_cm,
                    annot_kws={"size": 13, "weight": "bold", "color": CHART_TEXT},
                    cmap=sns.light_palette("#4F7A65", as_cmap=True),
                    linewidths=1, linecolor=CHART_BG
                )

                ax_cm.set_title("Confusion Matrix (Sample Subset)", color=CHART_TEXT, fontsize=12, pad=15)
                ax_cm.set_xlabel("Predicted State", color=CHART_SEC, fontweight='bold', labelpad=10)
                ax_cm.set_ylabel("Actual State", color=CHART_SEC, fontweight='bold', labelpad=10)

                # Rotate labels for better fit
                plt.xticks(rotation=15)
                plt.yticks(rotation=0)

                plt.tight_layout()
                st.pyplot(fig_cm)
        else:
            st.error("No samples were successfully evaluated.")
