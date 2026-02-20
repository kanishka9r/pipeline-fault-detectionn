import h5py
from scipy.io import loadmat
import numpy as np
import os
from scipy.signal import hilbert

def extract_signals(mat_path):
    try:
        # Try normal MATLAB loader
        data = loadmat(mat_path)   #load mat files in dict
        main_key = [k for k in data.keys() if not k.startswith("__")][0] #list of key only get y
        mat_struct = data[main_key][0,0] #main structure in y of data
        x = mat_struct['X'][0,0]['Data'].flatten()
        y = mat_struct['Y'][0,0]['Data'].flatten()
    except:
        # Fallback for MATLAB v7.3 files
        with h5py.File(mat_path, 'r') as f:
            main_key = list(f.keys())[0]
            mat_struct = f[main_key] #it is not in 2d array
            x = np.array(mat_struct['X']['Data']).flatten() #data is hdf5 version need to convert it in array
            y = np.array(mat_struct['Y']['Data']).flatten()
    return x, y

def get_label_from_folder(folder_name):
    # Healthy
    if folder_name.startswith("K00"):
        return 0 
    # Outer race
    elif folder_name.startswith("KA"):
        return 1 
    # Inner race
    elif folder_name.startswith("KI"):
        return 2
    # Ball
    elif folder_name.startswith("KB"):
        return 3  
    else:
        return -1

def create_windows(signal_x, signal_y, window_size=2048):
    windows = []
    length = len(signal_x)
    for start in range(0, length - window_size, window_size):
        end = start + window_size
        window = np.stack([
            signal_x[start:end],
            signal_y[start:end]
        ], axis=1)  # shape: (window_size, 2)
        windows.append(window)
    return np.array(windows) #shape(no of window , window_size , 2)

def load_dataset_with_files(root_dir, window_size=2048):
    X_data = []  #for data
    y_labels = []  #for labels
    file_ids = []   #for main file
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        label = get_label_from_folder(folder)
        if label == -1:
            continue
        for file in os.listdir(folder_path):
            if not file.endswith(".mat"):
                continue
            if not file.startswith("N"):
                continue
            file_path = os.path.join(folder_path, file)
            try:
                x_signal, y_signal = extract_signals(file_path)
            except Exception:
                continue
            windows = create_windows(x_signal, y_signal, window_size)
            for w in windows:
                X_data.append(w)
                y_labels.append(label)
                file_ids.append(file)  # track source file
    return np.array(X_data), np.array(y_labels), np.array(file_ids)

def to_fft(window):
    #1. Apply Hilbert to get the envelope 
    analytic_signal = hilbert(window, axis=0)
    amplitude_envelope = np.abs(analytic_signal)
    # 2. Compute FFT of the Envelope
    fft_vals = np.abs(np.fft.rfft(amplitude_envelope, axis=0))
    # 3. Log scale
    fft_vals = np.log1p(fft_vals)
    return fft_vals[:-1, :]
