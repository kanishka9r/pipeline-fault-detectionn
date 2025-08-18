import numpy as np
import joblib
from collections import deque
import torch

# Load trained scaler
scaler = joblib.load("scaler.pkl")

# Rolling buffer for last 60 timesteps
buffer = deque(maxlen=60)

def preprocess_step(new_step):
    """
    Preprocess a single incoming data point.
    new_step: array/list of shape (105,)
    """
    # Normalize using the training scaler
    new_step_scaled = scaler.transform([new_step])[0]
    
    # Add to rolling buffer
    buffer.append(new_step_scaled)

    # Only proceed if buffer is full
    if len(buffer) == 60:
        # Shape: (1, 60, 105) for model
        seq = np.array(buffer).reshape(1, 60, -1)
        return seq
    else:
        return None

# Example usage with a new incoming data point
def run_inference(new_step, model):
    seq = preprocess_step(new_step)
    if seq is None:
        return None  # buffer not ready yet
    
    # Convert to torch tensor
    x = torch.tensor(seq, dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    return output
