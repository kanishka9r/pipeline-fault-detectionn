import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from paderborn_loader import extract_signals

test_files = np.load("data_genration/reqdata/test_files.npy", allow_pickle=True)
print(f"Compressing all {len(test_files)} test files...")

output_data = {}
root_dir = os.path.join("data_genration", "pipelinedataset")

success_count = 0
for i, filename in enumerate(test_files):
    parts = filename.split('_')
    if len(parts) >= 4:
        folder = parts[3]
    else:
        continue
        
    file_path = os.path.join(root_dir, folder, filename)
    if os.path.exists(file_path):
        try:
            x, y = extract_signals(file_path)
            # Convert to float32 to save space
            output_data[f"{filename}_x"] = x.astype(np.float32)
            output_data[f"{filename}_y"] = y.astype(np.float32)
            success_count += 1
            if success_count % 50 == 0:
                print(f"Compressed {success_count}/{len(test_files)} files...")
        except Exception as e:
            pass

np.savez_compressed("data_genration/model/demo_signals.npz", **output_data)
print(f"\nSuccessfully compressed {success_count} files into demo_signals.npz!")
file_size_mb = os.path.getsize("data_genration/model/demo_signals.npz") / (1024 * 1024)
print(f"Final file size: {file_size_mb:.2f} MB")
