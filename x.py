import os
import pandas as pd


data_dir = "data/problems/normalized_data" 


def get_label_from_filename(filename):
    from pathlib import Path
    name = Path(filename).stem
    parts = name.split('_')[:-3]  # drop trailing number if your naming has it
    return '_'.join(parts)


raw_labels = []

def build_dataset_from_folder(data_dir):
      for root, dirs, files in os.walk(data_dir):   # 🔄 walks inside all subfolders
        for fname in files:
            if fname.endswith('.csv'):
                path = os.path.join(root, fname)
                df = pd.read_csv(path)   # load CSV if needed
                label = get_label_from_filename(fname)
                print("Found:", path, "→ Label:", label)  # 🟢 Debug print
                raw_labels.append(label)
    



build_dataset_from_folder(data_dir)
print("\nTotal labels found:", len(raw_labels))
print(raw_labels[:20], "...")        