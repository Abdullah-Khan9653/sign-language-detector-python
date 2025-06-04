import os
import numpy as np
import pickle

DATA_DIR = './data'
data = []
labels = []

# Automatically loop through A to Z
for label, dir_name in enumerate(sorted(os.listdir(DATA_DIR))):
    dir_path = os.path.join(DATA_DIR, dir_name)
    if os.path.isdir(dir_path):
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            try:
                data.append(np.load(file_path))
                labels.append(label)
            except:
                pass  # In case any corrupted .npy

data = np.array(data)
labels = np.array(labels)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset created with {len(data)} samples.")
