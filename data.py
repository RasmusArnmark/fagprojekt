import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import pickle
import torch
import mne
import numpy as np
import glob
import pandas as pd
import os

# Load metadata DataFrame
df_info = pd.read_pickle("data/FG_overview_df_v2.pkl")

# Define event IDs
event_labels = {'T1P': 301, 'T1Pn': 302, 'T3P': 303, 'T3Pn': 304,
                'T12P': 305, 'T12Pn': 306, 'T13P': 307, 'T13Pn': 308,
                'T23P': 309, 'T23Pn': 310}

file_paths = glob.glob("data/*_FG_preprocessed-epo.fif")
print(f"Found {len(file_paths)} EEG files.")

chunk_size = 4  # number of files per batch
num_patches = 6

for i in range(0, len(file_paths), chunk_size):
    batch_files = file_paths[i:i + chunk_size]
    all_eeg_data, all_labels = [], []

    for file_path in batch_files:
        file_name = file_path.split("/")[-1].split("_")[0]
        exp_id = file_name[:4]

        experiment_participants = df_info[df_info["Exp_id"] == exp_id]
        if experiment_participants.empty:
            continue

        try:
            epochs = mne.read_epochs(file_path, preload=True)
            epochs.resample(200)
            eeg_data = epochs.get_data()
            labels = epochs.events[:, -1]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        for _, row in experiment_participants.iterrows():
            subject_id = row["Subject_id"]
            eeg_device = row["EEG_device"]

            device_event_labels = {
                1: {301, 302, 305, 306, 307, 308},
                2: {303, 304, 305, 306, 309, 310},
                3: {307, 308, 309, 310, 303, 304},
            }
            valid_events = device_event_labels[eeg_device]
            valid_trials = [i for i, label in enumerate(labels) if label in valid_events]

            if not valid_trials:
                continue

            eeg_subject_data = eeg_data[valid_trials]
            labels_subject = labels[valid_trials]

            eeg_subject_data = (eeg_subject_data - eeg_subject_data.mean()) / eeg_subject_data.std()
            binary_labels = np.array([1 if label in {301, 303, 305, 307, 309} else 0 for label in labels_subject])

            all_eeg_data.append(eeg_subject_data)
            all_labels.append(binary_labels)

    if all_eeg_data:
        eeg_tensor = torch.tensor(np.concatenate(all_eeg_data, axis=0), dtype=torch.float32)
        labels_tensor = torch.tensor(np.concatenate(all_labels, axis=0), dtype=torch.long)
        time_steps_per_patch = eeg_tensor.shape[2] // num_patches
        eeg_tensor = eeg_tensor.reshape(eeg_tensor.shape[0], eeg_tensor.shape[1], num_patches, time_steps_per_patch)

        # Save the chunk
        torch.save(eeg_tensor, f"processed/eeg_chunk_{i//chunk_size}.pt")
        torch.save(labels_tensor, f"processed/labels_chunk_{i//chunk_size}.pt")
        print(f"Saved chunk {i//chunk_size}: shape {eeg_tensor.shape}")
