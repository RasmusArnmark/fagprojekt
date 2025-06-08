import numpy as np
import mne
import pandas as pd
import glob
import os
import torch

# Load metadata
df_info = pd.read_pickle("data/FG_overview_df_v2.pkl")

# Mapping of condition names to MNE event IDs
event_labels = {
    'T1P': 301, 'T1Pn': 302, 'T3P': 303, 'T3Pn': 304,
    'T12P': 305, 'T12Pn': 306, 'T13P': 307, 'T13Pn': 308,
    'T23P': 309, 'T23Pn': 310
}

# Manually define which event IDs each EEG device participated in
device_event_labels = {
    1: {301, 302, 303, 304, 305, 306, 307, 308},
    2: {303, 304, 305, 306, 309, 310},
    3: {303, 304, 307, 308, 309, 310},
}

file_paths = glob.glob("data/*_FG_preprocessed-epo.fif")
chunk_size = 4
num_patches = 6

for i in range(0, len(file_paths), chunk_size):
    batch_files = file_paths[i:i + chunk_size]

    fb_data, fb_labels = [], []
    nf_data, nf_labels = [], []

    for file_path in batch_files:
        file_name = file_path.split("/")[-1].split("_")[0]
        exp_id = file_name[:4]
        participants = df_info[df_info["Exp_id"] == exp_id]

        if participants.empty:
            continue

        try:
            epochs = mne.read_epochs(file_path, preload=True)
            epochs.resample(200)
            eeg_data = epochs.get_data()  # [trials, channels, time]
            labels = epochs.events[:, -1]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        for _, row in participants.iterrows():
            eeg_device = row["EEG_device"]
            valid_events = device_event_labels[eeg_device]
            valid_idxs = [j for j, lbl in enumerate(labels) if lbl in valid_events]

            if not valid_idxs:
                continue

            subject_data = eeg_data[valid_idxs]
            subject_labels = labels[valid_idxs]

            # Normalize
            subject_data = (subject_data - subject_data.mean()) / subject_data.std()

            # Categorize into feedback or no-feedback streams
            for j, lbl in enumerate(subject_labels):
                x = subject_data[j]

                if lbl in {301, 303, 305, 307, 309}:  # feedback
                    label = 0 if lbl == 301 else 1  # solo vs social
                    fb_data.append(x)
                    fb_labels.append(label)

                elif lbl in {302, 304, 306, 308, 310}:  # no feedback
                    label = 0 if lbl == 302 else 1
                    nf_data.append(x)
                    nf_labels.append(label)

    # Save feedback chunk
    if fb_data:
        fb_tensor = torch.tensor(np.stack(fb_data), dtype=torch.float32)
        fb_labels_tensor = torch.tensor(fb_labels, dtype=torch.long)
        time_per_patch = fb_tensor.shape[2] // num_patches
        fb_tensor = fb_tensor.reshape(fb_tensor.shape[0], fb_tensor.shape[1], num_patches, time_per_patch)
        torch.save(fb_tensor, f"processed2/social_feedback_chunk_{i//chunk_size}.pt")
        torch.save(fb_labels_tensor, f"processed2/social_feedback_labels_{i//chunk_size}.pt")
        print(f"Saved feedback chunk {i//chunk_size} - shape {fb_tensor.shape}")

    # Save no-feedback chunk
    if nf_data:
        nf_tensor = torch.tensor(np.stack(nf_data), dtype=torch.float32)
        nf_labels_tensor = torch.tensor(nf_labels, dtype=torch.long)
        time_per_patch = nf_tensor.shape[2] // num_patches
        nf_tensor = nf_tensor.reshape(nf_tensor.shape[0], nf_tensor.shape[1], num_patches, time_per_patch)
        torch.save(nf_tensor, f"processed2/social_nofeedback_chunk_{i//chunk_size}.pt")
        torch.save(nf_labels_tensor, f"processed2/social_nofeedback_labels_{i//chunk_size}.pt")
        print(f"Saved no-feedback chunk {i//chunk_size} - shape {nf_tensor.shape}")
