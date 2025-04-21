import mne 
import os

file_path = "/Users/philipkierkegaard/Desktop/sem4/fagprojekt/fagprojekt/data/eval/normal/01_tcp_ar/aaaaacad_s003_t000.edf"

raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

data = raw.get_data()

channel_names = raw.ch_names

def clean_channel_name(ch):
    if ch.startswith("EEG "):
        ch = ch.replace("EEG ", "")
    if ch.endswith("-REF"):
        ch = ch.replace("-REF", "")
    return ch

cleaned_channel_names = [clean_channel_name(ch) for ch in channel_names]

standard_10_20 = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4',
                  'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                  'T3', 'T4', 'T5', 'T6', 'A1', 'A2',
                  'FZ', 'CZ', 'PZ', 'T1', 'T2']

num_channels = len(standard_10_20)

print((cleaned_channel_names))


import mne
import os

normal_path = '/Users/philipkierkegaard/Desktop/sem4/fagprojekt/fagprojekt/data/eval/normal/01_tcp_ar'
abnormal_path = '/Users/philipkierkegaard/Desktop/sem4/fagprojekt/fagprojekt/data/eval/abnormal/01_tcp_ar'

import mne
import os
import numpy as np

# Define the standard 10-20 channel set you want to keep (23 channels total)
standard_10_20 = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4',
                  'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                  'T3', 'T4', 'T5', 'T6', 'A1', 'A2',
                  'FZ', 'CZ', 'PZ', 'T1', 'T2']


def clean_channel_name(ch):
    if ch.startswith("EEG "):
        ch = ch.replace("EEG ", "")
    if ch.endswith("-REF"):
        ch = ch.replace("-REF", "")
    return ch.upper()



def load_edf_data(folder_path):
    edf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".edf")]
    data_list = []

    for file in edf_files:
        try:
            raw = mne.io.read_raw_edf(file, preload=True, verbose=False)

            raw.resample(200, npad="auto")

            data = raw.get_data()  # shape: (n_channels, n_samples)
            channel_names = [clean_channel_name(ch) for ch in raw.ch_names]

            # Filter to standard 10–20 channels
            index_map = [channel_names.index(ch) for ch in standard_10_20 if ch in channel_names]

            if len(index_map) < len(standard_10_20):
                print(f"Skipping {os.path.basename(file)} (only found {len(index_map)} valid channels)")
                continue  # skip incomplete samples

            # Reorder and slice signal
            filtered_data = data[index_map, :]
            data_list.append(filtered_data)

        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    return data_list




normal_data = load_edf_data(normal_path)
abnormal_data = load_edf_data(abnormal_path)

X = normal_data + abnormal_data
y = [0] * len(normal_data) + [1] * len(abnormal_data)



normal_data = load_edf_data(normal_path)
abnormal_data = load_edf_data(abnormal_path)

X = normal_data + abnormal_data
y = [0] * len(normal_data) + [1] * len(abnormal_data)



import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torcheeg.models import LaBraM
from tqdm import tqdm  # 👈 Progress bar!

weights = '/Users/philipkierkegaard/Desktop/sem4/fagprojekt/fagprojekt/labram-base.pth'

def strip_student_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("student."):
            new_k = k[len("student."):]  # Remove 'student.' prefix
            new_state_dict[new_k] = v
    return new_state_dict

# 1. Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
electrode_names = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                   'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                   'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2']

model = LaBraM.base_patch200_200(in_channels=23, num_classes=2).to(device)
checkpoint = torch.load(weights, map_location=device, weights_only=False)
stripped_weights = strip_student_prefix(checkpoint["model"])
model.load_state_dict(stripped_weights, strict=False)
model.eval()

# 2. Segment function (10s windows at 200Hz = 2000 samples)
def segment_signal(signal, window_size=2000, stride=2000):
    segments = []
    for start in range(0, signal.shape[1] - window_size + 1, stride):
        segment = signal[:, start:start + window_size]
        segments.append(segment)
    return segments

# 3. Evaluate accuracy
def evaluate_model(model, X, y):
    preds = []

    with torch.no_grad():
        for i in tqdm(range(len(X)), desc="Evaluating"):  # 👈 Progress bar here
            signal = X[i]
            segments = segment_signal(signal, window_size=2000)
            outputs = []

            for segment in segments:
                x = torch.tensor(segment).float()
                x = x[:, :2000].reshape(23, -1, 200).unsqueeze(0).to(device)
                out = model(x, electrodes=electrode_names)
                outputs.append(torch.softmax(out, dim=1))

            avg_output = torch.stack(outputs).mean(dim=0)
            pred_class = torch.argmax(avg_output, dim=1).item()
            preds.append(pred_class)

    acc = accuracy_score(y, preds)
    print(f"Accuracy on {len(X)} files: {acc * 100:.2f}%")
    return preds


# X: your list of 276 np arrays (shape: [23, 251800] each)
# y: list of labels (0 for normal, 1 for abnormal)

predictions = evaluate_model(model, X, y)
