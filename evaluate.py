import torch
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torcheeg.models import LaBraM
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# ============================
# Dataset Definition
# ============================

class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]

# ============================
# Load Preprocessed Data
# ============================

eeg_files = sorted(glob.glob("processed/eeg_chunk_*.pt"))
label_files = sorted(glob.glob("processed/labels_chunk_*.pt"))

eeg_data = torch.cat([torch.load(f) for f in eeg_files], dim=0)
labels_data = torch.cat([torch.load(f) for f in label_files], dim=0)

print(f"Loaded EEG tensor of shape {eeg_data.shape}")
print(f"Loaded labels tensor of shape {labels_data.shape}")

# ============================
# Load Electrode Names
# ============================

example_fif = glob.glob("data/*_FG_preprocessed-epo.fif")[0]
epochs = mne.read_epochs(example_fif, preload=False)
electrode_names = [ch.upper() for ch in epochs.info['ch_names']]

# ============================
# Create Dataset and Dataloader (Only Test Set)
# ============================

dataset = EEGDataset(eeg_data, labels_data)
_, test_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=labels_data, random_state=42)
test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=16)

# ============================
# Load Model and Pretrained Weights
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LaBraM(in_channels=len(electrode_names), num_classes=2).to(device)

model_path = "models/eeg_labram_model_1st_experiment_90.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded pretrained model from {model_path}")
else:
    raise FileNotFoundError(f"Model weights not found at {model_path}")

model.eval()

# ============================
# Evaluation
# ============================

all_preds, all_trues = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs, electrodes=electrode_names)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_trues.extend(labels.cpu().numpy())

# ============================
# Results
# ============================

cm = confusion_matrix(all_trues, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Feedback", "Feedback"],
            yticklabels=["No Feedback", "Feedback"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

plt.savefig("confusion_matrix.png")

print("Classification Report:")
print(classification_report(all_trues, all_preds,
                            target_names=["No Feedback", "Feedback"]))
