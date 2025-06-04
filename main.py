# main.py

import torch
import numpy as np
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import pickle
import random

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torcheeg.models import LaBraM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


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

# Load one example file to get electrode names
example_epochs = pd.read_pickle("data/FG_overview_df_v2.pkl")
example_fif = glob.glob("data/*_FG_preprocessed-epo.fif")[0]
epochs = mne.read_epochs(example_fif, preload=False)
electrode_names = [ch.upper() for ch in epochs.info['ch_names']]

# ============================
# Create Dataset and Dataloaders
# ============================

dataset = EEGDataset(eeg_data, labels_data)
train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=labels_data, random_state=42)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=16)

# ============================
# Model Initialization
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LaBraM(in_channels=len(electrode_names), num_classes=2).to(device)

# Load pretrained weights if available
checkpoint = torch.load('models/labram-base.pth', map_location=device, weights_only=False)
print(f"found this: {checkpoint.keys()}")
state_dict = checkpoint["model"]

# Remove 'student.' prefix from all keys if present
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("student."):
        new_state_dict[k[len("student."):]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=False)
print("Loaded pretrained model (student weights).")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ============================
# Training Loop
# ============================

print('Starts training...')

num_epochs = 10
train = True
if train:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, electrodes=electrode_names)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "eeg_labram_model.pth")
print("Training complete and model saved.")

# ============================
# Evaluation
# ============================

model.eval()
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
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Feedback", "Feedback"], yticklabels=["No Feedback", "Feedback"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

print("Classification Report:")
print(classification_report(all_trues, all_preds, target_names=["No Feedback", "Feedback"]))