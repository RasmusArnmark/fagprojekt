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

processed_dir = "processed"
eeg_dir = os.path.join(processed_dir, "eeg_chunk")
labels_dir = os.path.join(processed_dir, "labels_chunk")

eeg_files = sorted(glob.glob(os.path.join(eeg_dir, "eeg_chunk_*.pt")))
label_files = sorted(glob.glob(os.path.join(labels_dir, "labels_chunk_*.pt")))

eeg_data = torch.cat([torch.load(f) for f in eeg_files], dim=0)
labels_data = torch.cat([torch.load(f) for f in label_files], dim=0)

print(f"Loaded EEG tensor of shape {eeg_data.shape}")
print(f"Loaded labels tensor of shape {labels_data.shape}")

# ============================
# Load Electrode Names
# ============================

data_dir = os.path.join("data", "PreprocessedEEGData")
example_fif = glob.glob(os.path.join(data_dir, "*_FG_preprocessed-epo.fif"))[0]
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

# Ensure model directory exists
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Load pretrained weights if available
pretrained_path = os.path.join(model_dir, "labram-base.pth")
if os.path.exists(pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
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


# ============================
# Loss and Optimizer Initialization
# ============================
# CrossEntropyLoss for criterion
c_e_l_weight = None
c_e_l_smoothing = 0.0
criterion = nn.CrossEntropyLoss(weight=c_e_l_weight, label_smoothing=c_e_l_smoothing)
# AdamW for optimizer initialization
weight_decay = 0.05  # Recommended in LaBraM paper
learning_rate = 0.001  # Or use the value you want

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ============================
# Training Loop
# ============================

# num_epochs = 10
# train = True
# if train:
#     print("Starting training...")
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs, electrodes=electrode_names)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

#     # Save model
#     torch.save(model.state_dict(), os.path.join(model_dir, "eeg_labram_model.pth"))
#     print("Training complete and model saved.")

# ============================
# Quick Test Training (for debugging, comment out in production)
# ============================

# Defining hyperparameters
quick_num_epochs = 4
test_subset_size = 256
quick_batch_size = 8

# Creating dataset
small_train_loader = DataLoader(Subset(dataset, list(train_idx)[:test_subset_size]), batch_size=quick_batch_size, shuffle=True)

# Quick test training loop
print("Starting quick test training loop...")
for epoch in range(quick_num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in small_train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, electrodes=electrode_names)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Quick Test] Epoch {epoch + 1}/{quick_num_epochs}, Loss: {running_loss / len(small_train_loader):.4f}")

# Save model in a cross-platform way (optional for quick test)
# torch.save(model.state_dict(), os.path.join(model_dir, "eeg_labram_model_quicktest.pth"))
# print("Quick test training complete and model saved.")

# Print model hyperparameters after training
warmup_epochs = 0  # To be implemented
layer_decay = 0  # To be implemented
drop_path = 0  # To be implemented
seed = 0  # To be implemented

TriosForLater = 0  # To be implemented

print("\nModel Information:")
print(f"  Model: {model.__class__.__name__}")
print(f"  Number of classes: {model.num_classes if hasattr(model, 'num_classes') else 2}")
print(f"  Seed: {seed}")
print(f"  Device: {device}")

print("\nTraining Hyperparameters:")
print(f"  CrossEntropyLoss weight: {c_e_l_weight}")
print(f"  CrossEntropyLoss label smoothing: {c_e_l_smoothing}")
print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
print(f"  Weight decay: {weight_decay}")
print(f"  Batch size: {small_train_loader.batch_size}")
print(f"  Epochs: {quick_num_epochs}")
print(f"  Warmup epochs: {warmup_epochs}")
print(f"  Layer decay: {layer_decay}")
print(f"  Drop path: {drop_path}")
print(f" D_R_B: {TriosForLater}")
print(f" A_P_E: {TriosForLater}")
print(f" D_Q_B: {TriosForLater}")

# ============================
# Evaluation
# ============================

model.eval()
all_preds, all_trues = [], []

print("Starting evaluation...")

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


