import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import pickle

import torch
import numpy as np
import glob
import pandas as pd

epochs = mne.read_epochs("data/301A_FG_preprocessed-epo.fif", preload=True)
epochs.average().plot()
epochs.average().plot_topomap(times=np.linspace(0.2,5,8), ch_type="eeg", show=False)


plt.show()

epochs = mne.read_epochs("data/301A_FG_preprocessed-epo.fif", preload=True)

epochs_resampled = epochs.copy().resample(200)

data_resampled = epochs_resampled.get_data()

print(f'Resampled EEG shape: {data_resampled.shape}')

labels = epochs.events[:,-1]
binary_labels = np.array([0 if l in [301, 303, 305, 307, 309] else 1 for l in labels])

# Load a pickle file
with open("data/FG_overview_df_v2.pkl", "rb") as file:
    data = pickle.load(file)

print(type(data))  # Check the type of the loaded object
data  # Print the contents


import torch
import mne
import numpy as np
import glob
import pandas as pd

# Load metadata DataFrame
df_info = pd.read_pickle("data/FG_overview_df_v2.pkl")  # Update with actual path

# Define event IDs
event_labels = {'T1P': 301, 'T1Pn': 302, 'T3P': 303, 'T3Pn': 304,
                'T12P': 305, 'T12Pn': 306, 'T13P': 307, 'T13Pn': 308,
                'T23P': 309, 'T23Pn': 310}

file_paths = glob.glob("data/*_FG_preprocessed-epo.fif")  # Update with actual data path
print(f"Found {len(file_paths)} EEG files.")

all_eeg_data, all_labels = [], []

for file_path in file_paths[:1]:
    # Extract filename (e.g., "301A")
    file_name = file_path.split("/")[-1].split("_")[0]  # Extract "301A"

    # Extract Experiment ID (e.g., "301")
    exp_id = file_name[:4]

    # Get participants for this experiment
    experiment_participants = df_info[df_info["Exp_id"] == exp_id]

    if experiment_participants.empty:
        print(f"Skipping {file_name}: No participants found for {exp_id}.")
        continue

    # Load EEG file
    epochs = mne.read_epochs(file_path, preload=True)
    epochs.resample(200)
    eeg_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1]  # Extract event labels

    # Process each participant in the experiment
    for _, row in experiment_participants.iterrows():
        subject_id = row["Subject_id"]
        eeg_device = row["EEG_device"]  # 1, 2, or 3

        # Map EEG device to the corresponding event labels
        device_event_labels = {
            1: {301, 302, 305, 306, 307, 308},  # T1 labels
            2: {303, 304, 305, 306, 309, 310},  # T2 labels
            3: {307, 308, 309, 310, 303, 304},  # T3 labels
        }

        valid_events = device_event_labels[eeg_device]

        # Filter trials for this subject
        valid_trials = [i for i, label in enumerate(labels) if label in valid_events]

        if len(valid_trials) == 0:
            print(f"Skipping subject {subject_id} in {exp_id}: No relevant trials for EEG device {eeg_device}.")
            continue

        # Keep only the relevant trials
        eeg_subject_data = eeg_data[valid_trials]
        labels_subject = labels[valid_trials]

        # Normalize per file
        eeg_subject_data = (eeg_subject_data - eeg_subject_data.mean()) / eeg_subject_data.std()

        # Convert labels to binary classification (feedback vs. no feedback)
        binary_labels = np.array([1 if label in {301, 303, 305, 307, 309} else 0 for label in labels_subject])

        all_eeg_data.append(eeg_subject_data)
        all_labels.append(binary_labels)

# Convert to PyTorch tensors
eeg_tensor = torch.tensor(np.concatenate(all_eeg_data, axis=0), dtype=torch.float32)
labels_tensor = torch.tensor(np.concatenate(all_labels, axis=0), dtype=torch.long)

num_patches = 6  # Choose the number of patches
time_steps_per_patch = eeg_tensor.shape[2] // num_patches  # Divide time into patches

print(eeg_tensor.shape)
# Reshape EEG tensor
eeg_tensor = eeg_tensor.reshape(eeg_tensor.shape[0], eeg_tensor.shape[1], num_patches, time_steps_per_patch)
eeg_tensor.shape

print(f"Final EEG Tensor Shape: {eeg_tensor.shape}")

epochs = mne.read_epochs(file_paths[0])
labels = epochs.info['ch_names']
electrode_names = [l.upper() for l in labels]

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torcheeg.models import LaBraM

# Define EEG Dataset class
class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]

# Convert dataset to PyTorch Dataset format
dataset = EEGDataset(eeg_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# Initialize LabRam Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LaBraM(in_channels=len(electrode_names), num_classes=2).to(device)


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        print(inputs.shape)

        optimizer.zero_grad()
        outputs = model(inputs, electrodes=electrode_names)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

print("Training Complete!")


from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Save the trained model
torch.save(model.state_dict(), "eeg_labram_model.pth")

# Evaluation
model.eval()
all_preds = []
all_labels = []

# Create test dataloader - using 20% of data for testing

# Generate indices for train and test
indices = list(range(len(dataset)))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels_tensor)

# Create test dataset and dataloader
test_dataset = Subset(dataset, test_indices)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluate
with torch.no_grad():
    correct = 0
    total = 0
    
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs, electrodes=electrode_names)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Confusion matrix and classification report

# Convert to numpy arrays for sklearn
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate and display confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Feedback', 'Feedback'],
            yticklabels=['No Feedback', 'Feedback'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, 
                           target_names=['No Feedback', 'Feedback']))