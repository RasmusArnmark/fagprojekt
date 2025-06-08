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
import wandb
from sklearn.metrics import accuracy_score, f1_score

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torcheeg.models import LaBraM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from timm.scheduler import CosineLRScheduler

# ============================
# Dug real deep for this one
# ============================

def get_parameter_groups(model, base_lr, weight_decay, layer_decay):
    parameter_group_names = {}
    parameter_groups = []

    num_layers = model.get_num_layers() if hasattr(model, "get_num_layers") else 12  # fallback
    # Assign each parameter to a layer id
    def get_layer_id_for_vit(var_name):
        if var_name in ['cls_token', 'pos_embed']:
            return 0
        elif var_name.startswith('patch_embed'):
            return 0
        elif var_name.startswith('blocks'):
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        else:
            return num_layers

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_name = "layer_%d" % get_layer_id_for_vit(name)
        if group_name not in parameter_group_names:
            scale = layer_decay ** (num_layers - get_layer_id_for_vit(name))
            parameter_group_names[group_name] = {
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": weight_decay,
            }
        parameter_group_names[group_name]["params"].append(param)

    for group_name in parameter_group_names:
        parameter_groups.append(parameter_group_names[group_name])
    return parameter_groups

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
# Hyperparameters
# ============================

base_learning_rate = 1e-3
weight_decay = 0.05
layer_decay = 0.65
drop_path =  0.1
batch_size = 64
num_epochs = 10
warmup_epochs = 5
scheduler_lr_min =  1e-6
scheduler_warmup_lr_init =  1e-6
cross_entropy_loss_smoothing = 0.1
disable_relative_positive_bias =  False

test_size =  0.2
cross_entropy_loss_weight =  None
scheduler_cycle_limit =  1

# ============================
# WandB Initialization
# ============================

wandb.init(
    project="Fagprojekt_eeg",      
    entity="fagprojekt_eeg",       
    name="LaBraM_finetune",        
    config={                      
        "base_lr": base_learning_rate,
        "batch_size": batch_size,
        "epochs": num_epochs,

    }
)


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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
drop_path = 0.1

model = LaBraM(
    in_channels=len(electrode_names),
    num_classes=2,
    drop_path=drop_path
).to(device)

wandb.watch(model, log="all", log_freq=10)   # log gradients & weights every 10 steps


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
criterion = nn.CrossEntropyLoss(weight=cross_entropy_loss_weight, label_smoothing=cross_entropy_loss_smoothing)

# AdamW for optimizer initialization
parameter_groups = get_parameter_groups(model, base_learning_rate, weight_decay, layer_decay)
optimizer = torch.optim.AdamW(parameter_groups)
scheduler = CosineLRScheduler(
    optimizer,
    t_initial=num_epochs,
    lr_min=1e-6,                # You can adjust this or make it a variable
    warmup_lr_init=1e-5,        # You can adjust this or make it a variable
    warmup_t=2,                 # Number of warmup epochs, adjust as needed
    cycle_limit=1,              # Number of cycles
    t_in_epochs=True,
)

# ============================
# Training Loop
# ============================

train = True
if train:
    print("Starting training...")
    for epoch in range(num_epochs):
        scheduler.step(epoch)
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, electrodes=electrode_names)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": epoch_loss,
            "train/lr": optimizer.param_groups[0]["lr"]
        })

#     # Save model
torch.save(model.state_dict(), os.path.join(model_dir, "eeg_labram_model.pth"))
print("Training complete and model saved.")

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

acc = accuracy_score(all_trues, all_preds)
f1  = f1_score(all_trues, all_preds)

wandb.log({"val/accuracy": acc, "val/f1": f1})
cm = confusion_matrix(all_trues, all_preds)

# log confusion matrix as an image
fig = plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No FB","FB"], yticklabels=["No FB","FB"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion")
plt.tight_layout()

wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.show() 

print(classification_report(all_trues, all_preds, target_names=["No Feedback", "Feedback"]))

wandb.finish()
