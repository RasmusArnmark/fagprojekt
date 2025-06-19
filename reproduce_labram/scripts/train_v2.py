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

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torcheeg.models import LaBraM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from timm.scheduler import CosineLRScheduler

import os
os.environ["WANDB_API_KEY"] = "ef4ef7e778de182ce7b4968ada1557aff12b7449"

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

class EEGPatchDataset(Dataset):
    def __init__(self, root_dir, label, patch_size=200, stride=200):
        self.root_dir = root_dir
        self.label = label  # 0 for normal, 1 for abnormal
        self.patch_size = patch_size
        self.stride = stride
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.pt")))
        self.index = self._make_index()

    def _make_index(self):
        index = []
        for i, fpath in enumerate(self.files):
            try:
                data = torch.load(fpath)
                _, T = data.shape
                num_patches = (T - self.patch_size) // self.stride + 1
                for j in range(num_patches):
                    index.append((i, j))
            except:
                continue
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, patch_idx = self.index[idx]
        fpath = self.files[file_idx]
        data = torch.load(fpath)
        start = patch_idx * self.stride
        patch = data[:, start:start + self.patch_size]  # shape: (C, T)
        patch = patch.unsqueeze(1)  # shape: (C, 1, T)
        return patch, self.label


    

# ============================
# Hyperparameters
# ============================

use_fixed_constants = True

if use_fixed_constants == True:
    ### Constant
    base_learning_rate = 1e-3
    weight_decay = 0.05
    layer_decay = 0.65
    drop_path =  0.1
    batch_size = 48
    num_epochs = 20
    warmup_epochs = 5
    scheduler_lr_min =  1e-6
    scheduler_warmup_lr_init =  1e-6
    cross_entropy_loss_smoothing = 0.1
    disable_relative_positive_bias =  False
    absolute_positive_embedding =  True
    disable_qkv_bias =   False
    test_size =  0.2
    cross_entropy_loss_weight =  None
    scheduler_cycle_limit =  1
    print("Using fixed constants for hyperparameters.")
    print("Consider using HyperTuner.py to improve results.")
else:
    ### From File (Use HyperTuner.py)
    csv_path = os.path.join("HyperParameter", "hyperparameter_search_results.csv")
    hyper_df = pd.read_csv(csv_path)

    # Find the row with the best F1 score
    best_row = hyper_df.loc[hyper_df['f1_score'].idxmax()]

    # Set each hyperparameter individually (cast to correct type as needed)
    base_learning_rate = float(best_row['base_learning_rate'])
    weight_decay = float(best_row['weight_decay'])
    layer_decay = float(best_row['layer_decay'])
    drop_path = float(best_row['drop_path'])
    batch_size = int(best_row['batch_size'])
    num_epochs = int(best_row['num_epochs'])
    warmup_epochs = int(best_row['warmup_epochs'])
    scheduler_lr_min = float(best_row['scheduler_lr_min'])
    scheduler_warmup_lr_init = float(best_row['scheduler_warmup_lr_init'])
    cross_entropy_loss_smoothing = float(best_row['cross_entropy_loss_smoothing'])
    disable_relative_positive_bias = bool(best_row['disable_relative_positive_bias'])
    absolute_positive_embedding = bool(best_row['absolute_positive_embedding'])
    disable_qkv_bias = bool(best_row['disable_qkv_bias'])
    test_size = float(best_row['test_size'])
    cross_entropy_loss_weight = None if pd.isna(best_row['cross_entropy_loss_weight']) else float(best_row['cross_entropy_loss_weight'])
    scheduler_cycle_limit = int(best_row['scheduler_cycle_limit'])

# =========================
# wandb
# =========================


wandb.init(
    project="labram-tuab",
    name="initial_run",
    config={
        "learning_rate": base_learning_rate,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "weight_decay": weight_decay,
    }
)


# ============================
# Load Preprocessed Data (Alternative Folder)
# ============================

from collections import defaultdict
from torch.utils.data import ConcatDataset, Subset

from collections import defaultdict

# Load datasets
normal_dataset = EEGPatchDataset("reproduce_labram/data/processed_v3/normal", label=0)
abnormal_dataset = EEGPatchDataset("reproduce_labram/data/processed_v3/abnormal", label=1)

# Helper function to extract subject ID
def extract_subject_id(fpath):
    return os.path.basename(fpath).split("_seg")[0]

# Group files by subject
def group_by_subject(dataset):
    subject_dict = defaultdict(list)
    for idx, fpath in enumerate(dataset.files):
        subj_id = extract_subject_id(fpath)
        subject_dict[subj_id].append(idx)
    return subject_dict

normal_groups = group_by_subject(normal_dataset)
abnormal_groups = group_by_subject(abnormal_dataset)

# Merge and stratify
all_groups = list(normal_groups.items()) + list(abnormal_groups.items())
group_labels = [0] * len(normal_groups) + [1] * len(abnormal_groups)

# Subject-level stratified split
train_subjects, test_subjects = train_test_split(
    all_groups,
    test_size=0.2,
    stratify=group_labels,
    random_state=42
)

# Map subject groups to sample-level indices
train_indices = []
test_indices = []

normal_len = len(normal_dataset)

for subj_id, indices in train_subjects:
    if subj_id in normal_groups:
        train_indices.extend(indices)
    else:
        train_indices.extend([i + normal_len for i in indices])

for subj_id, indices in test_subjects:
    if subj_id in normal_groups:
        test_indices.extend(indices)
    else:
        test_indices.extend([i + normal_len for i in indices])

# Create final DataLoaders
from torch.utils.data import ConcatDataset, Subset

full_dataset = ConcatDataset([normal_dataset, abnormal_dataset])
train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(Subset(full_dataset, test_indices), batch_size=batch_size)


# ============================
# Model Initialization
# ============================
labram_channels = [
    'FP1', 'FP2',
    'F3', 'F4', 'F7', 'F8', 'FZ',
    'C3', 'C4', 'CZ',
    'P3', 'P4', 'PZ',
    'O1', 'O2',
    'T3', 'T4', 'T5', 'T6',
    'A1', 'A2'
]

example_file = 'reproduce_labram/data/raw/abnormal/01_tcp_ar/aaaaakfj_s001_t000.edf'
raw = mne.io.read_raw_edf(example_file, preload=True)
ch_names = raw.ch_names
electrode_names = [ch_name.replace('EEG ','').replace('-REF','') for ch_name in ch_names]
electrode_names = [elec for elec in electrode_names if elec in labram_channels]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drop_path = 0.1

model = LaBraM(
    in_channels=len(electrode_names),
    num_classes=2,
    drop_path=drop_path
).to(device)

# Ensure model directory exists
model_dir = "reproduce_labram/models"
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


parameter_groups = get_parameter_groups(model, base_learning_rate, weight_decay, layer_decay)
class_weights = None
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cross_entropy_loss_smoothing)
optimizer = torch.optim.AdamW(parameter_groups)
scheduler = CosineLRScheduler(
    optimizer,
    t_initial=num_epochs,
    lr_min=1e-6,
    warmup_lr_init=1e-5,
    warmup_t=2,
    cycle_limit=1,
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
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, electrodes=electrode_names)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Iteration {i}')
            print(f'Loss: {loss}')

            running_loss += loss.item()

        wandb.log({"train_loss": running_loss / len(train_loader), "epoch": epoch})


        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

model_save_path = os.path.join(model_dir, "labram_final_model.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

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
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Print full classification report
report = classification_report(all_trues, all_preds, target_names=["Normal", "Abnormal"], output_dict=True)
print("Classification Report:")
print(pd.DataFrame(report).transpose())

# Extract macro metrics
accuracy = accuracy_score(all_trues, all_preds)
precision = precision_score(all_trues, all_preds, average='macro')
recall = recall_score(all_trues, all_preds, average='macro')
f1 = f1_score(all_trues, all_preds, average='macro')

# Log to wandb
wandb.log({
    "test_accuracy": accuracy,
    "test_precision_macro": precision,
    "test_recall_macro": recall,
    "test_f1_macro": f1,
    "confusion_matrix": wandb.plot.confusion_matrix(
        preds=all_preds, y_true=all_trues,
        class_names=["Normal", "Abnormal"]
    )
})
