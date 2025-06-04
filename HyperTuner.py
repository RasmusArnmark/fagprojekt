# ======================================================================== #
### Major HyperParameter Tuner for LaBraM Model on Group Equestrian data ###
# ======================================================================== #

# In any old good fashion, the imports will come first, look for the second section
# to find the hyperparameter space.

# ===================== #
# Imports and Functions #
# ===================== #

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
import copy
import wandb

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torcheeg.models import LaBraM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from timm.scheduler import CosineLRScheduler
from itertools import product
from pandas.plotting import parallel_coordinates
from sklearn.metrics import f1_score

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

class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]
    
# =========================== #
# Hyper Parameters & Pointers #
# =========================== #

### Seed
lucky_number = 69 # Nice

### Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Pointers, edit these to fit your directory
processed_dir = "processed" # From data.py, where the processed EEG and labels are stored
model_dir = "models" # Where the model is saved.

### Trials
n_trials = 30

### WandB config
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "avg_loss",   # or "f1_score" if you want to maximize F1
        "goal": "minimize"
    },
    "parameters": {
        "base_learning_rate": {
            "min": 1e-5,
            "max": 1e-2,
            "distribution": "log_uniform"
        },
        "weight_decay": {
            "min": 0.001,
            "max": 0.1,
            "distribution": "uniform"
        },
        "layer_decay": {
            "min": 0.5,
            "max": 0.9,
            "distribution": "uniform"
        },
        "drop_path": {
            "min": 0.01,
            "max": 0.3,
            "distribution": "uniform"
        },
        "batch_size": {
            "values": [8, 16, 32]
        },
        "num_epochs": {
            "value": 10
        },
        "warmup_epochs": {
            "min": 2,
            "max": 10,
            "distribution": "int_uniform"
        },
        "scheduler_lr_min": {
            "min": 1e-6,
            "max": 1e-3,
            "distribution": "log_uniform"
        },
        "scheduler_warmup_lr_init": {
            "min": 1e-6,
            "max": 1e-3,
            "distribution": "log_uniform"
        },
        "cross_entropy_loss_smoothing": {
            "min": 0.0,
            "max": 0.2,
            "distribution": "uniform"
        },
        "disable_relative_positive_bias": {
            "values": [False, True]
        },
        "absolute_positive_embedding": {
            "values": [False, True]
        },
        "disable_qkv_bias": {
            "values": [False]
        },
        "test_size": {
            "value": 0.2
        },
        "cross_entropy_loss_weight": {
            "value": None
        },
        "scheduler_cycle_limit": {
            "value": 1
        }
    }
}

# ================================ #
# Training and Evaluation Function #
# ================================ #

# More or less copied from the main document
# For any changes in the main, this should be updated as well.

def train_and_evaluate(seed=42, device=None):
    config = wandb.config
    base_learning_rate = config.base_learning_rate
    weight_decay = config.weight_decay
    layer_decay = config.layer_decay
    drop_path = config.drop_path
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    warmup_epochs = config.warmup_epochs
    scheduler_lr_min = config.scheduler_lr_min
    scheduler_warmup_lr_init = config.scheduler_warmup_lr_init
    cross_entropy_loss_smoothing = config.cross_entropy_loss_smoothing
    disable_relative_positive_bias = config.disable_relative_positive_bias
    absolute_positive_embedding = config.absolute_positive_embedding
    disable_qkv_bias = config.disable_qkv_bias
    test_size = config.test_size
    cross_entropy_loss_weight = config.cross_entropy_loss_weight
    scheduler_cycle_limit = config.scheduler_cycle_limit

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reload data (to avoid data leakage between trials)
    eeg_dir = os.path.join(processed_dir, "eeg_chunk")
    labels_dir = os.path.join(processed_dir, "labels_chunk")
    eeg_files = sorted(glob.glob(os.path.join(eeg_dir, "eeg_chunk_*.pt")))
    label_files = sorted(glob.glob(os.path.join(labels_dir, "labels_chunk_*.pt")))
    eeg_data = torch.cat([torch.load(f) for f in eeg_files], dim=0)
    labels_data = torch.cat([torch.load(f) for f in label_files], dim=0)

    # Load electrode names
    data_dir = os.path.join("data", "PreprocessedEEGData")
    example_fif = glob.glob(os.path.join(data_dir, "*_FG_preprocessed-epo.fif"))[0]
    epochs = mne.read_epochs(example_fif, preload=False)
    electrode_names = [ch.upper() for ch in epochs.info['ch_names']]

    # Create Dataset and Dataloaders
    dataset = EEGDataset(eeg_data, labels_data)
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        stratify=labels_data,
        random_state=seed
    )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)

    print(f"Data Sorted")

    # Model
    model = LaBraM(
        in_channels=len(electrode_names),
        num_classes=2,
        drop_path=drop_path,
        abs_pos_emb=absolute_positive_embedding,
        disable_rel_pos_bias=disable_relative_positive_bias,
        disable_qkv_bias=disable_qkv_bias
    ).to(device)

    # Load pretrained weights if available
    pretrained_path = os.path.join(model_dir, "labram-base.pth")
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("student."):
                new_state_dict[k[len("student."):]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=cross_entropy_loss_weight, label_smoothing=cross_entropy_loss_smoothing)
    parameter_groups = get_parameter_groups(model, base_learning_rate, weight_decay, layer_decay)
    optimizer = torch.optim.AdamW(parameter_groups)

    # Scheduler
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=scheduler_lr_min,
        warmup_lr_init=scheduler_warmup_lr_init,
        warmup_t=warmup_epochs,
        cycle_limit=scheduler_cycle_limit,
        t_in_epochs=True,
    )

    print(f"Model initialized")

    # Training Loop
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
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
        avg_loss = running_loss / len(train_loader)
        print(f"  -> Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

    print("Finished training. Evaluating on test set...")

    # Evaluation
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, electrodes=electrode_names)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(labels.cpu().numpy())

    # Compute weighted F1-score
    f1 = f1_score(all_trues, all_preds, average='weighted')
    print(f"  -> Finished evaluating. Weighted F1-score on test set: {f1:.4f}")
    return f1, avg_loss, model

# ===================== #
# Hyperparameter Tuning #
# ===================== #

def main():
    HyperParameter_dir = "HyperParameter"
    if not os.path.exists(HyperParameter_dir):
        os.makedirs(HyperParameter_dir)
        print(f"folder created: {HyperParameter_dir}")

    # Initialize wandb for this run (add entity if needed)
    wandb.init(project="labram-hyperparameter-tuning")
    try:
        print("\n=== New Trial ===")
        print("Testing hyperparameters:")
        for k, v in dict(wandb.config).items():
            print(f"  {k}: {v}")

        print("Initializing model, training and evaluation...")
        f1, avg_loss, model = train_and_evaluate(seed=lucky_number, device=device)

        print(f"Trial F1-score: {f1:.4f}")
        wandb.log({"avg_loss": avg_loss, "f1_score": f1})

        # Save model in the wandb run directory
        model_path = os.path.join(wandb.run.dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        wandb.save(model_path)

        # Save results to CSV (optional, for local backup)
        csv_path = os.path.join(HyperParameter_dir, "hyperparameter_search_results.csv")
        results_df = pd.DataFrame([{**dict(wandb.config), "f1_score": f1, "avg_loss": avg_loss}])
        if os.path.exists(csv_path):
            results_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(csv_path, index=False)

        print(f"Results saved to {csv_path}")
    except Exception as e:
        print(f"Run failed: {e}")
        wandb.alert(title="Run failed", text=str(e))
    finally:
        wandb.finish()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="labram-hyperparameter-tuning")
    print(f"Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, function=main)