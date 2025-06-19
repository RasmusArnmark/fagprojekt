import os, glob, random, pickle
import numpy as np
import pandas as pd
import torch, mne, wandb, matplotlib.pyplot as plt, seaborn as sns
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit        # key line
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torcheeg.models import LaBraM
from timm.scheduler import CosineLRScheduler
from fagprojekt.experiment1.data_exp1 import load_dataset                               # helper from data.py
# ----------------------------------------------------------------------

# 1) -------------------------------------------------------------------
#              LOAD tensors (eeg, labels, subject IDs)
# ----------------------------------------------------------------------
eeg_data, labels_data, subj_ids = load_dataset()
print("EEG :",     eeg_data.shape)
print("labels :",  labels_data.shape)
print("subjects:", subj_ids.shape, "unique =", subj_ids.unique().numel())

# 2) -------------------------------------------------------------------
#              SUBJECT-LEVEL train / test split
# ----------------------------------------------------------------------
gss = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(
    np.zeros(len(labels_data)),           # dummy X
    labels_data.numpy(),                  # y (optional)
    groups=subj_ids.numpy()               # ← group = subject
))

print("train subjects :", np.unique(subj_ids[train_idx].numpy()).size,
      "| test subjects :", np.unique(subj_ids[test_idx].numpy()).size)

# 3) -------------------------------------------------------------------
#              DataSet & DataLoader
# ----------------------------------------------------------------------
class EEGDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

dataset      = EEGDataset(eeg_data, labels_data)
batch_size   = 64
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(Subset(dataset,  test_idx), batch_size=batch_size)

# 4) -------------------------------------------------------------------
#              Model
# ----------------------------------------------------------------------
example_fif  = glob.glob("../data/preprocessed_data/*_FG_preprocessed-epo.fif")[0]
electrode_names = [ch.upper() for ch in mne.read_epochs(example_fif, preload=False).info["ch_names"]]

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model  = LaBraM(in_channels=len(electrode_names), num_classes=2, drop_path=0.1).to(device)

# optional: load LaBraM base weights as before (omitted here for brevity)


pretrained_path = os.path.join("../models", "labram-base.pth")
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

# 5) -------------------------------------------------------------------
#              Optimiser, scheduler, loss
# ----------------------------------------------------------------------
base_lr = 3e-4
num_epochs = 15
optimizer  = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
scheduler  = CosineLRScheduler(optimizer, t_initial=10, lr_min=1e-6, warmup_lr_init=1e-5, warmup_t=2)
criterion   = nn.CrossEntropyLoss(label_smoothing=0.0)

# 6) -------------------------------------------------------------------
#              WandB
# ----------------------------------------------------------------------
wandb.init(project="Fagprojekt_eeg",
           name="LaBraM_subject_split",
           config={"base_lr": base_lr, "batch": batch_size, "epochs": 10,
                   "n_subj_train": int(np.unique(subj_ids[train_idx].numpy()).size),
                   "n_subj_test":  int(np.unique(subj_ids[test_idx].numpy()).size)})

# 7) -------------------------------------------------------------------
#              Training loop
# ----------------------------------------------------------------------
for epoch in range(15):
    scheduler.step(epoch)
    model.train()
    running = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x, electrodes=electrode_names), y)
        loss.backward(); optimizer.step()
        running += loss.item()
    wandb.log({"epoch": epoch+1, "train/loss": running/len(train_loader),
               "train/lr": optimizer.param_groups[0]["lr"]})
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running/len(train_loader):.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

# 8) -------------------------------------------------------------------
#              Evaluation
# ----------------------------------------------------------------------
model.eval(); all_pred, all_true = [], []
with torch.no_grad():
    for x, y in test_loader:
        p = model(x.to(device), electrodes=electrode_names).argmax(1).cpu()
        all_pred.extend(p); all_true.extend(y)

torch.save(model.state_dict(), os.path.join("../models", f"EEG_model_FB_vs_NoFB.pth"))
print("Training complete and model saved.")
wandb.save(os.path.join("model_dir", "EEG_model_FB_vs_NoFB.pth"))

acc = accuracy_score(all_true, all_pred)
f1  = f1_score(all_true, all_pred)

wandb.log({"val/accuracy": acc, "val/f1": f1})
cm = confusion_matrix(all_true, all_pred)

# log confusion matrix as an image
fig = plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No FB", "FB"], yticklabels=["No FB", "FB"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion")
plt.tight_layout()

wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.show() 

print(classification_report(all_true, all_pred, target_names=["No FB", "FB"]))

wandb.finish()