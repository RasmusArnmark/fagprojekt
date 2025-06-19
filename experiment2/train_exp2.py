import os, glob, random
import numpy as np
import pandas as pd
import torch, mne, wandb, matplotlib.pyplot as plt, seaborn as sns
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torcheeg.models import LaBraM
from timm.scheduler import CosineLRScheduler

# ----------------------------------------------------------------------
# 1.  Choose condition: "fb"  or  "nfb"
# ----------------------------------------------------------------------
condition   = "nfb"          # <-- set "fb" or "nfb"
data_folder = f"train_ready_data_exp2/social_{condition}"

all_pt   = sorted(glob.glob(os.path.join(data_folder, f"social_{condition}_*.pt")))
eeg_files  = [f for f in all_pt if not f.endswith(("_labels.pt", "_subjects.pt"))]
lab_files  = [f for f in all_pt if f.endswith("_labels.pt")]
sid_files  = [f for f in all_pt if f.endswith("_subjects.pt")]

print("EEG  files:", eeg_files[:3])
print("LAB  files:", lab_files[:3])
print("SUBJ files:", sid_files[:3])

eeg_data   = torch.cat([torch.load(f) for f in eeg_files], dim=0)
labels     = torch.cat([torch.load(f) for f in lab_files], dim=0)
subject_id = torch.cat([torch.load(f) for f in sid_files], dim=0)
print("EEG   :", eeg_data.shape)
print("Label :", labels.shape)
print("Subj  :", subject_id.shape, "unique =", subject_id.unique().numel())

# ----------------------------------------------------------------------
# 2.  Group-aware train/test split
# ----------------------------------------------------------------------
gss = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(
    np.zeros(len(labels)), labels.numpy(), groups=subject_id.numpy()))
print("train subjects:", subject_id[train_idx].unique().numel(),
      "| test subjects:", subject_id[test_idx].unique().numel())

class EEGDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

dataset      = EEGDataset(eeg_data, labels)
batch_size   = 64

train_labels = labels[train_idx]

train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size)
test_loader  = DataLoader(Subset(dataset,  test_idx), batch_size=batch_size)

# ----------------------------------------------------------------------
# 3.  Model
# ----------------------------------------------------------------------
example_fif  = glob.glob("../data/preprocessed_data/*_FG_preprocessed-epo.fif")[0]
electrodes   = [ch.upper() for ch in mne.read_epochs(example_fif, preload=False).info["ch_names"]]

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
model = LaBraM(in_channels=len(electrodes), num_classes=2, drop_path=0.1).to(device)

# ----------------------------------------------------------------------
# 4. Load pretrained weights
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# 5.  Optimiser, scheduler, loss
# ----------------------------------------------------------------------
num_epochs = 30
base_lr    = 3e-4
optimizer  = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
scheduler  = CosineLRScheduler(optimizer, t_initial=num_epochs,
                               lr_min=1e-6, warmup_lr_init=1e-5, warmup_t=3)


criterion = nn.CrossEntropyLoss(label_smoothing=0.0) 
# ----------------------------------------------------------------------
# 6.  WandB
# ----------------------------------------------------------------------
wandb.init(
    project="Fagprojekt_eeg",
    name=f"LaBraM_solo-vs-group_{condition.upper()}",
    config=dict(base_lr=base_lr, batch=batch_size, epochs=num_epochs,
                n_subj_train=int(subject_id[train_idx].unique().numel()),
                n_subj_test=int(subject_id[test_idx].unique().numel()))
)
wandb.watch(model, log="gradients", log_freq=100)

# ----------------------------------------------------------------------
# 7.  Training loop
# ----------------------------------------------------------------------
for epoch in range(num_epochs):
    scheduler.step(epoch)
    model.train(); running = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x, electrodes=electrodes), y)
        loss.backward(); optimizer.step()
        running += loss.item()

    # quick eval each epoch
    model.eval(); y_hat, y_true = [], []
    with torch.no_grad():
        for x, y in test_loader:
            y_hat.append(model(x.to(device), electrodes=electrodes).argmax(1).cpu())
            y_true.append(y)
    y_hat = torch.cat(y_hat); y_true = torch.cat(y_true)
    wandb.log({"epoch": epoch+1,
               "train/loss": running / len(train_loader),
               "val/accuracy": accuracy_score(y_true, y_hat),
               "val/f1_macro": f1_score(y_true, y_hat, average="macro"),
               "lr": optimizer.param_groups[0]["lr"]})
    print(f"Epoch {epoch+1}/{num_epochs}  loss={running/len(train_loader):.4f}")

# ----------------------------------------------------------------------
# 8.  Final evaluation & save
# ----------------------------------------------------------------------
torch.save(model.state_dict(), f"../models/LaBraM_solo_vs_group_{condition}.pth")
wandb.save(f"models/LaBraM_solo_vs_group_{condition}.pth")

cm = confusion_matrix(y_true, y_hat)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Solo","Group"], yticklabels=["Solo","Group"])
plt.title("Confusion"); plt.tight_layout()
wandb.log({"confusion_matrix": wandb.Image(plt)}); plt.show()

print(classification_report(y_true, y_hat, target_names=["Solo","Group"]))
wandb.finish()



