import os, glob
import numpy as np
import torch, mne
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torcheeg.models import LaBraM
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from data_exp2 import load_dataset

# Load tensors
eeg_data, labels_data, subj_ids = load_dataset()

print("EEG :",     eeg_data.shape)
print("labels :",  labels_data.shape)
print("subjects:", subj_ids.shape, "unique =", subj_ids.unique().numel())

# Subject-level split (same seed = same split)
gss = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(
    np.zeros(len(labels_data)),
    labels_data.cpu().numpy(),         # <- fix here
    groups=subj_ids.cpu().numpy()      # <- and here
))

class EEGDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

dataset     = EEGDataset(eeg_data, labels_data)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=64)

# Load model
example_fif = glob.glob("../data/preprocessed_data/*_FG_preprocessed-epo.fif")[0]
electrode_names = [ch.upper() for ch in mne.read_epochs(example_fif, preload=False).info["ch_names"]]

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = LaBraM(in_channels=len(electrode_names), num_classes=2, drop_path=0.1).to(device)

# Load trained weights
model_path = os.path.join("../models", "EEG_model_FB_vs_NoFB.pth")
assert os.path.exists(model_path), f"Model file not found: {model_path}"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Evaluation
all_pred, all_true = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x, electrodes=electrode_names).argmax(1).cpu()
        all_pred.extend(preds)
        all_true.extend(y.cpu())  # <- fix here

acc = accuracy_score(all_true, all_pred)
f1  = f1_score(all_true, all_pred)
cm  = confusion_matrix(all_true, all_pred)

print("Accuracy :", acc)
print("F1-score :", f1)
print(classification_report(all_true, all_pred, target_names=["No FB", "FB"]))

# Confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No FB", "FB"], yticklabels=["No FB", "FB"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()