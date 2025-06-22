import os, glob
import numpy as np
import torch, mne
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torcheeg.models import LaBraM
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Choose condition: "fb" or "nfb"
condition   = "fb"
data_folder = f"train_ready_data_exp3/social_{condition}"

eeg_files  = sorted([f for f in glob.glob(os.path.join(data_folder, f"social_{condition}_*.pt")) if not f.endswith(("_labels.pt", "_subjects.pt"))])
lab_files  = sorted([f for f in glob.glob(os.path.join(data_folder, f"social_{condition}_*.pt")) if f.endswith("_labels.pt")])
sid_files  = sorted([f for f in glob.glob(os.path.join(data_folder, f"social_{condition}_*.pt")) if f.endswith("_subjects.pt")])

eeg_data   = torch.cat([torch.load(f) for f in eeg_files], dim=0)
labels     = torch.cat([torch.load(f) for f in lab_files], dim=0)
subject_id = torch.cat([torch.load(f) for f in sid_files], dim=0)

gss = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(
    np.zeros(len(labels)), labels.cpu().numpy(), groups=subject_id.cpu().numpy()))

class EEGDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

dataset     = EEGDataset(eeg_data, labels)
test_loader = DataLoader(Subset(dataset, test_idx), batch_size=64)

# Model setup
example_fif = glob.glob("../data/preprocessed_data/*_FG_preprocessed-epo.fif")[0]
electrodes  = [ch.upper() for ch in mne.read_epochs(example_fif, preload=False).info["ch_names"]]

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = LaBraM(in_channels=len(electrodes), num_classes=2, drop_path=0.1).to(device)

# Load trained model weights
model_path = f"../models/LaBraM_solo_vs_group_{condition}.pth"
assert os.path.exists(model_path), f"Model not found: {model_path}"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Evaluation
all_pred, all_true = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x, electrodes=electrodes).argmax(1).cpu()
        all_pred.extend(preds)
        all_true.extend(y.cpu())

acc = accuracy_score(all_true, all_pred)
f1  = f1_score(all_true, all_pred)
cm  = confusion_matrix(all_true, all_pred)

print("Accuracy :", acc)
print("F1-score :", f1)
print(classification_report(all_true, all_pred, target_names=["Solo", "Group"]))

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Solo", "Group"], yticklabels=["Solo", "Group"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()