import torch
import numpy as np
import glob
import os
import mne
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torcheeg.models import LaBraM
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]


def load_data(condition: str):
    eeg_files = sorted(glob.glob(f"processed2/social_{condition}_chunk_*.pt"))
    label_files = sorted(glob.glob(f"processed2/social_{condition}_labels_*.pt"))

    eeg_data = torch.cat([torch.load(f) for f in eeg_files], dim=0)
    labels = torch.cat([torch.load(f) for f in label_files], dim=0)
    return eeg_data, labels


def load_electrode_names():
    fif_file = glob.glob("data/*_FG_preprocessed-epo.fif")[0]
    epochs = mne.read_epochs(fif_file, preload=False)
    return [ch.upper() for ch in epochs.info['ch_names']]


def train_model(eeg_data, labels, electrode_names, model_path, n_epochs=10):
    dataset = EEGDataset(eeg_data, labels)
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=labels, random_state=42)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LaBraM(in_channels=len(electrode_names), num_classes=2).to(device)

    if os.path.exists("models/labram-base.pth"):
        model.load_state_dict(torch.load("models/labram-base.pth"))
        print("Loaded pretrained model.")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {len(train_idx)} samples, testing on {len(test_idx)}")

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X, electrodes=electrode_names)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

    # Evaluation
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X, electrodes=electrode_names)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(y.cpu().numpy())

    cm = confusion_matrix(all_trues, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Solo', 'Social'], yticklabels=['Solo', 'Social'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    print("Classification Report:")
    print(classification_report(all_trues, all_preds, target_names=["Solo", "Social"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["feedback", "nofeedback"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    eeg_data, labels = load_data(args.condition)
    electrodes = load_electrode_names()
    model_path = f"models/social_{args.condition}_model.pth"
    train_model(eeg_data, labels, electrodes, model_path, n_epochs=args.epochs)
