# data.py
# ------------------------------------------------------------
#  Pre-process FG triad EEG → PyTorch tensors with subject IDs
# ------------------------------------------------------------
import os, glob, re
import numpy as np
import pandas as pd
import torch
import mne
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List

# ---------------- USER-CONFIGURABLE PATHS --------------------
RAW_DIR      = "data/PreprocessedData"          # *.fif files
OVERVIEW_PKL = "data/FG_overview_df_v2.pkl"
FORCE_PKL    = "data/forcedf.pkl"                  # loaded but not yet used
OUT_DIR      = "processed"                         # will hold eeg_chunk/ labels_chunk/
CHUNK_SIZE   = 4                                   # files per saved chunk
RESAMPLE_HZ  = 200
NUM_PATCHES  = 6                                   # for LaBraM window reshape
# -------------------------------------------------------------

SOLO_CODES   = {301, 302}                 # T1P, T1Pn     – always keep
TRIAD_CODES  = {303, 304}                 # T3P, T3Pn     – all three keep
DYAD_CODES   = {                          # per EEG_device
    1: {305, 306, 307, 308},              # T12*, T13*
    2: {305, 306, 309, 310},              # T12*, T23*
    3: {307, 308, 309, 310},              # T13*, T23*
}          # T3P, T3Pn

# event IDs grouped by *who is involved*

# Solo / dyad codes that carry *feedback* (odd numbers) vs *no-feedback* (even)
IS_FEEDBACK = lambda ev_id: ev_id % 2 == 1

def keep_events(device_id: int) -> set[int]:
    """Return the exact event-IDs that involve the current headset."""
    return SOLO_CODES | TRIAD_CODES | DYAD_CODES[device_id]

# -------------------------------------------------------------------------
def _parse_subject_code(fname: str) -> str:
    """
    '326A_FG_preprocessed-epo.fif' -> '326A'
    Raise ValueError if the pattern does not match.
    """
    m = re.match(r"(\d{3}[A-Z])_FG", os.path.basename(fname))
    if not m:
        raise ValueError(f"Cannot parse subject code from {fname}")
    return m.group(1)

def row_from_exp_id(exp_id: str, overview: pd.DataFrame) -> pd.Series:
    """Return the single row whose Exp_id matches the file code (e.g. '301A')."""
    row = overview.loc[overview["Exp_id"] == exp_id]
    if row.empty:
        raise KeyError(f"{exp_id} not found in overview table.")
    return row.iloc[0] 

# -------------------------------------------------------------------------
def preprocess_and_save():
    overview = pd.read_pickle(OVERVIEW_PKL)
    _ = pd.read_pickle(FORCE_PKL)

    fif_files = sorted(glob.glob(os.path.join(RAW_DIR, "*_FG_preprocessed-epo.fif")))
    print("Found", len(fif_files), "FIF files")

    eeg_out = os.path.join(OUT_DIR, "eeg_chunk")
    lab_out = os.path.join(OUT_DIR, "labels_chunk")
    for d in (OUT_DIR, eeg_out, lab_out):
        os.makedirs(d, exist_ok=True)

    # ▼ 1.  Fit once on *all* participant IDs so the mapping is stable
    label_enc = LabelEncoder()
    label_enc.fit(overview["Subject_id"].astype(str).unique())

    for batch_start in range(0, len(fif_files), CHUNK_SIZE):
        batch_files  = fif_files[batch_start: batch_start + CHUNK_SIZE]
        batch_eeg, batch_lbl, batch_sid = [], [], []

        for fpath in batch_files:

            
            exp_code = _parse_subject_code(fpath)
            row      = row_from_exp_id(exp_code, overview)

            subject_id = str(row.Subject_id)            # '1049'
            device_id  = int(row.EEG_device)            # 1 / 2 / 3

            epochs = mne.read_epochs(fpath, preload=True, verbose="ERROR")
            epochs.resample(RESAMPLE_HZ)

            valid_set = keep_events(device_id)
            keep_idx  = np.isin(epochs.events[:, -1], list(valid_set))
            if not np.any(keep_idx):
                continue

            if batch_start == 0 and fpath == batch_files[0]:      # only first file
                unique, counts = np.unique(epochs.events[keep_idx, -1], return_counts=True)
                print("Kept events + counts for", exp_code, ":", dict(zip(unique, counts)))

            data   = epochs.get_data()[keep_idx]
            labels = (epochs.events[keep_idx, -1] % 2 == 1).astype(int)

            data = (data - data.mean(axis=-1, keepdims=True)) / \
                   (data.std(axis=-1, keepdims=True) + 1e-8)

            batch_eeg.append(data)
            batch_lbl.append(labels)
            batch_sid.append(np.full(len(labels), subject_id))

            print(f"▶ {os.path.basename(fpath)}  device={device_id}", end=" ")

            valid_set = keep_events(device_id)
            keep_idx  = np.isin(epochs.events[:, -1], list(valid_set))
            print(f"kept={keep_idx.sum()}") 

        if not batch_eeg:
            print("Batch", batch_start // CHUNK_SIZE, "empty → skipped")
            continue

        eeg_tensor = torch.tensor(np.concatenate(batch_eeg, 0), dtype=torch.float32)
        lbl_tensor = torch.tensor(np.concatenate(batch_lbl),   dtype=torch.long)

        # ▼ 2.  Only transform (no fit) because encoder is fixed
        sid_int    = label_enc.transform(np.concatenate(batch_sid))
        sid_tensor = torch.tensor(sid_int, dtype=torch.int32)

        t_per_patch = eeg_tensor.shape[2] // NUM_PATCHES
        eeg_tensor  = eeg_tensor.reshape(eeg_tensor.shape[0],
                                         eeg_tensor.shape[1],
                                         NUM_PATCHES,
                                         t_per_patch)

        idx = batch_start // CHUNK_SIZE
        torch.save(eeg_tensor, os.path.join(eeg_out, f"eeg_chunk_{idx}.pt"))
        torch.save(lbl_tensor, os.path.join(lab_out, f"labels_chunk_{idx}.pt"))
        torch.save(sid_tensor, os.path.join(lab_out, f"subjects_chunk_{idx}.pt"))
        print(f"Saved chunk {idx}: EEG {tuple(eeg_tensor.shape)} | {len(lbl_tensor)} labels")

# -------------------------------------------------------------------------
def load_dataset() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Utility for main.py – loads all saved chunks into three big tensors."""
    eeg_files  = sorted(glob.glob(os.path.join(OUT_DIR, "eeg_chunk",   "eeg_chunk_*.pt")))
    lbl_files  = sorted(glob.glob(os.path.join(OUT_DIR, "labels_chunk","labels_chunk_*.pt")))
    sid_files  = sorted(glob.glob(os.path.join(OUT_DIR, "labels_chunk","subjects_chunk_*.pt")))

    eeg  = torch.cat([torch.load(f) for f in eeg_files], dim=0)
    lbl  = torch.cat([torch.load(f) for f in lbl_files], dim=0)
    sid  = torch.cat([torch.load(f) for f in sid_files], dim=0)
    assert eeg.shape[0] == lbl.shape[0] == sid.shape[0]
    return eeg, lbl, sid
# -------------------------------------------------------------------------

if __name__ == "__main__":
    preprocess_and_save()
