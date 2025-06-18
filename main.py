import os, glob, re
import numpy as np
import pandas as pd
import torch
import mne
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# ---------------- CONFIG ----------------
RAW_DIR      = "data/PreprocessedData"
OVERVIEW_PKL = "data/FG_overview_df_v2.pkl"
OUT_DIR      = "processed_groupsolo"
CHUNK_SIZE   = 4
RESAMPLE_HZ  = 200
NUM_PATCHES  = 6
# ----------------------------------------

# Event code definitions
event_labels = {
    'T1P': 301, 'T1Pn': 302,     # solo
    'T3P': 303, 'T3Pn': 304,     # triad
    'T12P': 305, 'T12Pn': 306,   # dyads
    'T13P': 307, 'T13Pn': 308,
    'T23P': 309, 'T23Pn': 310
}

solo_events  = {301, 302}
group_events = {303, 304, 305, 306, 307, 308, 309, 310}

event_to_label = {}
for name, code in event_labels.items():
    if code in solo_events:
        event_to_label[code] = 0
    elif code in group_events:
        event_to_label[code] = 1
    else:
        raise ValueError(f"Unmapped event: {code}")

def parse_exp_code(fname: str) -> str:
    m = re.match(r"(\d{3}[A-Z])_FG", os.path.basename(fname))
    if not m:
        raise ValueError(f"Cannot parse subject code from {fname}")
    return m.group(1)

def main():
    overview = pd.read_pickle(OVERVIEW_PKL)
    fif_files = sorted(glob.glob(os.path.join(RAW_DIR, "*_FG_preprocessed-epo.fif")))
    print("Found", len(fif_files), "FIF files")

    eeg_dir = os.path.join(OUT_DIR, "eeg_chunk")
    lab_dir = os.path.join(OUT_DIR, "labels_chunk")
    for d in (OUT_DIR, eeg_dir, lab_dir):
        os.makedirs(d, exist_ok=True)

    label_enc = LabelEncoder().fit(overview["Subject_id"].astype(str).unique())

    for b0 in range(0, len(fif_files), CHUNK_SIZE):
        eeg_list, lab_list, sid_list = [], [], []
        batch_files = fif_files[b0:b0 + CHUNK_SIZE]

        for fpath in batch_files:
            exp_id = parse_exp_code(fpath)
            row = overview.loc[overview.Exp_id == exp_id].iloc[0]
            subj = str(row.Subject_id)

            epochs = mne.read_epochs(fpath, preload=True, verbose=False)
            epochs.resample(RESAMPLE_HZ)

            ev_ids = epochs.events[:, -1]
            keep_idx = np.isin(ev_ids, list(event_to_label.keys()))
            if not keep_idx.any():
                continue

            kept_ids = ev_ids[keep_idx]
            data = epochs.get_data()[keep_idx]
            data = (data - data.mean(axis=-1, keepdims=True)) / (data.std(axis=-1, keepdims=True) + 1e-8)

            for x, e in zip(data, kept_ids):
                if e not in event_to_label:
                    raise ValueError(f"Unknown event ID: {e}")
                eeg_list.append(x)
                lab_list.append(event_to_label[e])
                sid_list.append(subj)

            print(f"▶ {os.path.basename(fpath)}  → {len(kept_ids)} kept events")

        if not eeg_list:
            print("⏭️ Skipping empty chunk", b0 // CHUNK_SIZE)
            continue

        eeg_tensor  = torch.tensor(np.stack(eeg_list), dtype=torch.float32)
        lab_tensor  = torch.tensor(lab_list, dtype=torch.long)
        sid_tensor  = torch.tensor(label_enc.transform(np.array(sid_list)), dtype=torch.int32)

        # Reshape to patches
        t_per_patch = eeg_tensor.shape[2] // NUM_PATCHES
        eeg_tensor = eeg_tensor.reshape(eeg_tensor.shape[0], eeg_tensor.shape[1], NUM_PATCHES, t_per_patch)

        idx = b0 // CHUNK_SIZE
        torch.save(eeg_tensor, os.path.join(eeg_dir, f"eeg_chunk_{idx}.pt"))
        torch.save(lab_tensor, os.path.join(lab_dir, f"labels_chunk_{idx}.pt"))
        torch.save(sid_tensor, os.path.join(lab_dir, f"subjects_chunk_{idx}.pt"))

        print(f"💾 Saved chunk {idx} → EEG: {tuple(eeg_tensor.shape)}, Labels: {Counter(lab_list)}")

if __name__ == "__main__":
    main()