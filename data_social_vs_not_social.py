import os, glob, re
import numpy as np
import pandas as pd
import mne, torch
from sklearn.preprocessing import LabelEncoder
from typing import List

RAW_DIR      = "data/PreprocessedData"
OVERVIEW_PKL = "data/FG_overview_df_v2.pkl"
OUT_DIR      = "processed2"
CHUNK_SIZE   = 4
RESAMPLE_HZ  = 200
NUM_PATCHES  = 6

SOLO_CODES   = {301, 302}                 # T1P, T1Pn  – always solo
TRIAD_CODES  = {303, 304}                 # T3P, T3Pn  – social for everyone
DYAD_CODES   = {                          # device-specific social codes
    1: {305, 306, 307, 308},              # T12*, T13*
    2: {305, 306, 309, 310},              # T12*, T23*
    3: {307, 308, 309, 310},              # T13*, T23*
}

def keep_set(device: int) -> set[int]:
    return SOLO_CODES | TRIAD_CODES | DYAD_CODES[device]

def parse_exp_code(fname: str) -> str:
    m = re.match(r"(\d{3}[A-Z])_FG", os.path.basename(fname))
    if not m:
        raise ValueError(f"cannot parse Exp_id from {fname}")
    return m.group(1)

def main():
    overview = pd.read_pickle(OVERVIEW_PKL)
    label_enc = LabelEncoder().fit(overview["Subject_id"].astype(str))
    os.makedirs(os.path.join(OUT_DIR, "social_fb"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "social_nfb"), exist_ok=True)

    fif_files = sorted(glob.glob(os.path.join(RAW_DIR, "*_FG_preprocessed-epo.fif")))
    for b0 in range(0, len(fif_files), CHUNK_SIZE):
        fb_eeg, fb_lab, fb_sid = [], [], []
        nf_eeg, nf_lab, nf_sid = [], [], []
        for fpath in fif_files[b0:b0 + CHUNK_SIZE]:
            exp_id = parse_exp_code(fpath)
            row = overview.loc[overview.Exp_id == exp_id].iloc[0]
            device = int(row.EEG_device)
            subj   = str(row.Subject_id)

            epochs = mne.read_epochs(fpath, preload=True, verbose=False)
            epochs.resample(RESAMPLE_HZ)
            ev_ids = epochs.events[:, -1]
            mask   = np.isin(ev_ids, list(keep_set(device)))
            if not mask.any():
                continue

            data = epochs.get_data()[mask]                       # (N, C, T)
            data = (data - data.mean(axis=-1, keepdims=True)) / \
                   (data.std(axis=-1, keepdims=True) + 1e-8)
            ev_kept = ev_ids[mask]

            for d, e in zip(data, ev_kept):
                label = 0 if e in SOLO_CODES else 1              # 0 = solo, 1 = social
                if e % 2 == 1:                                   # feedback
                    fb_eeg.append(d);  fb_lab.append(label);  fb_sid.append(subj)
                else:                                            # no-feedback
                    nf_eeg.append(d);  nf_lab.append(label);  nf_sid.append(subj)

        # ---------- save one chunk ----------
        if fb_eeg:
            save_chunk(fb_eeg, fb_lab, fb_sid, "social_fb",  b0//CHUNK_SIZE, label_enc)
        if nf_eeg:
            save_chunk(nf_eeg, nf_lab, nf_sid, "social_nfb", b0//CHUNK_SIZE, label_enc)

def save_chunk(eeg_list, lab_list, sid_list, subfolder, idx, enc):
    eeg  = torch.tensor(np.stack(eeg_list), dtype=torch.float32)
    labs = torch.tensor(lab_list,          dtype=torch.long)
    sids = torch.tensor(enc.transform(np.array(sid_list)), dtype=torch.int32)

    t_per_patch = eeg.shape[2] // NUM_PATCHES
    eeg = eeg.reshape(eeg.shape[0], eeg.shape[1], NUM_PATCHES, t_per_patch)

    base = os.path.join(OUT_DIR, subfolder, f"{subfolder}_{idx}")
    torch.save(eeg,  f"{base}.pt")
    torch.save(labs, f"{base}_labels.pt")
    torch.save(sids, f"{base}_subjects.pt")
    print(f"Saved {subfolder} chunk {idx}: EEG {tuple(eeg.shape)} | {len(labs)} labels")

if __name__ == "__main__":
    main()