import os
import glob
import torch
import mne
from tqdm import tqdm
import wandb

# =============================
# SETTINGS
# =============================

raw_base_dirs = {
    "normal": "reproduce_labram/data/raw/normal/01_tcp_ar",
    "abnormal": "reproduce_labram/data/raw/abnormal/01_tcp_ar"
}
processed_base_dir = "reproduce_labram/data/processed_v3"
os.makedirs(processed_base_dir, exist_ok=True)

# LaBraM target configuration
target_sampling_rate = 200
segment_duration_sec = 10
samples_per_segment = target_sampling_rate * segment_duration_sec

labram_channels = [
    'FP1', 'FP2',
    'F3', 'F4', 'F7', 'F8', 'FZ',
    'C3', 'C4', 'CZ',
    'P3', 'P4', 'PZ',
    'O1', 'O2',
    'T3', 'T4', 'T5', 'T6',
    'A1', 'A2'
]

# =============================
# EDF File Processing Function
# =============================

def process_edf_file(edf_path):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # Normalize channel names
        edf_channels = [
            ch.upper().replace('-REF', '').replace('EEG ', '').strip()
            for ch in raw.ch_names
        ]
        usable_channels = sorted(list(set(labram_channels) & set(edf_channels)))
        if len(usable_channels) < 6:
            print(f"Skipping {edf_path}: too few LaBraM-compatible channels.")
            return []

        # Map to original names
        edf_mapping = {
            ch.upper().replace('-REF', '').replace('EEG ', '').strip(): ch
            for ch in raw.ch_names
        }
        picked_edf_channels = [edf_mapping[ch] for ch in usable_channels]

        # Preprocessing
        raw.pick_channels(picked_edf_channels)
        raw.filter(0.1, 75., fir_design='firwin')
        raw.notch_filter(freqs=50)
        raw.resample(target_sampling_rate)

        # Extract segments
        data = raw.get_data()
        total_samples = data.shape[1]
        segments = []
        for start in range(0, total_samples - samples_per_segment + 1, samples_per_segment):
            segment = data[:, start:start + samples_per_segment]
            segment = (segment - segment.mean(axis=1, keepdims=True)) / (segment.std(axis=1, keepdims=True) + 1e-6)
            tensor = torch.tensor(segment, dtype=torch.float32)
            segments.append(tensor)

        return segments

    except Exception as e:
        print(f"Error processing {edf_path}: {e}")
        return []

# =============================
# Loop Through All Files
# =============================

wandb.init(project="labram-tuab", name="data_processing")

for label in ["normal", "abnormal"]:
    input_dir = raw_base_dirs[label]
    output_dir = os.path.join(processed_base_dir, label)
    os.makedirs(output_dir, exist_ok=True)

    edf_files = glob.glob(os.path.join(input_dir, "**/*.edf"), recursive=True)
    processed_count = 0

    for edf_path in tqdm(edf_files, desc=f"Processing {label}"):
        segments = process_edf_file(edf_path)
        if not segments:
            continue

        base_filename = os.path.splitext(os.path.basename(edf_path))[0]
        for i, segment in enumerate(segments):
            output_path = os.path.join(output_dir, f"{base_filename}_seg{i}.pt")
            torch.save(segment, output_path)
            processed_count += 1

        wandb.log({'processed_count': processed_count})
