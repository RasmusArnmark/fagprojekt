import mne, os, re 
import glob
data_folder = "data"
eeg_files = glob.glob(os.path.join(data_folder, "*_FG_preprocessed-epo.fif"))
print(f"Found {len(eeg_files)} EEG files.")
epochs = mne.read_epochs(eeg_files[0], preload=False)

# 1) Whole header --------------------------------------------------
print("\n--- INFO KEYS ---")
print(epochs.info.keys())

# 2) Subject sub-dict ---------------------------------------------
subj = epochs.info.get('subject_info', None)
if subj:
    print("\nSubject info found:")
    print(subj)
else:
    print("\nNo subject_info in header.")

# 3) Metadata ------------------------------------------------------
if epochs.metadata is not None:
    print("\nMetadata columns:", epochs.metadata.columns.tolist())
    print(epochs.metadata.head())
else:
    print("\nNo metadata DataFrame.")

# 4) Filename fallback --------------------------------------------
fname = os.path.basename(eeg_files[0])
match = re.search(r'P\d{2}', fname)
if match:
    print("\nSubject inferred from filename:", match.group(0))
