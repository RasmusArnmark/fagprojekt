# EEG Social Context Classification using LaBraM

This project explores the use of the LaBraM transformer-based model to classify social and sensory conditions in EEG recordings. It is organized around a set of experiments evaluating feedback vs. no-feedback and solo vs. group coordination.

---

## 🔧 Project Structure

```
fagprojekt/
│
├── experiment1/
│   ├── train_exp1.py        # Train on feedback vs. no-feedback
│   ├── evaluate_exp1.py     # Evaluate feedback vs. no-feedback
│   └── data_exp1.py         # Data loading helper for Experiment 1
│
├── train_ready_data_exp2/   # Preprocessed EEG .pt files for Experiment 2
│   └── social_fb/           # Feedback condition
│   └── social_nfb/          # No-feedback condition
│
├── models/                  # Directory to store trained model weights
│   ├── labram-base.pth      # Optional pretrained LaBraM weights
│   ├── EEG_model_FB_vs_NoFB.pth
│   ├── LaBraM_solo_vs_group_fb.pth
│   └── LaBraM_solo_vs_group_nfb.pth
│
├── evaluate_exp2.py         # Evaluation script for Experiment 2
└── README.md
```

---

##  Running the Project

### 1. Setup
Ensure you have a Python environment with the following key dependencies:

- `torch`, `torcheeg`, `mne`, `scikit-learn`, `wandb`, `matplotlib`, `seaborn`, `timm`

We recommend using `conda`:

```bash
conda create -n labram python=3.10
conda activate labram
pip install -r requirements.txt
```

---

### 2. Training

#### Experiment 1: Feedback vs No Feedback

```bash
python experiment1/train_exp1.py
```

This will train the model and save it to `models/EEG_model_FB_vs_NoFB.pth`.

#### Experiment 2: Solo vs Group

Edit the condition inside the training script:

```python
# Inside train_exp2.py (currently inline in main script)
condition = "nfb"  # or "fb"
```

Then run:

```bash
python train_exp2.py
```

---

### 3. Evaluation

Evaluate a saved model on its held-out test set:

#### Feedback vs No Feedback:

```bash
python experiment1/evaluate_exp1.py
```

#### Solo vs Group (change `condition` in script):

```bash
python evaluate_exp2.py
```

Make sure the corresponding model is saved at:

```
models/LaBraM_solo_vs_group_fb.pth
models/LaBraM_solo_vs_group_nfb.pth
```

---

## 📂 Data Locations

- Experiment 1 expects data to be loaded from `.pt` files using the `data_exp1.py` loader.
- Experiment 2 expects data in:
  ```
  train_ready_data_exp2/social_fb/
  train_ready_data_exp2/social_nfb/
  ```
  Each folder must contain EEG tensors (`.pt`), label files (`*_labels.pt`), and subject ID files (`*_subjects.pt`).

---

## 💾 Models

Models are saved to the `/models/` folder automatically after training. Evaluation scripts will also load from this folder.

---

## 📝 Notes

- Subject-level splits are used for all experiments (GroupShuffleSplit).
- Evaluation is based on accuracy, F1-score, and confusion matrix.
- All scripts support MPS, CUDA, or CPU depending on what's available.

