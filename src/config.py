"""
Central configuration for the ADHD-EEG robust benchmark.

Every experiment script imports from here so that seeds, fold counts,
and training hyperparameters are defined in exactly one place.
"""
import os

# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------
# The released dataset lives on Kaggle at
# "abinayajone/adhd-eeg-dataset" and is NOT bundled with this repository
# (it is not redistributable from here, and this repo does not claim a
# download location beyond what the original notebooks used). Point this at
# your local copy, e.g. the Kaggle input mount, before running real
# experiments.
DATA_ROOT = os.environ.get("ADHD_EEG_DATA_ROOT", "/kaggle/input/datasets/abinayajone/adhd-eeg-dataset")

DATA_FOLDERS = {
    "ADHD_part1": 1,
    "ADHD_part2": 1,
    "Control_part1": 0,
    "Control_part2": 0,
}

# ---------------------------------------------------------------------------
# Preprocessing (Pipeline A = the notebooks' current, unverified assumption)
# ---------------------------------------------------------------------------
FS_ASSUMED_HZ = 128
EPOCH_SECONDS = 30
EPOCH_LEN_A = FS_ASSUMED_HZ * EPOCH_SECONDS  # 3840 samples
N_CHANNELS = 19

# Pipeline B = 128 -> 512 Hz sensitivity experiment. This is NOT a
# reconstruction of the original 512 Hz acquisition; it is polyphase/FFT
# interpolation of the same 128 Hz samples, clearly labeled as such
# everywhere it is used (see src/data.py: resample_pipeline_b).
FS_INTERP_HZ = 512
EPOCH_LEN_B = FS_INTERP_HZ * EPOCH_SECONDS  # 15360 samples

# ---------------------------------------------------------------------------
# Cross-validation design
# ---------------------------------------------------------------------------
OUTER_FOLDS = 5
INNER_FOLDS = 4
REPEATS = 5
RANDOM_SEEDS = [42, 43, 44, 45, 46]  # one per repeat, len must equal REPEATS

assert len(RANDOM_SEEDS) == REPEATS, "RANDOM_SEEDS must have one entry per repeat"

# ---------------------------------------------------------------------------
# CNN training protocol (kept identical to the original notebooks except
# for where validation_data points -- see evaluation.py)
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 8
LEARNING_RATE = 1e-4

# ---------------------------------------------------------------------------
# Bootstrap / CI
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 2000
CI_ALPHA = 0.05  # 95% CI

# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
RESULTS_DIR = os.environ.get("ADHD_EEG_RESULTS_DIR", "results")
FIGURES_DIR = os.environ.get("ADHD_EEG_FIGURES_DIR", "figures")
TABLES_DIR = os.environ.get("ADHD_EEG_TABLES_DIR", "tables")

for _d in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
    os.makedirs(_d, exist_ok=True)
