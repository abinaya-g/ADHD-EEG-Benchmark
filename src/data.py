"""
Data loading, epoching, normalization, and the 128->512 Hz sensitivity
resampling.

Loading logic (folder layout, epoch cutting, per-epoch z-score
normalization) is reused verbatim from the audited original notebooks
(adhd-baseline.ipynb cells 6/9, adhd-coral.ipynb cell 0) -- see
AUDIT_REPORT.md. It is not rewritten; only wrapped into importable
functions with saved provenance.
"""
import glob
import os

import numpy as np

from . import config


def load_dataset(data_root=None, folders=None, fs_assumed=None, epoch_seconds=None):
    """
    Reproduces the exact loading/epoching logic of the original notebooks.

    Returns
    -------
    X : (n_epochs, n_channels, n_timesamples, 1) float array
    y : (n_epochs,) int array, 1=ADHD, 0=Control
    groups : (n_epochs,) str array of subject IDs (one per source .mat file)
    file_epoch_index : (n_epochs,) int array, index of the epoch within its
        source file (0-based) -- needed later to prove no duplicate/leaked
        epochs and to reconstruct subject-level groupings from raw predictions.
    """
    from scipy.io import loadmat

    data_root = data_root or config.DATA_ROOT
    folders = folders or config.DATA_FOLDERS
    fs_assumed = fs_assumed or config.FS_ASSUMED_HZ
    epoch_seconds = epoch_seconds or config.EPOCH_SECONDS
    epoch_len = fs_assumed * epoch_seconds

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"DATA_ROOT '{data_root}' does not exist in this environment. "
            "The released dataset is hosted on Kaggle "
            "(abinayajone/adhd-eeg-dataset) and is not bundled with this "
            "repository. Set ADHD_EEG_DATA_ROOT to your local copy (e.g. "
            "the Kaggle input mount) before calling load_dataset()."
        )

    all_epochs, all_labels, all_subject_ids, all_epoch_idx = [], [], [], []
    for folder_name, label in folders.items():
        folder_path = f"{data_root}/{folder_name}/{folder_name}"
        files = sorted(glob.glob(folder_path + "/*.mat"))
        for f in files:
            mat = loadmat(f)
            key = [k for k in mat.keys() if not k.startswith("__")][0]
            arr = mat[key]  # (n_samples, n_channels)
            n_epochs = arr.shape[0] // epoch_len
            subject_id = os.path.basename(f).replace(".mat", "")
            for i in range(n_epochs):
                segment = arr[i * epoch_len:(i + 1) * epoch_len, :]
                all_epochs.append(segment.T)  # -> (n_channels, n_timesamples)
                all_labels.append(label)
                all_subject_ids.append(subject_id)
                all_epoch_idx.append(i)

    X = np.array(all_epochs)[..., np.newaxis]
    y = np.array(all_labels)
    groups = np.array(all_subject_ids)
    file_epoch_index = np.array(all_epoch_idx)
    return X, y, groups, file_epoch_index


def normalize_epoch(epoch):
    """Per-epoch, per-channel z-score. Uses only that epoch's own samples,
    so it is leakage-free by construction regardless of how folds are later
    drawn (see AUDIT_REPORT.md, item 2)."""
    mean = epoch.mean(axis=1, keepdims=True)
    std = epoch.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)
    return (epoch - mean) / std


def normalize_all(X):
    return np.array([normalize_epoch(e) for e in X])


def resample_pipeline_b(X, orig_fs=None, target_fs=None):
    """
    128 Hz -> 512 Hz sensitivity pipeline (Phase 3 / Table 7).

    IMPORTANT: this is polyphase FFT-based interpolation
    (scipy.signal.resample) of the *same* 128 Hz samples the released
    dataset provides. It manufactures no new information and is NOT a
    reconstruction of TaghiBeyglou et al.'s original 512 Hz acquisition.
    Any experiment using this function's output must be reported as
    "128->512 Hz interpolation", never as "512 Hz original pipeline" or
    "reproduction of the original acquisition". See AUDIT_REPORT.md Phase 3
    section for the full justification of why exact reproduction is not
    possible from the released files.

    Parameters
    ----------
    X : (n_epochs, n_channels, n_timesamples, 1) array at orig_fs
    Returns
    -------
    X_resampled : (n_epochs, n_channels, n_timesamples * target_fs/orig_fs, 1)
    """
    from scipy.signal import resample

    orig_fs = orig_fs or config.FS_ASSUMED_HZ
    target_fs = target_fs or config.FS_INTERP_HZ
    n_epochs, n_channels, n_samples, _ = X.shape
    target_samples = int(round(n_samples * target_fs / orig_fs))

    out = np.empty((n_epochs, n_channels, target_samples, 1), dtype=X.dtype)
    for e in range(n_epochs):
        out[e, :, :, 0] = resample(X[e, :, :, 0], target_samples, axis=1)
    return out


# ---------------------------------------------------------------------------
# Synthetic surrogate data, used ONLY to validate the pipeline's mechanics
# (leakage checks, shapes, aggregation logic) when the real dataset is not
# reachable in the current environment. Never treat output derived from
# this function as a real result.
# ---------------------------------------------------------------------------
def make_synthetic_dataset(n_adhd_subjects=20, n_control_subjects=20,
                            min_epochs=3, max_epochs=6,
                            n_channels=None, n_timesamples=None,
                            class_signal=0.6, seed=0):
    """
    Generates a synthetic dataset with the SAME array shapes and subject/
    epoch structure as the real one (variable epochs per subject, subject
    IDs as groups), but with random Gaussian data plus a small class-
    dependent mean shift so that a leakage-free classifier scores well
    above chance and a leaked one scores implausibly higher. This is a
    software self-test fixture, not EEG data, and must be labeled as such
    in any output it produces.
    """
    rng = np.random.default_rng(seed)
    n_channels = n_channels or config.N_CHANNELS
    n_timesamples = n_timesamples or config.EPOCH_LEN_A

    X_list, y_list, g_list, idx_list = [], [], [], []
    subj_counter = 0
    for label, n_subj in ((1, n_adhd_subjects), (0, n_control_subjects)):
        for _ in range(n_subj):
            subj_id = f"SYN{subj_counter:04d}"
            subj_counter += 1
            n_ep = rng.integers(min_epochs, max_epochs + 1)
            base_shift = rng.normal(0, 0.3, size=(n_channels, 1))
            for ep_i in range(n_ep):
                signal = rng.normal(0, 1, size=(n_channels, n_timesamples))
                signal += base_shift
                signal += class_signal * label
                X_list.append(signal)
                y_list.append(label)
                g_list.append(subj_id)
                idx_list.append(ep_i)

    X = np.array(X_list)[..., np.newaxis].astype(np.float32)
    y = np.array(y_list)
    groups = np.array(g_list)
    file_epoch_index = np.array(idx_list)
    return X, y, groups, file_epoch_index
