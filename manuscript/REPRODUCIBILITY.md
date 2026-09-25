# Reproducibility Statement

This file consolidates, in one place, everything needed to reproduce the
results reported in `manuscript.md`. It duplicates Section 8 of the
manuscript itself so it can be read or checked independently of the full
paper.

## Code and version control

- **Repository**: https://github.com/abinaya-g/ADHD-EEG-Benchmark
- **Branch**: `claude/adhd-eeg-manuscript-revision-boeb24`
- **Commit at time of writing**: `81c1b3b` (see `git log` in the repository for the full history; the commit that added this manuscript directory will postdate this one)
- **Entry point**: `run_all_experiments.py`, orchestrating the modules in `src/` (`data.py`, `models.py`, `evaluation.py`, `coral.py`, `statistics.py`, `reporting.py`, `sanity_checks.py`, `visualization.py`, `explainability.py`)
- **Automated leakage audit**: `src/sanity_checks.py`, run automatically at the end of every experiment and printed as a 14-point pass/fail report; the report corresponding to the final combined run is reproduced in `manuscript/TABLES/leakage_sanity_checks.csv`

## Dataset

- Public EEG dataset introduced by TaghiBeyglou et al. (2022) — 121 subjects (61 ADHD, 60 typically developing controls), 19-channel EEG, recorded during a visual-attention (cartoon-counting) task.
- Hosted on Kaggle; **not redistributed** by this repository or by this manuscript's supplementary files, consistent with the dataset's own release terms.
- Expected local layout and the environment variable used to point the code at a local copy are documented in the repository's `README.md`.

## Cross-validation design (exact values used for every result in this manuscript)

| Parameter | Value |
|---|---|
| Outer folds | 5 |
| Inner folds | 4 |
| Repetitions | 5 |
| Seeds | 42, 43, 44, 45, 46 |
| Max training epochs | 100 |
| Batch size | 16 |
| Early-stopping patience | 8 |
| Learning rate | 1 × 10⁻⁴ |
| Bootstrap resamples (CI) | 2000 |
| CI method | Percentile bootstrap, subject-level resampling |

Source: `manuscript/TABLES/TABLE_1_DATASET_SUMMARY.csv` and `run_config.txt` from the same run, both identical.

## Software environment

- **Frameworks**: TensorFlow/Keras (model definitions and training), scikit-learn (classical classifiers, GroupKFold), SciPy (bootstrap-adjacent utilities, `scipy.signal.resample` for the resolution experiment, `scipy.stats.wilcoxon`), NumPy, pandas.
- **Hardware**: cloud GPU instance (NVIDIA Tesla T4), confirmed from a CUDA device-initialization line in the run's own console output.
- **Exact package versions — NOT YET CONFIRMED.** This is stated as an open item rather than filled in with an assumed value. The repository's `requirements.txt` pins `tensorflow==2.21.0`, but the repository's own `README.md` explicitly instructs against installing this file on Kaggle/Colab and instead using whatever TensorFlow build is preinstalled on the platform's base image — which is exactly the environment used to produce the results in this manuscript. **Before this manuscript is finalized for submission, run the following in the same notebook environment used for the final result-producing run (or a fresh run reproducing it) and record the output here:**

  ```python
  import tensorflow as tf, sklearn, numpy, scipy, sys
  print("Python:", sys.version)
  print("TensorFlow:", tf.__version__)
  print("scikit-learn:", sklearn.__version__)
  print("NumPy:", numpy.__version__)
  print("SciPy:", scipy.__version__)
  ```

## Determinism

- Every model-fitting call is preceded by a call seeding Python's `random`, NumPy, and TensorFlow with a seed derived from the repetition and outer-fold index (`models.set_all_seeds`), and `TF_DETERMINISTIC_OPS=1` / `TF_CUDNN_DETERMINISTIC=1` are set automatically.
- CPU execution is deterministic under these seeds for the layers used in this study.
- GPU execution may show small run-to-run variation in some convolution backward-pass kernels even with the above settings, a documented property of the underlying framework rather than of this codebase specifically.

## How to reproduce the exact tables in this manuscript

1. Obtain the dataset and set `ADHD_EEG_DATA_ROOT` per the repository's `README.md`.
2. Run `python run_all_experiments.py` with no skip flags for a full from-scratch run (computationally substantial — see the repository's own cost estimate in `README.md`), or reuse the phase-checkpoint files already produced (`predictions_nested_cnn.csv`, `predictions_architectures.csv`, `predictions_resolution.csv`) if available, and run with the corresponding `--skip-*` flags to recombine without retraining.
3. Confirm the recombine step reports `Combined 3 prediction file(s) -> 34544 rows` and `Sanity check report: 14/14 passed` before treating any downstream table as final — this exact signature was used in this project to confirm the correct, complete result set was being used (see `MANUSCRIPT_AUDIT.md` for the full account of two earlier, incomplete attempts that did **not** match this signature and were correctly identified and discarded before this manuscript was written).
4. The resulting `tables/TABLE_1_DATASET_SUMMARY.csv` through `TABLE_6_128HZ_VS_128TO512_paired_test.csv`, and `figures/fig3_*.png` through `fig10_*.png`, are the files copied into `manuscript/TABLES/` and `manuscript/FIGURES/` and cited throughout `manuscript.md`.

## Known gaps in this reproducibility record

- Exact package versions (see above) — open item.
- The current state of `ADHD_new.ipynb` (the interactive notebook used to run these experiments) was not re-captured as part of this manuscript's audit trail; the canonical, checkable artifact for reproduction is `run_all_experiments.py` and its output files, not the notebook.
- Ablation-study configurations exist in the codebase (`src/models.py:ABLATION_CONFIGS`) but were not part of the final combined result set this manuscript reports on; reproducing an ablation table is out of scope for reproducing the results in this manuscript as written.
