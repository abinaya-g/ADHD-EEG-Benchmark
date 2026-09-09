# ADHD-EEG Robust Benchmark

A leakage-corrected, subject-independent evaluation framework built around
the CNN/hybrid-classifier pipeline from the manuscript *"Assessing the
Robustness of CNN and Hybrid Classifiers for Subject-Independent ADHD
Detection from Raw EEG"* (desk-rejected from *Computer Methods and Programs
in Biomedicine Update*; see `original_notebooks/manuscript_desk_rejected.pdf`).

**Read `AUDIT_REPORT.md` first.** It documents the leakage found in the
original notebooks (outer test fold used as CNN early-stopping validation
data) and exactly how this codebase fixes it. Read `FINAL_REPORT.md` for an
honest accounting of what was and was not executed with real data.

## What this repository is (and isn't)

This codebase was developed in a sandboxed environment with **no access to
the actual dataset** (`abinayajone/adhd-eeg-dataset`, hosted on Kaggle — not
bundled here, never claimed to be downloadable from this repo) and **no
GPU**. Every module was validated against a synthetic surrogate dataset with
the same array shapes and subject/epoch structure as the real one
(`src/data.py:make_synthetic_dataset`), run end-to-end on CPU
(`tests/test_synthetic_pipeline.py`), to prove the pipeline's *mechanics*
are correct before it ever touches real data. **No number produced by the
synthetic self-test is a real result** — every such output is prefixed
`SYNTHETIC_` and must never be copied into a manuscript table or figure.

To get real, publication-usable numbers, run `run_all_experiments.py`
(without `--smoke-test`) in an environment with the actual dataset and,
strongly recommended, a GPU — e.g. the same Kaggle environment the original
notebooks used.

## Dataset

- Source: Kaggle dataset `abinayajone/adhd-eeg-dataset` (as used by the
  original notebooks). This repository does not redistribute it and makes
  no claim about any other download location.
- Expected structure once downloaded, pointed to by `ADHD_EEG_DATA_ROOT`:
  ```
  <ADHD_EEG_DATA_ROOT>/
    ADHD_part1/ADHD_part1/*.mat
    ADHD_part2/ADHD_part2/*.mat
    Control_part1/Control_part1/*.mat
    Control_part2/Control_part2/*.mat
    Standard-10-20-Cap19new/Standard-10-20-Cap19new.ced
  ```
- 121 subjects (61 ADHD / 60 Control), 508 total 30-second epochs (288
  ADHD / 220 Control), 19 EEG channels. Sampling rate is **assumed** 128 Hz
  (no sampling-rate metadata is present in the released files — see
  `AUDIT_REPORT.md` §5). Treat this as an assumption, not a verified fact,
  in any text you write about it.

## Setup

**On Kaggle/Colab: do not run `pip install -r requirements.txt`.** Those
platforms already ship numpy/scipy/pandas/scikit-learn/matplotlib/tensorflow
preinstalled and mutually compatible. Installing a different
`tensorflow`/`tensorflow-cpu` wheel on top of the existing one partially
overwrites its compiled `.so` files and produces `ImportError: undefined
symbol ...` (typically inside `tensorflow.lite`, even though nothing here
uses TFLite) the moment anything imports `tensorflow`. If you hit that:
restart the kernel (this resets `dist-packages` back to the clean base
image) and just run the scripts as-is against Kaggle's own TensorFlow —
skip installing requirements.txt entirely. Only set the data path:

```bash
export ADHD_EEG_DATA_ROOT=/kaggle/input/datasets/abinayajone/adhd-eeg-dataset
```

**On a bare environment** (fresh venv, no preinstalled ML stack):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ADHD_EEG_DATA_ROOT=/path/to/adhd-eeg-dataset
```

Random seeds: `config.RANDOM_SEEDS = [42, 43, 44, 45, 46]`, one per repeat.
Every `.fit()` call in `src/evaluation.py` is preceded by
`models.set_all_seeds(fold_seed)` with a distinct, derived seed per outer
fold × repeat, so runs are reproducible on CPU. **GPU nondeterminism**: even
with `TF_DETERMINISTIC_OPS=1` set (done automatically in
`models.set_all_seeds`), some cuDNN convolution backward-pass kernels remain
nondeterministic on GPU; the original manuscript's own Limitations section
already reports up to ~1 percentage point / 0.014 AUC run-to-run variation
attributable to this. CPU execution is deterministic given these seeds.

## Running

```bash
# Prove the pipeline works, on synthetic data, in ~1-2 minutes on CPU:
python tests/test_synthetic_pipeline.py

# Full benchmark on the real dataset (needs ADHD_EEG_DATA_ROOT + ideally a GPU):
python run_all_experiments.py

# Faster partial runs while iterating:
python run_all_experiments.py --skip-architectures --skip-ablation --repeats 1
```

### Computational cost (real data, default config)

`OUTER_FOLDS=5`, `INNER_FOLDS=4`, `REPEATS=5`. Per repeat, per architecture:
5 outer folds × (4 inner-CV models for epoch selection + 1 final model) = 25
CNN trainings. Across `REPEATS=5` and 4 architectures (CNN, EEGNet,
ShallowConvNet, DeepConvNet) that is **500 CNN trainings** for the main
nested+repeated CV alone, plus 25 more per ablation config (5 configs ×
single repeat = 125), plus 25 for the resolution experiment. Use
`--outer-folds`, `--inner-folds`, `--repeats`, `--max-epochs` to scale this
down for a first pass, and a GPU for the full run.

## Repository layout

```
src/
  config.py          all seeds/fold-counts/hyperparameters in one place
  data.py            loading, per-epoch normalization, 128->512 interpolation, synthetic generator
  models.py          build_cnn (reused verbatim) + ablation variants + EEGNet/ShallowConvNet/DeepConvNet
  coral.py           CORAL transform (reused verbatim) + covariance distance
  evaluation.py       nested+repeated leakage-corrected CV, subject-level aggregation
  statistics.py      subject-level bootstrap CI, Wilcoxon + Holm + rank-biserial effect size
  visualization.py   figure generators (Figures 4-10; see docstring for 1/2/3/11)
  explainability.py  Integrated Gradients channel/temporal importance
  sanity_checks.py   Phase 20 automated leakage/consistency checks
run_all_experiments.py   main entrypoint (Phase 17)
tests/test_synthetic_pipeline.py   synthetic self-test, actually executed in this session
AUDIT_REPORT.md      Phase 1 + Phase 16 leakage audit
FINAL_REPORT.md       Phase 22 submission-readiness report
original_notebooks/  the two audited source notebooks + the desk-rejected manuscript PDF, verbatim
results/ tables/ figures/   generated outputs (git-ignored contents except .gitkeep-style placeholders)
```

## Reproducing tables and figures

`run_all_experiments.py` writes every table to `tables/*.csv` and every
figure to `figures/*.png`, all derived only from `results/predictions.csv`
(and `results/coral_results.csv`, `results/fold_records.csv`) — nothing in
`tables/` or `figures/` is hand-typed. To regenerate tables/figures from an
existing `results/predictions.csv` without retraining, import the relevant
`src.statistics` / `src.visualization` functions directly (see
`notebooks/adhd_robust_benchmark.ipynb` for worked examples of each).

## Software versions used to build and validate this framework

Validated in-session with: `numpy` 2.4.6, `scipy` 1.17.1, `scikit-learn`
1.9.0, `pandas` 3.0.5, `tensorflow-cpu` 2.21.0, Python 3.11.15, CPU only (no
GPU available in the development sandbox). These are newer than the
TensorFlow build the original Kaggle notebooks ran on (visible in their
saved outputs as TF using `Tesla T4` GPUs); if you reproduce on Kaggle with
an older/GPU TensorFlow build, re-run `tests/test_synthetic_pipeline.py`
first to confirm the pipeline still behaves identically in that environment
before trusting real-data numbers from it.
