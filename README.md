# ADHD-EEG Robust Benchmark

A leakage-corrected, subject-independent evaluation framework for CNN and
hybrid-classifier models that classify ADHD from raw EEG: nested, repeated,
subject-grouped cross-validation with subject-level aggregation.

The original notebooks used the outer test fold as the CNN early-stopping
validation data, which lets the stopping rule see held-out subjects. This
codebase fixes that: early stopping uses inner folds drawn from the
outer-training subjects only (see `src/evaluation.py`).

## What this repository is

Code for the evaluation framework, the saved predictions it produced, and
the analysis scripts built on them. The EEG data (Kaggle dataset
`abinayajone/adhd-eeg-dataset`) are not bundled here. Every module can be
run end-to-end on a synthetic surrogate dataset with the same array shapes
and subject/epoch structure as the real one (`src/data.py:make_synthetic_dataset`,
`tests/test_synthetic_pipeline.py`) to check the pipeline's mechanics.
**No number produced by the synthetic self-test is a real result**; every
such output is prefixed `SYNTHETIC_`.

To reproduce real results, run `run_all_experiments.py` (without
`--smoke-test`) with the dataset available and, preferably, a GPU, for
example in the same Kaggle environment the original notebooks used.

## Data and results in this repository

- `results_final/predictions/all_predictions.csv` holds the epoch-level
  out-of-fold predictions from the real-data run: 34,544 rows covering 12
  model configurations (EEGNet, ShallowConvNet, DeepConvNet, the CNN, six
  CNN + classical-classifier hybrids, and the CORAL / no-CORAL variants) over
  5 repeats. Columns: `architecture, model, preprocessing, repeat,
  outer_fold, seed, subject_id, y_true, y_pred, y_proba`.
- Summary tables and figures are **derived** from these predictions (see
  "Reproducing tables and figures" below). The `tables/` and `figures/`
  folders in this repository are currently empty placeholders; regenerate
  them rather than expecting committed copies.
- `results/` contains only the `SYNTHETIC_*` self-test outputs. Nothing in
  `results/` is a real result.

## Revision analyses (`results_revision/`)

Additional analyses added during manuscript revision. Scripts are in `src/`
and the two `run_revision_*.py` drivers; outputs are in `results_revision/`.
Each script's docstring states its inputs and limitations; the short version:

| Analysis | Script | Needs |
|----------|--------|-------|
| Per-repetition results, all-pairs comparisons among leading models (Holm), label-permutation chance tests, 128 Hz vs 128->512 Hz on repetition 0 | `src/revision_analyses.py` | saved predictions only |
| Subject-level paired model comparison (one paired observation per subject; replaces an earlier 25-observation Wilcoxon test that pseudo-replicated subjects) | `src/subject_level_paired_test.py` | saved predictions only |
| Artefact proxies, group differences, relation to model outputs, band-power baseline sensitivity | `src/artefact_analysis.py` | raw EEG, CPU |
| Classical band-power baseline; leakage comparison; retrain-on-permuted-labels null | `run_revision_experiments.py` | raw EEG, CPU |
| Leakage-permitting vs subject-grouped evaluation for CNN + heads, EEGNet, ShallowConvNet, DeepConvNet; CORAL lambda sweep; resolution repetitions 1-4 | `run_revision_deep.py` (`--task leakage / coral / resolution`), summarised by `src/revision_deep_analysis.py` | raw EEG, TensorFlow, ideally GPU |
| Matched-head control for the CNN+LR comparison | `src/models_matched.py` | raw EEG, TensorFlow |
| 128 Hz vs 128->512 Hz over all five repetitions | `src/resolution_five_repetitions.py` | `results_revision/resolution_interp_predictions.csv` |
| Manuscript figures at 300 dpi | `src/fig_manuscript_hires.py`, `src/fig_confusion_hires.py`, `src/fig_architecture.py` | saved predictions |

Notes:

- **The `leakage_epoch_level_predictions_*.csv` files are leakage-permitting by
  design** (folds drawn at the epoch level, so one child can appear in both
  training and test). They exist to show how much such a split changes the
  numbers. They are not performance estimates and must not be reported as
  results.
- `versions_*.json` record the software environment of each run
  (`*_kaggle*` files are Kaggle runs).
- `notebooks/kaggle_revision_runs.ipynb` runs the TensorFlow experiments on
  Kaggle.
- `src/evaluation.py` gained two optional arguments: `only_outer_folds`
  (resume a run fold by fold) and `coral_lambdas` (the lambda sweep). The
  defaults reproduce the primary analysis, and early stopping still uses inner
  folds from the outer-training subjects only.

## Dataset

- Source: Kaggle dataset `abinayajone/adhd-eeg-dataset` (as used by the
  original notebooks), originally released by Nasrabadi et al. (2020),
  IEEE DataPort, doi:10.21227/rzfh-zn36. This repository does not
  redistribute it.
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
  (no sampling-rate metadata is present in the released files). Treat this
  as an assumption, not a verified fact, in any text you write about it.

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
fold × repeat, so runs are reproducible on CPU. **Log noise**: importing `src` (any entry point does this) sets
`TF_CPP_MIN_LOG_LEVEL=3` and raises TensorFlow's Python logger level,
which removes the repeated `NodeDef mentions attribute
use_unbounded_threadpool` lines and `tf.function retracing` warnings
(expected here, since a fresh model is built per fold by design). Two
one-time startup lines can still appear (`absl::InitializeLog` and a
oneDNN info banner) — both harmless. To also silence the oneDNN line, set
`TF_ENABLE_ONEDNN_OPTS=0` yourself before running; this is not set by
default because it can shift floating-point results slightly (per
TensorFlow's own message), and this codebase does not change numeric
behavior as a side effect of a logging request.

**GPU nondeterminism**: even
with `TF_DETERMINISTIC_OPS=1` set (done automatically in
`models.set_all_seeds`), some cuDNN convolution backward-pass kernels remain
nondeterministic on GPU; run-to-run variation of up to about 1 percentage point / 0.014 AUC has been
observed on GPU. CPU execution is deterministic given these seeds.

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
tests/test_synthetic_pipeline.py   synthetic self-test
original_notebooks/  the two original source notebooks, verbatim
results_final/predictions/   committed epoch-level out-of-fold predictions from the real-data run
results_revision/            revision analyses (see "Revision analyses")
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

Validated with: `numpy` 2.4.6, `scipy` 1.17.1, `scikit-learn`
1.9.0, `pandas` 3.0.5, `tensorflow-cpu` 2.21.0, Python 3.11.15, CPU only (no
GPU available in the validation environment). These are newer than the
TensorFlow build the original Kaggle notebooks ran on (visible in their
saved outputs as TF using `Tesla T4` GPUs); if you reproduce on Kaggle with
an older/GPU TensorFlow build, re-run `tests/test_synthetic_pipeline.py`
first to confirm the pipeline still behaves identically in that environment
before trusting real-data numbers from it.

## Limitations

- Single public dataset (121 subjects, 508 thirty-second epochs, 19 channels);
  no external validation on other cohorts, recording sites or headsets.
- The 128 Hz sampling rate is an assumption: the released files carry no
  sampling-rate metadata.
- Metrics from the original notebooks are not comparable with this framework,
  because those notebooks let the early-stopping rule see the outer test fold.
- On GPU, run-to-run variation of up to about 1 percentage point / 0.014 AUC
  has been observed; CPU runs are deterministic given the seeds.
- Outputs prefixed `SYNTHETIC_` come from the synthetic self-test and are not
  results.

## Dataset attribution

EEG data: Nasrabadi et al. (2020), IEEE DataPort,
doi:10.21227/rzfh-zn36. Please cite the original dataset if you use it.
