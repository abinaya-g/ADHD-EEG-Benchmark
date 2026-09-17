# Audit Report — ADHD-EEG Robust Benchmark Framework

This is a from-scratch audit of the **current repository** (`src/`,
`run_all_experiments.py`, `tests/`, `notebooks/`), not the original two
notebooks — that earlier audit is preserved unchanged below as Appendix A.
Every row below was checked by reading the actual code and, where a real
execution exists, the actual saved/reported output — not by assuming a
function's existence means the experiment happened. Where "Actually
Executed?" says yes, the evidence column names exactly what was inspected.

## Phase 1 — Experiment inventory

| Experiment | Implemented? | Actually Executed? | Valid? | Evidence/File | Action Required |
|---|---|---|---|---|---|
| Nested (outer+inner) GroupKFold CV | Yes | Yes — once, real data, 5 outer × 4 inner × 5 repeats, 508 epochs/121 subjects | Yes | `src/evaluation.py:run_nested_repeat`, `select_n_epochs_via_inner_cv`; real run's own `sanity check report: 14/14 passed` (user-provided Kaggle notebook transcript, cell 8/9) | **Output not currently in this repo** — that Kaggle session died before the results were pushed/persisted (confirmed by user). Re-execute once; no code fix needed. |
| GroupKFold / seeded-shuffled grouped K-fold (sklearn has no `random_state` for `GroupKFold`, so a custom seeded variant is used for outer folds — see Appendix A6) | Yes | Yes (same run) | Yes | `src/evaluation.py:make_grouped_shuffled_folds` (outer), plain `sklearn.GroupKFold` (inner) | None |
| Subject IDs tracked end-to-end | Yes | Yes | Yes | `src/data.py:load_dataset` (subject_id = source filename); present in every saved prediction/fold-record row | None |
| Outer test subjects never touch model fitting | Yes | Yes | Yes | `src/evaluation.py:run_nested_repeat` — final `.fit()` call has no `validation_data` argument at all | None |
| Inner validation drawn only from outer-train subjects | Yes | Yes | Yes | `src/evaluation.py:select_n_epochs_via_inner_cv` (`GroupKFold` on the outer-train subset only) | None |
| Early stopping uses inner-val only | Yes | Yes | Yes | same function; this is the fix for the ORIGINAL notebooks' critical leakage (Appendix A2, finding L1) | None |
| Model selection (epoch count) never sees outer test | Yes | Yes | Yes | median best-inner-val-loss epoch across inner folds, fixed before the final fit | None |
| Random seeds (Python/NumPy/TF, per-fold, per-repeat) | Yes | Yes | Yes — real run's sanity check 13 showed 5 distinct seeds per repeat | `src/models.py:set_all_seeds`, `fold_seed = seed*1000+outer_fold_idx` in `evaluation.py` | Documented GPU nondeterminism residual (README "Log noise/GPU nondeterminism") — not fixable, correctly disclosed rather than hidden |
| Repeated CV (5 independent repeats) | Yes | Yes, real data | Yes | `config.REPEATS=5`, `config.RANDOM_SEEDS` | Re-execute + persist (same reason as row 1) |
| Subject-level aggregation (mean-probability primary, majority-vote secondary) | Yes | Yes | Yes — unit-tested to reject inconsistent per-subject labels | `src/evaluation.py:aggregate_subject_level`; `tests/test_synthetic_pipeline.py` explicit negative test | None |
| Subject-level bootstrap 95% CI | Yes | **No, until this session** — the function existed (`src/statistics.py:bootstrap_subject_level_ci`) and was demonstrated manually in the notebook, but `run_all_experiments.py` never called it, so no CI table was ever produced by a real run | N/A before this session | — | **Fixed this session**: `src/reporting.py:table4_confidence_intervals`, wired into `run_all_experiments.py` → `TABLE_4_CONFIDENCE_INTERVALS.csv`. Verified via `--smoke-test` (see Phase 1 note below). |
| CNN (original architecture, reused verbatim) | Yes | Yes, real data | Yes | `src/models.py:build_cnn`; real run measured acc=0.512/AUC=0.518 (near chance) under the corrected protocol | Re-execute + persist |
| EEGNet / ShallowConvNet / DeepConvNet | Yes | Yes, real data, same protocol | Yes | `src/models.py`; real run: EEGNet acc=0.773/AUC=0.836, ShallowConvNet acc=0.676/AUC=0.776, DeepConvNet acc=0.520/AUC=0.573 | Re-execute + persist |
| LR / RBF-SVM (NLSVM) / RF / GNB / KNN / LinearSVM on CNN Dense-1 features | Yes | Yes, real data | Yes | `src/models.py:make_classifiers`, fit only on `outer_train` features (sanity check 8) | Re-execute + persist |
| CORAL, explicitly labelled transductive/unsupervised | Yes | Yes, real data | Yes | `src/coral.py`, `evaluation.py:run_nested_repeat(run_coral=True)`; no test **labels** used anywhere (structural) | Re-execute + persist |
| Integrated Gradients (explainability) | Yes | **No, until this session** — proven working in `tests/test_synthetic_pipeline.py` and demonstrated in the notebook, but the `--skip-explainability` CLI flag was a dead no-op in `run_all_experiments.py`: defined, never read | N/A before this session | `src/explainability.py` | **Fixed this session**: wired to train a full-dataset visualization-only model (same precedent as the ORIGINAL notebook's own filter-visualization step, `adhd-coral.ipynb` cell-7) and save channel/temporal importance CSVs. Not a performance claim. |
| 128 Hz primary pipeline | Yes | Yes, real data | Yes | `src/data.py`; sampling rate remains an unverified **assumption** (no metadata in released files — see Appendix A5) | Re-execute + persist |
| 128→512 Hz interpolation sensitivity experiment | Yes | Yes, real data, single repeat (deliberately reuses the primary run's first-repeat seed/partition for a valid paired comparison) | Yes | `src/data.py:resample_pipeline_b`; explicitly labelled `128to512_interp` everywhere, never "512Hz" | Re-execute + persist. **New this session**: `TABLE_6_128HZ_VS_128TO512.csv` + a proper paired statistical comparison (`src/reporting.py:table6_resolution_comparison`) — previously this comparison had no dedicated table. |
| Leakage (general) | Audited exhaustively | — | One critical finding (L1, Appendix A2) fixed in a prior session; two further bugs (sanity-check false-positive across experiments, Table 3/4 preprocessing conflation) found from a real run and fixed in a later session | `sanity_checks.py`, git history (`82c714e`) | None outstanding |
| Raw per-epoch predictions saved (subject_id/fold/repeat/model/preprocessing/seed) | Yes | Yes | Yes | `run_all_experiments.py:PhaseCheckpoint`, incremental per-phase CSVs | None |
| Metrics: accuracy/balanced accuracy/sensitivity/specificity/precision/F1/MCC/AUC, epoch- and subject-level | Yes | Yes | Yes | `src/evaluation.py:compute_metrics` | None |
| Paired statistical model comparison (Holm-corrected, effect size, subject-level or paired-by-fold) | Yes (`src/statistics.py:compare_all_models`) | **No, until this session** — only demonstrated manually in the notebook against epoch-level per-fold accuracy, never automated, and never using the subject-level unit the brief prefers | N/A before this session | — | **Fixed this session**: `src/reporting.py:table5_model_comparison` — pairs models using the per-outer-fold **subject-level** metric, only across (repeat, outer_fold) keys the two groups actually share (so e.g. a single-repeat ablation config is validly compared only against the matching repeat of the 5-repeat primary run, never padded). Wired to `TABLE_5_MODEL_COMPARISON_STATISTICS.csv`. |
| Dataset summary table | No | No | N/A | — | **New this session**: `src/reporting.py:table1_dataset_summary` → `TABLE_1_DATASET_SUMMARY.csv` |

**Note on "Actually Executed? Yes, real data" rows above**: these describe a
run that genuinely happened (12,881s / ~3.6 hours on Kaggle, 5×4×5 nested CV
+ 3 extra architectures + 5 ablation configs + the resolution experiment,
all 14 leakage sanity checks passing, verified directly from the notebook's
own saved cell outputs the user shared). **None of that run's numeric output
currently exists in this repository or anywhere durable** — the Kaggle
session ended before `results_final/`/git-push persistence (added only
after that run) existed, and the working directory was ephemeral. This is a
**storage/infrastructure gap, not a methodology or implementation defect**:
the design was validated correct at execution time. See
`FINAL_EXPERIMENT_REPORT.md` for the exact re-run instructions.

## Phase 2 — Subject-level leakage audit

The implemented data flow, verified against the code (not assumed):

```
Outer train subjects (GroupKFold-style partition, seeded per repeat)
      |
      v
Inner GroupKFold on the OUTER-TRAIN subjects only (never touches outer-test)
      |
      v
Per inner fold: train with early stopping monitoring INNER-validation loss only
      |
      v
Take the MEDIAN best-inner-val-loss epoch count across inner folds
      |
      v
Model fixed: retrain ONE model on ALL outer-train subjects for exactly that
many epochs -- no validation_data argument at all in this call, so nothing
about outer-test can influence it, structurally (not by convention)
      |
      v
Completely unseen outer-test subjects: model.predict() called exactly once
      |
      v
Epoch-level predictions saved (with subject_id, fold, repeat, model, seed)
      |
      v
Subject-level aggregation (mean predicted probability across that subject's
test epochs; majority vote saved alongside for comparison)
```

Checked directly against the 8 requirements in the review brief:

1. **Every subject assigned entirely to one outer fold** — `make_grouped_shuffled_folds` assigns each unique subject to exactly one fold index via a dict (`fold_of_subject`), so no subject can appear in two outer-test sets *within the same (repeat, architecture, preprocessing) design*. Verified at runtime by `sanity_checks.py` check 1 (14/14 passed on the real run). Note: a subject legitimately appears in the outer-test set of *both* the primary run and, separately, the resolution experiment, when those two experiments deliberately reuse the same seed — that is two independent designs sharing a partition on purpose, not one design double-assigning a subject; check 1 groups by `(repeat, architecture, preprocessing)` specifically to not conflate the two (see the sanity-check bug fixed in commit `82c714e`).
2. **No subject in both outer train and outer test** — check 2, row-level, 14/14 passed.
3. **Outer-test subjects never used for hyperparameter/model selection, early stopping, threshold selection, feature-normalization fitting, preprocessing-parameter estimation, CORAL fitting, or feature scaling** — verified structurally per item:
   - Hyperparameter/model selection & early stopping: covered by the data-flow diagram above (inner-only).
   - Threshold selection: fixed at 0.5 everywhere, never tuned on any data — no leakage possible because there is no tuning step.
   - Feature normalization: `data.normalize_epoch` computes mean/std from *that single epoch's own samples only* — no statistic is ever estimated across epochs or subjects, so no split (outer, inner, or repeat) can leak through it. This was true even in the ORIGINAL (audited) notebooks and is unchanged.
   - CORAL fitting: `coral_transform(train_feats, test_feats)` uses the outer-test subjects' **unlabelled** Dense-1 feature covariance only — no test *label* is ever read. This is a transductive unsupervised setting, explicitly tagged as such in every saved CORAL result row (`adaptation_type` column), never presented as an ordinary inductive comparison.
   - Feature scaling: no scaler beyond per-epoch normalization exists anywhere in the pipeline (classical classifiers are fit directly on CNN Dense-1 features, unscaled, matching the original notebooks).
4. **Any scaler/normalization/PCA fitted only on appropriate training data** — no PCA exists in this pipeline; the only "fitting" step is per-epoch normalization, which needs no cross-sample fitting at all (see above).
5. **CNN-derived features generated without outer-test leakage** — `feature_model` and `model` are returned from the *same* `build_fn()` call, used only within that fold's loop iteration; the model that produces test-set features is exactly the model whose training never saw the outer test set (sanity check 7).
6. **Inner validation created exclusively from outer-training subjects** — `select_n_epochs_via_inner_cv` receives only `X_tr/y_tr/groups_tr` (the outer-train subset); its own `GroupKFold` split is internal to that subset (sanity check 3).
7. **Early stopping uses only the inner validation set** — confirmed above; the *final* fit (the one whose weights get evaluated) has no early stopping and no validation split at all (sanity check 4).
8. **Final outer-test evaluation performed only after the model is completely fixed** — `model.predict(X_te)` is called after `.fit()` returns; `y[outer_test_idx]` (the labels) are never read by any code path before that point (sanity check 5).

All 14 automated checks (`src/sanity_checks.py`) passed both on synthetic
self-test data (`tests/test_synthetic_pipeline.py`, run in this session,
438s) and on the real 508-epoch dataset (the now-lost Kaggle run, verified
from the user-shared transcript). No violation of the above structure was
found in either run.

---

## Appendix A: Leakage & Methodology Audit of the ORIGINAL two notebooks (prior session)

Audited by: automated revision assistant, for the manuscript *"Assessing the
Robustness of CNN and Hybrid Classifiers for Subject-Independent ADHD
Detection from Raw EEG"* (desk-rejected from *Computer Methods and Programs
in Biomedicine Update*). Source notebooks are archived verbatim in
`original_notebooks/` for provenance. This document is Phase 1 (audit) and
Phase 16 (final leakage-control table) of the revision brief.

### A1. Pipeline inventory

| Stage | Location in original notebooks | Behavior |
|---|---|---|
| Data loading | `adhd-baseline.ipynb` cells 0–6, reused in `adhd-coral.ipynb` cell 0 | `.mat` files under `ADHD_part{1,2}` / `Control_part{1,2}`, loaded with `scipy.io.loadmat`. One file = one subject. Subject ID = filename. |
| Epoching | same cells | Fixed `epoch_len = fs_assumed * 30 = 3840` samples; each file's array (`n_samples, 19`) is cut into non-overlapping 3840-sample epochs; a trailing partial segment is discarded (`n_samples // epoch_len`). |
| Array shapes | verified by re-running the loading logic (`src/data.py:load_dataset`, mirrors the notebooks exactly) | `X: (508, 19, 3840, 1)`, `y: (508,)`, `groups: (508,) -> 121 unique subjects`. 61 ADHD subjects (30+31 files) / 60 Control subjects (30+30 files); 288 ADHD epochs / 220 Control epochs. Matches the manuscript's stated dataset characteristics. |
| Normalization | `adhd-baseline.ipynb` cell 9 / cell 24; `adhd-coral.ipynb` cell 0 | `normalize_epoch()`: per-epoch, per-channel z-score using **that epoch's own** mean/std only. |
| CNN architecture | `adhd-baseline.ipynb` cell 24; `adhd-coral.ipynb` cell 0 | `build_cnn()`: Conv2D(16,10×1)→BN→AvgPool→Conv2D(16,4×1)→BN→AvgPool→reshape→Conv1D(32,fs/4)→BN→AvgPool→Conv1D(32,fs/8)→BN→AvgPool→Flatten→Dense(64,"dense1")→Dense(32,"dense2")→Dense(1,sigmoid). `feature_model` outputs `dense1` (64-d) alongside `flat`/`dense2` in some cells. Matches the manuscript's described architecture. |
| Feature extraction for hybrid classifiers | `adhd-baseline.ipynb` cells 10/17/19/22/25/27; `adhd-coral.ipynb` cells 3/4/8 | `feature_model.predict(...)` on the Dense-1 layer, computed separately for the fold's train and test partitions from the **same fold-specific trained CNN**. |
| Outer CV | all main-result cells | `GroupKFold(n_splits=10)` on `groups` (subject IDs) — correctly prevents a subject's epochs from being split across train/test. |
| Classical classifiers | `adhd-baseline.ipynb` cells 17–30; `adhd-coral.ipynb` cell 8 | LR, RBF-SVM ("NLSVM"), RF, GNB, KNN, fit on `train_feats`/`y_fold_train`, evaluated on `test_feats`. `class_weight="balanced"` added for LR/SVM/RF partway through the baseline notebook (cells 21–22 onward); GNB/KNN have no such parameter in scikit-learn. |
| Repeated single-split analysis | `adhd-coral.ipynb` cells 10–12 | `GroupShuffleSplit(test_size=0.2)`, 1 / 20 / 50 repeats, `random_state=repeat_i` controlling **only the split**, not CNN weight initialization. |
| CORAL | `adhd-coral.ipynb` cells 1–6 | `coral_transform()` whitens fold-train Dense-1 features and re-colors them with fold-**test** Dense-1 features' covariance (unlabelled), then fits LR on the aligned train features and evaluates on (unaligned) test features. |
| Statistics | `adhd-baseline.ipynb` cells 18/30; `adhd-coral.ipynb` cell 6 | Paired Wilcoxon signed-rank test across the 10 outer folds; no multiple-comparison correction, no effect size. |
| Seeding | `adhd-baseline.ipynb` cell 23; used at the top of cells 25/27; `adhd-coral.ipynb` "`set_all_seeds(42)`" before its fold loops | `os.environ["PYTHONHASHSEED"]`, `random.seed`, `np.random.seed`, `tf.random.set_seed` — called **once** before a 10-fold loop, or once before a 20/50-repeat loop. TensorFlow's global RNG state advances across the loop's successive `build_cnn()`/`.fit()` calls, so folds/repeats are not independently, reproducibly seeded relative to each other. |
| Checkpointing | throughout | Results pickled to `/kaggle/working/*.pkl` after every fold — good practice for surviving Kaggle session drops, but the pickled objects hold only pooled `y_true`/`pred`/`proba` lists, not subject IDs or fold IDs per prediction, so per-subject / per-fold reconstruction after the fact is not directly possible from those checkpoint files as saved. |
| Sampling rate | `adhd-baseline.ipynb` cells 0–3 | `.ced` channel-location file inspected for sampling-rate metadata; none found. `fs_assumed = 128` is stated everywhere as an **assumption**, consistent with the manuscript's own disclosure. |

### A2. Leakage findings

### L1 — CRITICAL: outer test fold used as `validation_data` for early stopping

Every fold-loop cell that produces a manuscript-reported number (`adhd-baseline.ipynb`
cells 17, 19, 22, 25, 27; `adhd-coral.ipynb` cells 3, 4, 8, 10, 11, 12) calls:

```python
model_fold.fit(X_fold_train, y_fold_train, epochs=100, batch_size=16, verbose=0,
                validation_data=(X_fold_test, y_fold_test),
                callbacks=[early_stop_fold])
```

`early_stop_fold` monitors `val_loss` with `restore_best_weights=True`. Because
`val_loss` here **is** the outer test fold's loss, the checkpoint that ends up
being evaluated on that same outer test fold was selected *using* that fold's
loss trajectory. This is model-selection leakage: the reported "held-out"
performance was obtained by a stopping rule that already looked at the held-out
set. It inflates and destabilizes the reported numbers (the run-to-run
"minor" 1pp/0.014-AUC variation the manuscript's own Limitations section
notes is a symptom of exactly this: different early-stopping checkpoints get
selected on effectively-noisy small test folds).

Notably, `adhd-baseline.ipynb` cells 12–16 *did* build a proper inner
train/validation split (`GroupShuffleSplit` on `groups[train_idx]`, i.e.
subjects drawn only from the outer training set) for a small hyperparameter
grid search over `patience`/`learning_rate`. That leakage-free inner-validation
pattern exists in the notebook but was **not** carried into the actual
10-fold result-producing cells that follow — those revert to
`validation_data=(X_fold_test, y_fold_test)`. One cell (`adhd-baseline.ipynb`
cell 17) even contains an inline comment acknowledging this is "a milder form
of the leakage concern... worth stating explicitly as a limitation," but the
main results (cells 19 onward, and the entirety of `adhd-coral.ipynb`) do not
carry that caveat forward and use it as the paper's principal evaluation.

**Fix implemented:** `src/evaluation.py:run_nested_repeat` /
`select_n_epochs_via_inner_cv`. For every outer fold, an inner `GroupKFold`
(default 4 folds) is drawn from the outer-training subjects only; early
stopping runs against inner-validation subjects; the epoch count is fixed to
the median best-inner-val-loss epoch across inner folds; a fresh model is
then trained on **all** outer-training subjects for exactly that many epochs
(no validation split, so nothing about the outer test set can influence this
final fit); only then is the outer test fold touched, exactly once, for
prediction. See `src/sanity_checks.py` checks 3–5, which assert this
structurally.

### L2 — Repeated single-split analysis conflates split variance with untracked weight-init variance

`adhd-coral.ipynb` cells 10–12 (single/20/50 repeats) also carry the L1 issue
(`validation_data=(X_test_r, y_test_r)`), *plus* the manuscript's own
Limitations section already flags that CNN weight initialization is not
independently controlled across repeats — confirmed by code inspection:
`set_all_seeds` is not re-called inside the repeat loop, so each repeat's
`build_cnn()`/`.fit()` draws from wherever TensorFlow's global RNG state
happened to land after the previous repeat, not from a fixed, reproducible,
repeat-specific seed.

**Fix implemented:** `src/evaluation.py` derives a `fold_seed =
seed*1000 + outer_fold_idx` per fold and calls `models.set_all_seeds(fold_seed)`
immediately before every `build_fn()` call (both inner-CV epoch-selection runs
and the final per-fold fit). Each of Phase 5's `REPEATS` independent
repetitions uses a distinct top-level seed from `config.RANDOM_SEEDS`, so
weight initialization is both independently controlled *and* reproducible.

### L3 (not leakage, but a validity concern) — CORAL's covariance target is the unlabelled test set

`coral_transform(train_feats, test_feats)` estimates `cov_target` from the
outer-fold **test** subjects' Dense-1 features (no test *labels* are used,
only the unlabelled feature covariance). This is a legitimate
transductive unsupervised domain-adaptation setup, but the original notebook
and manuscript do not distinguish it from an ordinary inductive
subject-independent evaluation — Table/Figure captions describe it simply as
"LR with/without CORAL" under the same 10-fold GroupKFold umbrella as every
other classifier, which invites readers to assume it was evaluated under the
same assumptions.

**Fix implemented:** `src/evaluation.py:run_nested_repeat(run_coral=True)`
tags every CORAL result row with
`adaptation_type = "transductive_unsupervised (test features, no test labels, used to estimate target covariance)"`,
and saves covariance distance before/after per fold so this can be reported
as a quantified, explicitly-labelled experiment (Phase 13 / Table 9), never
folded into the standard inductive comparison table.

### Not leakage — normalization

`normalize_epoch()` computes mean/std from **that single epoch's own 3840
samples**, independently for every epoch, before any train/test split is
drawn. No statistic is ever estimated across epochs or across subjects, so no
choice of split (outer, inner, or repeat) can leak information through this
step. Confirmed by direct inspection and preserved unchanged in
`src/data.py:normalize_epoch`.

### Not leakage — GroupKFold subject grouping

`GroupKFold(n_splits=10)` on subject-ID groups is correct and was already
leakage-free in the original notebooks: no subject's epochs are ever split
across train and test within a fold. This is retained (generalized to a
seeded, repeatable, class-balanced variant — see `src/evaluation.py:
make_grouped_shuffled_folds` — because plain `GroupKFold` has no
`random_state` and therefore cannot produce the different subject
partitions Phase 5's repeated-CV design calls for).

### A3. Reusable functions (kept, not rewritten)

- `build_cnn()` — reused verbatim in `src/models.py`, same layer names,
  same default `fs=128` kernel scaling.
- `normalize_epoch()` — reused verbatim in `src/data.py`.
- Data-loading/epoching logic — reused verbatim in `src/data.py:load_dataset`.
- `coral_transform()` / `covariance_distance()` — reused verbatim in
  `src/coral.py`, only their *usage/reporting* changed (see L3).
- `compute_metrics()` — extended (balanced accuracy, sensitivity,
  specificity, MCC added; accuracy kept on the 0–1 scale rather than ×100
  since downstream statistics code expects a consistent scale) but built on
  the same `sklearn.metrics` calls as the original.

### A4. Missing experiments identified in the original notebooks

- No inner/outer nested design (see L1).
- No subject-level aggregation or subject-level metrics anywhere — every
  reported number pools all 508 epochs.
- No bootstrap confidence intervals of any kind.
- No Holm/Bonferroni correction across the many pairwise Wilcoxon tests run
  (CNN vs. 5 classifiers × 2 metrics = 10 tests, all at raw α=0.05).
- No effect size for the Wilcoxon comparisons.
- No EEGNet / ShallowConvNet / DeepConvNet baseline — only the one custom CNN.
- No architecture ablation.
- No true 512 Hz reproduction, and no explicit 128→512 Hz *interpolation*
  sensitivity experiment either (the manuscript only discusses the
  resolution difference qualitatively in its Discussion).
- No explainability / attribution analysis.
- No formal leakage checklist (this document is the first).

### A5. Phase 3 note: is exact 512 Hz reproduction possible from the released files?

No. The released `.mat` files contain exactly the sample counts consistent
with a 128 Hz assumption (confirmed by `adhd-baseline.ipynb` cells 3–4: no
sampling-rate field in the `.ced` file or in any `.mat` key, and per-file
sample counts are exact multiples of `128 × 30 = 3840`, not `512 × 30 =
15360`). There is no way to recover the higher-frequency information the
original 512 Hz acquisition would have contained — that information was
never in this release to begin with. `src/data.py:resample_pipeline_b`
implements FFT-based interpolation (`scipy.signal.resample`) from the
existing 128 Hz samples up to a 512 Hz-shaped array **for sensitivity-analysis
purposes only**; it manufactures no new information and must never be
described as "the original 512 Hz pipeline" or "reproduction of the
original acquisition" in any manuscript text, table, or figure caption. Every
function and output that touches this path is labelled `128to512_interp` /
"interpolation" throughout this codebase specifically to prevent that
conflation.

### A6. Leakage-control table (original notebooks)

| Component | Potential leakage | Original implementation | Corrected implementation |
|---|---|---|---|
| Epoching | Epoch boundaries computed before splitting | Fixed per-file cut, no leakage possible (epochs never cross files/subjects) | Unchanged (`src/data.py:load_dataset`) |
| Normalization | Fitting scale/shift on pooled data before CV | Per-epoch, per-channel, independent of split | Unchanged; confirmed leakage-free (`src/data.py:normalize_epoch`) |
| Outer split | Subject appears in both outer train/test | `GroupKFold` on subject IDs — already correct | Unchanged in spirit; made seed-repeatable (`evaluation.make_grouped_shuffled_folds`) |
| CNN early stopping | **Outer test used as `validation_data`** | Yes — every result-producing cell | Inner `GroupKFold` on outer-train subjects only selects epoch count; final fit uses no validation split (`evaluation.select_n_epochs_via_inner_cv`, `run_nested_repeat`) |
| CNN weight init across folds/repeats | Global seed set once, not per fold/repeat | Yes | `fold_seed` derived and re-seeded before every `build_fn()` call (`models.set_all_seeds`) |
| Feature extraction (Dense-1) | Using a CNN trained/selected via leaked early stopping | Yes, inherits L1 | Inherits the L1 fix — features come from the corrected per-fold model |
| Classifier fitting | Fitting on pooled or test features | No — always `train_feats` only | Unchanged; verified structurally (`sanity_checks.py` check 8) |
| Hyperparameter tuning | Tuned using outer test performance | The one grid-search cell (baseline nb. cell 13) *did* use an inner split correctly, but was not connected to the main results | Inner-CV pattern generalized and applied to every experiment, not a one-off cell |
| Threshold selection | Threshold chosen using test-set performance | Fixed 0.5 everywhere — no leakage | Unchanged (fixed 0.5); documented explicitly |
| CORAL | Using test-set information (labels) for model fitting | Uses unlabelled test-feature covariance only — no label leakage, but transductive | Unchanged mechanics; now explicitly labelled transductive/unsupervised in every saved result row and table (`evaluation.run_nested_repeat(run_coral=True)`) |
| Class balancing | `class_weight` fit using test-set class distribution | No — `class_weight="balanced"` is computed from `y_fold_train` only by scikit-learn | Unchanged; not a leakage source |
| Bootstrap CI | Resampling individual epochs (pseudo-replication) | N/A (no bootstrap existed) | Implemented at the subject level only (`statistics.bootstrap_subject_level_ci`) |
| Explainability | Attribution computed using leaked model | N/A (none existed) | Uses the same corrected per-fold model; reported with cross-fold mean ± SD, not a single fold (`explainability.py`) |

---
*This audit was produced by direct inspection of `original_notebooks/adhd-baseline.ipynb` and
`original_notebooks/adhd-coral.ipynb`, cell by cell, cross-checked against the manuscript PDF
(`original_notebooks/manuscript_desk_rejected.pdf`). No claim in this document is inferred or
assumed beyond what those three source files show.*
