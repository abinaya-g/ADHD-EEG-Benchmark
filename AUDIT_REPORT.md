# Leakage & Methodology Audit — `adhd-baseline.ipynb` and `adhd-coral.ipynb`

Audited by: automated revision assistant, for the manuscript *"Assessing the
Robustness of CNN and Hybrid Classifiers for Subject-Independent ADHD
Detection from Raw EEG"* (desk-rejected from *Computer Methods and Programs
in Biomedicine Update*). Source notebooks are archived verbatim in
`original_notebooks/` for provenance. This document is Phase 1 (audit) and
Phase 16 (final leakage-control table) of the revision brief.

## 1. Pipeline inventory

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

## 2. Leakage findings

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

## 3. Reusable functions (kept, not rewritten)

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

## 4. Missing experiments identified in the original notebooks

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

## 5. Phase 3 note: is exact 512 Hz reproduction possible from the released files?

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

## 6. Phase 16 — Leakage-control table

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
