# Submission-Readiness Report (Phase 22)

This document is an honest accounting of what this revision effort actually
produced, what it did not, and exactly why — per the review brief's explicit
instruction not to write manuscript claims based on expected rather than
computed results, and per this session's standing instruction to flag
uncertainty rather than assert it away.

## 1. The environment this was built in could not run the real experiments

Two facts governed everything below and need to be stated plainly:

1. **No access to the real dataset.** The environment this framework was
   developed in is a fresh, empty repository with no network path to
   Kaggle's `abinayajone/adhd-eeg-dataset` (the only place the original
   notebooks source their `.mat` files from, at `/kaggle/input/...`). I
   confirmed this by checking `os.path.isdir(DATA_ROOT)` — it does not
   exist here, and I did not attempt to fabricate or hallucinate a
   substitute.
2. **No GPU.** `nvidia-smi` is not installed and TensorFlow reports zero
   physical GPU devices. The original notebooks' own logs show they ran on
   dual Tesla T4 GPUs on Kaggle.

Given these two constraints, **no number produced in this session is a real
result**, and none is presented as one anywhere in this repository. What I
built instead:

- A complete, leakage-corrected implementation of every phase in the review
  brief that is expressible as code (Phases 1–2, 4–21 as code; Phase 3 and
  6's *architectures* are implemented, their *real-data results* are not).
- A synthetic surrogate dataset generator (`src/data.py:make_synthetic_dataset`)
  producing the same array shapes, subject/epoch structure, and class
  balance pattern as the real dataset, but with random data.
- `tests/test_synthetic_pipeline.py`, which I actually executed in this
  session (CPU, `tensorflow-cpu==2.21.0`), exercising every code path: nested
  CV, repeated CV with independently-seeded weight init, classical
  classifiers, CORAL, the ablation configs, EEGNet/ShallowConvNet/DeepConvNet,
  the 128→512 interpolation reshape, subject-level aggregation, bootstrap CI,
  Wilcoxon+Holm+effect-size comparison, and Integrated Gradients — and all 14
  of the Phase 20 sanity checks passed. Full log excerpted below (§4).

This is the honest ceiling of what "actually implement and execute the code"
can mean without the real dataset or a GPU: the code executes correctly; its
output on real EEG is unknown until you run it there.

## 2. What was successfully implemented (code, verified executable)

| Phase | Status | Notes |
|---|---|---|
| 1. Audit | Done | `AUDIT_REPORT.md` |
| 2. Clean framework | Done | `src/`, config-driven, all seeds in one place |
| 3. 128 vs 512 Hz | Code done, **not run on real data** | `data.resample_pipeline_b`; correctly labelled interpolation, not reproduction (see AUDIT_REPORT.md §5) |
| 4. Nested CV | Done, verified on synthetic data | `evaluation.run_nested_repeat` — fixes the L1 leakage finding |
| 5. Repeated CV | Done, verified on synthetic data | independently seeded per fold/repeat |
| 6. EEGNet/Shallow/DeepConvNet | Code done, verified runs on synthetic data, **not run on real data** | `src/models.py`; verify layer widths/kernels against the original papers before citing exactly |
| 7. Classical baselines | Done, verified on synthetic data | LR/LinearSVM/RBF-SVM/RF/GNB/KNN, class_weight documented |
| 8. Subject-level evaluation | Done, verified on synthetic data | mean-probability and majority-vote aggregation |
| 9. Confidence intervals | Done, verified on synthetic data | subject-level bootstrap only |
| 10. Statistical testing | Done, verified on synthetic data | Wilcoxon + Holm + rank-biserial effect size |
| 11. Balanced metrics | Done | balanced accuracy/sensitivity/specificity/MCC/AUC computed for every model by default |
| 12. Ablation | Code done, verified on synthetic data, **not run on real data** | full / no-spatial / no-temporal / no-BN / no-pooling configs; D/E/F handled as evaluation-protocol choices, not new architectures |
| 13. CORAL | Code done, verified on synthetic data, **not run on real data** | explicitly labelled transductive; covariance distance saved before/after |
| 14. Domain shift / subject variability | Partially done | subject-level probability/correctness table falls out of Phase 8's output; no separate feature-distance-between-subjects analysis was built (time/scope) |
| 15. Explainability | Code done, verified on synthetic data, **not run on real data** | Integrated Gradients only (no SHAP/Grad-CAM implementation — see §3) |
| 16. Leakage checklist | Done | `AUDIT_REPORT.md` §6 |
| 17. Reproducibility | Done | `requirements.txt`, `README.md`, `run_all_experiments.py` |
| 18. Tables | Partially done | `run_all_experiments.py` generates Tables 3/4/9/10 directly; Tables 1/2/5/6/7/8 require a real run (their generation code exists in `notebooks/adhd_robust_benchmark.ipynb` and `src/statistics.py`, but there is no real data to populate them with here) |
| 19. Figures | Partially done | Figures 4–10 generator functions exist and were smoke-tested structurally; Figures 1/2 are explicitly out of scope for code generation (see `src/visualization.py` docstring); Figure 11 code exists but was not run to produce a saved image on real data |
| 20. Sanity checks | Done, **actually run**, all 14 passed | see §4 below |
| 21. Efficiency | Done | feature reuse within a fold, checkpointing pattern preserved, cost estimate in README |
| 22. This report | Done | you're reading it |

## 3. What could not be completed, and why

- **No real numerical results for any experiment.** Root cause: no dataset
  access, no GPU, in this development environment (§1). This is not a
  scope-reduction choice — it is a hard environmental constraint.
- **No SHAP or Grad-CAM implementation**, only Integrated Gradients. Given
  the environment constraints above, implementing multiple explainability
  methods without ever being able to validate any of them on real EEG
  seemed a poor use of effort relative to getting the core leakage fix,
  nested CV, and statistics right. Integrated Gradients needs no extra
  dependency beyond TensorFlow already in use and is directly applicable to
  the Keras models here.
- **Figures 1 and 2** (workflow diagram, architecture diagrams) are
  explicitly not code-generated — they are conceptual diagrams better
  authored directly in whatever tool produces the rest of the manuscript's
  figures, not synthesized from result tables that don't exist yet.
- **True 512 Hz reproduction is impossible**, not merely undone — see
  `AUDIT_REPORT.md` §5. The released files contain no information beyond
  what a 128 Hz assumption implies; interpolation cannot recover it. The
  code correctly implements the interpolation sensitivity experiment (Phase
  3's fallback instruction) instead, and labels it as such everywhere.
- **Domain-shift feature-distance-between-subjects analysis** (Phase 14,
  second half) was not built; only the subject-level probability/
  correctness/confidence table that Phase 8 already produces.
- **Tables 1, 2, 5, 6, 7, 8** are not materialized as CSVs in this
  repository because they require real per-subject/per-fold results as
  input; the code to produce them from `results/predictions.csv` exists
  (`notebooks/adhd_robust_benchmark.ipynb`, `src/statistics.py`) and was
  exercised on synthetic data, but running it is meaningless without real
  numbers to summarize.

## 4. Evidence the framework actually runs (synthetic data only)

`tests/test_synthetic_pipeline.py`, executed in this session on
`tensorflow-cpu==2.21.0`, CPU only, finished in 298 seconds with exit code 0.
All 14 Phase-20 sanity checks passed:

```
=== Sanity check report: 14/14 passed ===
[PASS] 1_subject_in_exactly_one_outer_test_fold_per_repeat
[PASS] 2_no_subject_in_both_outer_train_and_test
[PASS] 3_inner_val_disjoint_from_inner_train
[PASS] 4_outer_test_never_used_for_early_stopping_or_selection
[PASS] 5_outer_test_labels_never_used_for_model_selection
[PASS] 6_normalization_leakage_free
[PASS] 7_features_from_correct_fold_model
[PASS] 8_classifiers_fit_on_outer_train_features_only
[PASS] 9_predictions_have_subject_ids
[PASS] 10_pooled_metrics_reconstructable_from_saved_predictions
[PASS] 11_subject_level_aggregation_validated
[PASS] 12_bootstrap_ci_is_subject_level
[PASS] 13_repeats_use_distinct_seeds
[PASS] 14_no_duplicate_subjects_from_epoching
```

Also verified in that run: CORAL reduced train/test feature covariance
distance in all 6 folds it was tried on; `aggregate_subject_level` correctly
raised `ValueError` when fed deliberately-corrupted inconsistent labels;
EEGNet/ShallowConvNet/DeepConvNet all built, trained, and predicted without
shape errors (26,193 / 146,929 / 340,629 parameters respectively at the
synthetic input size); the 128→512-equivalent interpolation produced exactly
the expected doubled sample count. Synthetic-data accuracies cluster near
50% (chance) — expected and unconcerning: the smoke-test's training budget
(`max_epochs=6`, inner-selected epoch counts of 1–4) is deliberately tiny
for speed, not tuned for the synthetic signal to be learnable; the test
verifies mechanics, not learning capacity. Full raw log is available on
request; `results/SYNTHETIC_predictions.csv` and `results/SYNTHETIC_coral.csv`
(committed) are the actual saved output of that run.

## 5. Exact commands to reproduce on the real dataset

```bash
pip install -r requirements.txt
export ADHD_EEG_DATA_ROOT=/path/to/adhd-eeg-dataset   # Kaggle input mount or local copy
python tests/test_synthetic_pipeline.py                # optional: re-confirm mechanics in your env first
python run_all_experiments.py                          # full run — see README "Computational cost"
```

Partial/faster runs while iterating:
```bash
python run_all_experiments.py --skip-architectures --skip-ablation --repeats 1 --outer-folds 3
```

## 6. Manuscript claims that must be softened or removed, and why

- **"Grouped 10-fold cross-validation"** as the paper's central validation
  claim must be qualified: the CNN's early stopping in every result-producing
  cell used the outer test fold as `validation_data` (AUDIT_REPORT.md finding
  L1). The reported 65–70% pooled accuracy range was obtained under a
  procedure with this leakage; it is not known whether, or by how much, the
  corrected nested procedure in this repository changes those numbers,
  because it has not yet been run on the real data. The manuscript should not
  claim the reported numbers came from a leakage-free procedure until
  `run_all_experiments.py` has actually been run and Table 3 regenerated.
- **The repeated single-split analysis's conclusions** ("previously reported
  accuracy exceeded the maximum achievable through favorable sample splitting
  alone" for LR/NLSVM/RF) rest on 20/50 repeats that share the L1 leakage
  *and* uncontrolled weight initialization (manuscript's own Limitations
  section already flags the latter). Both issues are fixed in
  `src/evaluation.py`'s repeated-CV path (independently seeded per repeat,
  inner-CV early stopping), but the conclusion should be treated as
  provisional until re-run under the corrected procedure.
- **CORAL results** ("no significant improvement") should keep the negative
  finding but must add the transductive-vs-inductive distinction this audit
  surfaced (AUDIT_REPORT.md L3): the original framing did not tell readers
  that CORAL's covariance target was estimated from the unlabelled test set.
  This doesn't invalidate the negative result, but the Methods text should
  say so explicitly rather than implying CORAL was evaluated under the same
  assumptions as every other classifier.
- **The 128 Hz sampling-rate assumption** is already appropriately hedged in
  the manuscript's Limitations section — keep that language, and add that a
  128→512 Hz *interpolation* sensitivity experiment (not a reproduction) is
  now available via `data.resample_pipeline_b`, with results pending a real
  run.
- Do **not** adopt language implying "previous reported accuracies are
  simply inflated/wrong" (the review brief explicitly asks to avoid this
  framing) — the more defensible claim, once real nested-CV numbers exist,
  is the one the brief proposes: that subject-independent performance and
  its apparent variability are jointly shaped by validation strategy,
  temporal resolution, architecture, and subject heterogeneity — stated only
  to the extent the real results actually support each piece.

## 7. What you need to do next

1. Run `run_all_experiments.py` (or `notebooks/adhd_robust_benchmark.ipynb`
   cell by cell) on the real dataset, ideally with a GPU, using the
   `--repeats 1 --skip-architectures --skip-ablation` fast path first to
   confirm real-data behavior before committing to the full multi-hour run.
2. Re-run `tests/test_synthetic_pipeline.py` in that same environment first
   if the TensorFlow/scikit-learn versions differ from those listed in
   `README.md`, to catch any environment-specific breakage before spending
   real compute.
3. Once `results/predictions.csv` exists from a real run, regenerate Tables
   1–10 and Figures 4–11 (functions already exist; see
   `notebooks/adhd_robust_benchmark.ipynb`) and only then update manuscript
   text with the actual numbers.
4. Author Figures 1–2 (workflow / architecture diagrams) by hand.
5. Revisit §6 of this report against the real results before finalizing any
   manuscript claim.
