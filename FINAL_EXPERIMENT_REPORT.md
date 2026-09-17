# Final Experiment Report (Phase 18)

## 1. What was already correctly implemented (before this session's audit)

- Leakage-corrected nested subject-independent GroupKFold CV (5 outer × 4
  inner), with early stopping confined to inner-validation subjects and a
  final fit that never receives a `validation_data` argument at all.
- Repeated evaluation across 5 independently, reproducibly seeded
  repetitions.
- Subject-level aggregation (mean-probability primary, majority-vote
  secondary), with a unit test confirming it rejects inconsistent
  per-subject labels rather than silently averaging over a bug.
- The full core model set: CNN, LR, RBF-SVM, RF, GNB, KNN, LinearSVM on
  CNN Dense-1 features, plus EEGNet/ShallowConvNet/DeepConvNet as
  established-architecture baselines.
- CORAL, explicitly scoped as transductive/unsupervised (no test labels
  used, tagged as such in every saved row).
- A 128→512 Hz interpolation sensitivity experiment, explicitly and
  consistently labelled as interpolation — never claimed as a
  reproduction of the original 512 Hz acquisition (which is impossible
  from the released files — see `AUDIT_REPORT.md` Appendix A5).
- A CNN architectural ablation (5 configs: full, no-spatial, no-temporal,
  no-batchnorm, no-pooling).
- Integrated Gradients explainability code (`src/explainability.py`),
  proven correct via the synthetic self-test.
- 14 automated leakage/consistency sanity checks, run on every invocation.
- Incremental per-phase checkpointing so a mid-run crash only costs the
  phase in progress, not everything before it.

## 2. What was missing (found by this session's audit)

- No subject-level bootstrap confidence-interval table was ever produced
  by the automated script — the underlying function existed but was only
  ever demonstrated manually in the notebook.
- No automated paired statistical model-comparison table — same situation:
  the logic existed, was demonstrated once on epoch-level data in the
  notebook, but was never wired into `run_all_experiments.py`, and never
  used the subject-level (or paired-by-fold) statistical unit the brief
  requires.
- No dataset-summary table.
- No dedicated 128-vs-interpolation comparison table with a paired
  statistical test (the two conditions' results existed side by side in
  the general epoch/subject tables, but nothing compared them directly).
- The `--skip-explainability` CLI flag was defined but never read —
  explainability never ran as part of the automated pipeline.
- **The one real 3.6-hour execution on real data (5×4×5 nested CV + 3
  architectures + 5 ablation configs + the resolution experiment, all 14
  sanity checks passing) was never persisted anywhere durable.** The
  Kaggle session it ran in ended before any git-push or Drive-mount
  persistence step existed in the user's workflow, and Kaggle's working
  directory does not survive a session end. This is the single most
  consequential gap: real, valid results were produced and then lost to
  infrastructure, not to any methodological problem.

## 3. What was corrected this session

- `src/reporting.py` (new module): `table1_dataset_summary`,
  `table4_confidence_intervals` (subject-level bootstrap CI per model ×
  preprocessing group), `table5_model_comparison` (Holm-corrected paired
  comparison using the per-outer-fold subject-level metric, pairing only
  on `(repeat, outer_fold)` keys two groups actually share), and
  `table6_resolution_comparison` (128 Hz vs interpolation, with a paired
  test).
- `run_all_experiments.py`: wired all of the above into the main run,
  producing `TABLE_1` through `TABLE_7` (exact filenames per the brief's
  Phase 15) in addition to the pre-existing table files; wired
  `--skip-explainability` to an actual Integrated Gradients run (trained
  on the full dataset for visualization only, mirroring the ORIGINAL
  notebook's own precedent for filter visualization — never a performance
  claim); added a `results_final/` mirror (Phase 14's requested directory
  structure) populated only on real-data runs, separate from the working
  `results/`/`tables/`/`figures/` checkpoint directories so nothing is
  overwritten.
- `AUDIT_REPORT.md` restructured with the requested Phase 1 table
  (Experiment | Implemented? | Actually Executed? | Valid? |
  Evidence/File | Action Required) and the exact Phase 2 data-flow
  diagram, checked against the code line by line; the previous session's
  audit of the two original notebooks is preserved unchanged as
  Appendix A.
- `EXPERIMENT_DECISION.md` (new): classifies every candidate experiment
  as REQUIRED/RECOMMENDED/OPTIONAL/UNNECESSARY. No new models, no new
  architectures, no hyperparameter search were added anywhere in this
  session — every change is either automation of an already-designed
  analysis or a fix to a reporting bug.

## 4. What was actually run in this session, and its status

**No real-data execution happened in this session** — this development
environment has no GPU and no access to the dataset (see the very first
audit in this project's history). Everything above was validated against
`tests/test_synthetic_pipeline.py`'s synthetic surrogate data and a
`run_all_experiments.py --smoke-test` end-to-end run, both actually
executed in this session. Every number either of those two produces is
explicitly labelled `SYNTHETIC` and is not a real result.

The one genuinely real execution referenced throughout this report and
`AUDIT_REPORT.md` (5×4×5 nested CV, 508 real epochs, 121 real subjects,
12,881s, 14/14 sanity checks) happened in a **separate session, on
Kaggle**, before this audit began, and its numeric output no longer exists
anywhere retrievable.

## 5. Experiment configuration (as designed; to be used for the real re-run)

| Parameter | Value |
|---|---|
| Outer folds | 5 |
| Inner folds | 4 |
| Repeats | 5, seeds `[42, 43, 44, 45, 46]` |
| Max epochs (upper bound; actual per-fold epoch count is selected via inner CV) | 100 |
| Batch size | 16 |
| Early-stopping patience (inner CV only) | 8 |
| Learning rate | 1e-4 |
| Subjects / epochs (real dataset) | 121 subjects (61 ADHD / 60 Control), 508 epochs (288 ADHD / 220 Control), 19 channels, 3840 samples/epoch, assumed 128 Hz |
| Models | CNN, CNN+LR, CNN+RBF-SVM (NLSVM), CNN+RF, CNN+GNB, CNN+KNN, CNN+LinearSVM, EEGNet, ShallowConvNet, DeepConvNet |
| Ablation configs | A_full, B_no_spatial, C_no_temporal, G_no_batchnorm, H_no_pooling (single repeat, seed 42) |
| Resolution experiment | 128 Hz vs 128→512 Hz interpolation (single repeat, seed 42, same subject partition as the primary run's repeat 0) |
| CORAL | transductive/unsupervised, no test labels, within the primary 5×4×5 run |

## 6-14. Subject-level / epoch-level results, confidence intervals, statistical tests, 128 vs 512 results, CORAL results, explainability results

**Not available.** These require the real execution described in §4/§5,
which has not happened since this audit's code fixes landed. The prior
(now-lost) real run's headline epoch-level numbers were relayed earlier in
this conversation (e.g. CNN acc≈0.512/AUC≈0.518 near chance;
CNN+RBF-SVM acc≈0.648/AUC≈0.688; EEGNet acc≈0.773/AUC≈0.836; `B_no_spatial`
ablation acc≈0.825 alone) — those numbers are **not reproduced here**
because they were never saved to a file this report can cite as evidence,
only relayed in chat from a screenshot of a terminal that no longer
exists. Do not use them in a manuscript; re-run and cite the actual
`TABLE_*.csv` files instead.

## 15. Remaining limitations

- Ablation and resolution experiments are single-seed (by design, per
  `EXPERIMENT_DECISION.md` — flagged as provisional, not silently
  presented as equally powered to the 5-repeat primary comparison).
- Explainability currently reports channel/temporal importance from one
  visualization-only model, not mean ± SD across outer folds — sufficient
  for a supplementary figure, not for a claim about attribution stability
  (`EXPERIMENT_DECISION.md`, Optional row).
- True 512 Hz reproduction remains impossible from the released files;
  the interpolation experiment is a sensitivity check only.
- GPU nondeterminism (documented in `README.md`) means exact repeat-level
  numbers may vary by up to ~1 percentage point / 0.014 AUC run to run,
  consistent with what the original manuscript's own Limitations section
  already reported.

## 16. Publication-readiness assessment

**NOT READY — the following specific items remain:**

1. **Run `run_all_experiments.py` once on the real dataset**, on a GPU
   environment where the output will actually be persisted (Colab with
   Google Drive-mounted `ADHD_EEG_RESULTS_DIR`/`ADHD_EEG_TABLES_DIR`/
   `ADHD_EEG_FIGURES_DIR`, or Kaggle with a git-push-per-phase habit — see
   `README.md`). This single step produces `TABLE_1` through `TABLE_7`,
   the confidence intervals, the statistical comparisons, and populates
   `results_final/` automatically; no further code changes are anticipated
   to be necessary for it to succeed, since the underlying design was
   already validated correct on real data once.
2. Once that run completes, regenerate Figures 3–10 from the saved
   `results_final/predictions/all_predictions.csv` (the figure-generation
   functions in `src/visualization.py` are unchanged and already
   validated) and author Figures 1/2 (workflow/architecture diagrams) by
   hand, as previously noted in `original` `FINAL_REPORT.md`.
3. Re-read this document's §6–14 once the real `TABLE_*.csv` files exist,
   and only then write the corresponding manuscript sections — using the
   actual numbers in those files, not the ones relayed in chat.

Nothing on this list is a new experiment, a new model, or a design
change — everything required is now implemented and waiting on one real
execution.
