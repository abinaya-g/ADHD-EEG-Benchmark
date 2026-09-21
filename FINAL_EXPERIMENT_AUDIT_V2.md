# Final Experiment Audit V2

Written against direct inspection of the `results_final` package the user
provided (extracted and read row-by-row, not assumed) and the exact
transcript of their live Colab session's cell 5. Per this project's
standing rule: nothing below is stated as complete unless it was actually
verified from a saved file or a real execution log.

## 1. What was already completed (verified from the provided `results_final` package)

- 121 subjects (61 ADHD / 60 Control), 508 epochs (288 ADHD / 220 Control),
  19 channels, 3840 samples/epoch, assumed 128 Hz — confirmed in
  `tables/TABLE_1_DATASET_SUMMARY.csv`.
- 5 outer folds × 4 inner folds × 5 repetitions, seeds 42–46 — confirmed in
  `configs/run_config.txt`.
- Subject-level predictions, aggregation, and bootstrap CI — confirmed
  present (`subject_level/TABLE_3_SUBJECT_LEVEL_RESULTS.csv`,
  `confidence_intervals/TABLE_4_CONFIDENCE_INTERVALS.csv`, subject-level
  bootstrap per `sanity_checks.py` check 12, which passed).
- All 14 leakage sanity checks passed (`audit/leakage_sanity_checks.csv`,
  every row `True`).
- CNN + LR/RBF-SVM(NLSVM)/RF/GNB/KNN/LinearSVM, EEGNet, ShallowConvNet —
  each confirmed at 2540 epoch-level prediction rows in
  `predictions/all_predictions.csv`, i.e. 508 epochs × 5 repeats, the full
  design.
- CORAL comparison (`CNN+LR_noCORAL` vs `CNN+LR_withCORAL`) — present at
  the full 2540 rows each.
- `TABLE_5_MODEL_COMPARISON_STATISTICS.csv` already correctly pairs on
  `(repeat, outer_fold)` and reports `n_paired_folds=25` for every
  fully-repeated model against the CNN|128Hz reference, Holm-corrected.

**None of the above was rerun.** This session's changes only added new
capability (a way to resume a single incomplete architecture, and EEGNet
in the resolution experiment) — they do not touch or regenerate the
already-valid rows above.

## 2. What this session found and fixed (code, not results)

Two real gaps, found by inspecting the actual data rather than trusting
that "implemented" meant "executed correctly":

- **DeepConvNet was incomplete**: 508 rows (1 repeat × 5 folds) against
  2540 for every other model. Confirmed directly:
  `predictions/all_predictions.csv` groupby(`model`,`preprocessing`) shows
  `DeepConvNet 128Hz 508` versus `2540` for CNN/EEGNet/ShallowConvNet/all
  classifiers. `TABLE_5`'s own `n_paired_folds` column already reflected
  this correctly (5, not 25, for the CNN-vs-DeepConvNet row) — the
  statistics code was never wrong, the underlying data was incomplete.
- **Root cause, from the user's own cell 5 transcript**: a full rerun
  (no `--skip` flags — this redid already-complete nested_cnn/EEGNet/
  ShallowConvNet work, which the top of this task explicitly said not to
  do) reached `DeepConvNet repeat 3 (seed=45) done and saved` and was then
  interrupted (`^C`) before repeat 4 (seed 46) could complete and save.
- **No resolution experiment had been run at all** in the provided
  package — no `128to512_interp` rows exist anywhere in
  `predictions/all_predictions.csv`, and no `TABLE_6` file was in the zip.

Fixed (commit `c352699`):
- `--only-architecture NAME` / `--repeat-indices i,j,k`, so the missing
  DeepConvNet repeat can be completed by appending to the existing
  `predictions_architectures.csv` without touching or re-running EEGNet/
  ShallowConvNet. Verified directly (not just compiled): reproduced the
  exact interrupted state on synthetic data and confirmed EEGNet/
  ShallowConvNet rows come back byte-identical while DeepConvNet gains
  the missing repeat with zero duplicate rows.
- The resolution experiment now runs EEGNet alongside CNN (previously
  CNN-only), per this task's "if computationally feasible, also test
  EEGNet" instruction, both reusing the primary run's first-repeat seed
  so the 128Hz-vs-interpolation comparison is validly paired.
- `TABLE_6`/its figure and the ablation figure were gated on whether
  *this* invocation ran that phase rather than on whether the data
  actually exists on disk — a real bug that would have silently skipped
  rebuilding `TABLE_6` on a combine-only rerun even after the resolution
  phase had completed in an earlier invocation. Fixed to check data
  presence, matching how `TABLE_7`/CORAL already worked.

## 3. Final number of repetitions for every model

**As of the last data this audit could inspect** (the provided zip; the
commands below had not yet been run against real data when this document
was written):

| Model | Repetitions confirmed | Rows |
|---|---|---|
| CNN, CNN+LR, CNN+NLSVM, CNN+RF, CNN+GNB, CNN+KNN, CNN+LinearSVM, CNN+LR_noCORAL, CNN+LR_withCORAL | 5/5 | 2540 each |
| EEGNet | 5/5 | 2540 |
| ShallowConvNet | 5/5 | 2540 |
| DeepConvNet | 1/5 in the zip; 4/5 reached (then interrupted) in the live session per the cell 5 transcript | 508 in the zip |

**This will change** once the resume command (`--only-architecture
DeepConvNet --repeat-indices 4`) actually runs — expected to bring
DeepConvNet to 5/5 (2540 rows), but that execution had not happened as of
this writing. Do not treat this document as confirming it did.

## 4. Does every model have subject-level predictions?

Yes for the 11 models with complete data (see table above) —
`TABLE_3_SUBJECT_LEVEL_RESULTS.csv` in the provided package has rows for
all of them. DeepConvNet has subject-level predictions too, but computed
from only 1 repeat's worth of epochs per subject rather than 5 — valid as
far as it goes, but based on less data than the other models until the
resume completes.

## 5. Does every model have subject-level bootstrap CI?

Yes, structurally — `TABLE_4_CONFIDENCE_INTERVALS.csv` in the provided
package has rows for every model present in the predictions file,
including DeepConvNet. DeepConvNet's CI is simply wider/less precise than
it will be once all 5 repeats are pooled into the subject-level
aggregation, because it currently reflects only 1 repeat's epochs.

## 6. Was the 128→512 sensitivity analysis completed?

**No, not yet.** Confirmed absent from the provided package (no
`128to512_interp` rows in `all_predictions.csv`, no `TABLE_6` file). The
code to run it (now including EEGNet) is implemented and was validated on
synthetic data this session (both CNN and EEGNet appearing correctly at
`128to512_interp`, `TABLE_6` and its paired test built for both). Running
it for real is the command given in this session's reply, not yet
executed against real data.

## 7. Does DeepConvNet now have 5 repetitions?

**Not as of this document.** It reached 4/5 in the live session before
being interrupted; the code to complete the 5th without disturbing
anything else is implemented and verified (§2); executing it against the
real dataset is the pending step.

## 8. Was any leakage detected?

No. All 14 automated sanity checks passed in the provided package (every
row of `audit/leakage_sanity_checks.csv` is `True`), and the two new code
paths added this session (resumable architecture completion, EEGNet in
the resolution experiment) reuse the same `evaluation.run_nested_repeat`
leakage-controlled machinery already audited in `AUDIT_REPORT.md` —
they do not introduce a new training/evaluation path, only new ways to
select which (architecture, repeat) combinations to run and combine.

## 9. Is any remaining experiment scientifically necessary?

No new experiment. The only remaining necessary work is **executing** the
two already-implemented, already-validated pieces of code against the
real dataset:

1. Complete DeepConvNet's 5th repetition (`--only-architecture
   DeepConvNet --repeat-indices 4`).
2. Run the 128→512 interpolation sensitivity experiment, now including
   EEGNet (`--skip-nested-cnn --skip-architectures --skip-ablation`).

Per this task's explicit instructions, and consistent with
`EXPERIMENT_DECISION.md`'s prior classification: no new architectures, no
additional classifiers, no hyperparameter search, no additional random
splits, and no additional CORAL variants are needed or were added.

## Note on Task 3 (statistical analysis unit)

Confirmed directly from `TABLE_5_MODEL_COMPARISON_STATISTICS.csv`: every
fully-repeated model comparison already carries `n_paired_folds=25`,
correctly meaning 5 repetitions × 5 outer folds, not 25 independent
folds — the `n_paired_folds` column exists specifically so this is never
ambiguous in downstream manuscript text. Holm correction is applied
(`p_value_holm` column) across the full comparison set against the CNN
reference. Once DeepConvNet reaches 5/5 repetitions, its comparison row
will automatically read `n_paired_folds=25` too (it currently reads 5,
correctly reflecting its 1-repeat data) — no statistics code needs to
change for this; it will simply reflect more paired folds once more data
exists, as designed.

## Readiness after the two pending executions

Once both commands in this session's reply have been run for real and the
resulting `TABLE_1` through `TABLE_7` regenerated:

**READY FOR MANUSCRIPT WRITING**, on the condition that the resulting
`TABLE_5` shows `n_paired_folds=25` for the DeepConvNet comparison (not
5), and `TABLE_6` contains both CNN and EEGNet rows under
`128to512_interp`. If either condition doesn't hold after running the
commands, that specific gap — not a new experiment — is what remains.
