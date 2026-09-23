# Final Experiment Audit V2

Written against direct inspection of the user's live Colab notebook
transcript (`ADHD_new.ipynb`, cells 0–9 as executed). Per this project's
standing rule: nothing below is stated as complete unless it was actually
verified from a printed execution log or a saved file — not inferred from
what the code is supposed to do.

## 1. What was already completed (verified)

- 121 subjects (61 ADHD / 60 Control), 508 epochs (288 ADHD / 220 Control),
  19 channels, 3840 samples/epoch, assumed 128 Hz — confirmed in cell 4's
  `Loaded X=(508, 19, 3840, 1) ... unique_subjects=121` line.
- 5 outer folds × 4 inner folds × 5 repetitions, seeds [42, 43, 44, 45, 46]
  — confirmed in every cell's own printed config line.
- All 14 leakage sanity checks passed, in every one of cells 4, 8, and the
  start of cell 9 (each prints `Sanity check report: 14/14 passed` with
  every individual check listed `PASS`).
- CNN + LR/RBF-SVM(NLSVM)/RF/GNB/KNN/LinearSVM, EEGNet, ShallowConvNet,
  CORAL (`CNN+LR_noCORAL` vs `CNN+LR_withCORAL`) — each confirmed at 2540
  epoch-level prediction rows (508 epochs × 5 repeats) in cell 8's
  "Building tables" printout.
- **DeepConvNet is now complete at 5/5 repetitions (2540 rows)** — see §2.

## 2. DeepConvNet completion (Task 1) — verified complete

Cell 7 (run before the resume, i.e. the actual pre-fix state) printed:

```
DeepConvNet     0    508
                1    508
                2    508
                3    508
EEGNet          0-4  508 each (2540 total)
ShallowConvNet  0-4  508 each (2540 total)
```

DeepConvNet had repeats 0–3 only (2032 rows) — EEGNet and ShallowConvNet
were already complete at 5/5 and untouched.

Cell 8 ran the resume command
(`--only-architecture DeepConvNet --repeat-indices 4`) and printed:
```
RESUMING (appending to existing file): architecture=DeepConvNet, repeat_indices=[4]
DeepConvNet repeat 4 (seed=46) done and saved
```
followed by all 14 sanity checks passing again, and the rebuilt table
showing `DeepConvNet 128Hz 2540`. Arithmetic check: 2032 (repeats 0–3) +
508 (repeat 4) = 2540, exactly matching the printed post-resume total —
independent confirmation that repeat 4 was genuinely added, not that the
file was silently regenerated some other way.

| model | available repeats | prediction rows | complete 5-repetition CV? |
|---|---|---|---|
| CNN, CNN+LR, CNN+NLSVM, CNN+RF, CNN+GNB, CNN+KNN, CNN+LinearSVM, CNN+LR_noCORAL, CNN+LR_withCORAL | 0,1,2,3,4 | 2540 each | Yes |
| EEGNet | 0,1,2,3,4 | 2540 | Yes (already complete pre-fix, untouched by the resume) |
| ShallowConvNet | 0,1,2,3,4 | 2540 | Yes (already complete pre-fix, untouched by the resume) |
| DeepConvNet | 0,1,2,3,4 | 2540 | **Yes — completed in this session via the resume command** |

No architecture other than DeepConvNet was retrained. EEGNet and
ShallowConvNet's row counts and (as far as the printed table's accuracy
figures show, e.g. `EEGNet 0.772835` identical to the pre-resume table)
their values are unchanged between cell 4/7 and cell 8 — consistent with
the append-only design, not a full rerun.

## 3. Resolution experiment (Task 2) — NOT complete

Cell 9 (`--skip-nested-cnn --skip-architectures --skip-ablation`, i.e.
only the resolution phase) printed only:
```
--- 128Hz vs 128->512Hz-interpolation sensitivity experiment ---
    NOTE: pipeline B is FFT interpolation of the same 128Hz samples,
    NOT the original 512Hz acquisition. See AUDIT_REPORT.md Phase 3.
```
and nothing further — no `"CNN resolution experiment done and saved"`,
no `"EEGNet resolution experiment done and saved"`, no table-build
output. The cell was still running (or had not yet produced further
output) at the moment this notebook was exported. **There is no evidence
`TABLE_6_128HZ_VS_128TO512.csv` exists yet.** This audit does not claim
it does.

Once cell 9 finishes, it should print, in order: `CNN resolution
experiment done and saved`, `EEGNet resolution experiment done and
saved`, then `TABLE_6: models=['CNN', 'EEGNet'], N summary rows, M
paired-test rows` during table building. Confirm those lines (or the two
files directly, per the check given in the chat reply) before treating
Task 2 as done.

## 4. Subject-level results

Confirmed present for every model with complete data (all 12 models in
the §2 table) via `TABLE_3_SUBJECT_LEVEL_RESULTS.csv`, built automatically
at the end of cell 8's run — regenerated fresh from the now-complete
`predictions_architectures.csv`, so DeepConvNet's subject-level rows now
reflect the full 5 repeats' worth of epochs per subject, not the partial
1-repeat data an earlier version of this audit had to caveat.

Not yet available for the `128to512_interp` condition (CNN/EEGNet at
128→512 resolution) — depends on Task 2/cell 9 completing.

## 5. Confidence intervals

`TABLE_4_CONFIDENCE_INTERVALS.csv` was rebuilt in cell 8 (`TABLE_4: 96
(group x metric) rows, subject-level bootstrap, n_boot=2000`) — this
count is unchanged from cell 4's original 96, which is expected: the
number of (model, preprocessing, metric) groups didn't change, only
DeepConvNet's underlying data within its existing group got more
complete. DeepConvNet's CI is now computed from the full 5-repeat subject
aggregation rather than 1 repeat's worth.

No CI yet for `128to512_interp` — pending Task 2.

## 6. Any remaining methodological issue?

None found. All 14 sanity checks continue to pass after the DeepConvNet
resume, confirming the append-only resume did not introduce any
train/test leakage or partition inconsistency. `TABLE_5`'s paired
comparisons use `n_paired_folds` computed from actual shared
`(repeat, outer_fold)` keys between two groups (see
`src/reporting.py:table5_model_comparison`) — now that DeepConvNet has
all 5 repeats on the same seeds/partitions as every other model in the
primary run, its comparison row should read `n_paired_folds=25` rather
than the `5` it read before the resume; this has not been directly
re-inspected in the CSV in this session (only the printed accuracy table
was visible in the transcript), so treat it as expected-but-not-yet-
independently-confirmed until you check `TABLE_5_MODEL_COMPARISON_STATISTICS.csv`
directly:
```python
import pandas as pd, os
t5 = pd.read_csv(os.path.join(os.environ['ADHD_EEG_TABLES_DIR'], 'TABLE_5_MODEL_COMPARISON_STATISTICS.csv'))
print(t5[t5['model_b'].str.contains('DeepConvNet')][['model_a','model_b','n_paired_folds']])
```

## 7. Is the experiment set now frozen?

**Not yet.** DeepConvNet's completion (Task 1) is done and frozen — no
further action needed there. The set as a whole cannot be declared frozen
until Task 2 (resolution experiment) actually finishes and its outputs
are confirmed to exist, since "frozen" implies every planned table is
populated, and `TABLE_6` is not yet confirmed to exist.

No new classifiers, architectures, additional random splits, additional
CORAL variants, or hyperparameter searches were added or are needed, per
Task 3 — nothing in this session's work touched that boundary.

---

## Final response

**NOT READY — the 128→512 Hz interpolation/resampling sensitivity
experiment (Task 2) has not finished executing.** Cell 9's output ends
immediately after the phase banner, with no confirmation that CNN or
EEGNet completed at the `128to512_interp` condition, and no evidence that
`TABLE_6_128HZ_VS_128TO512.csv` exists. Everything else audited in this
document (DeepConvNet's 5-repetition completion, all 14 leakage checks,
subject-level results and CIs for the 12 complete models) is verified and
requires no further action. Once cell 9 finishes and the existence of
`TABLE_6_128HZ_VS_128TO512.csv` /
`TABLE_6_128HZ_VS_128TO512_paired_test.csv` is confirmed (and, ideally,
`TABLE_5`'s DeepConvNet row is confirmed to read `n_paired_folds=25`),
this experiment set is ready for manuscript writing with no remaining
gaps identified.
