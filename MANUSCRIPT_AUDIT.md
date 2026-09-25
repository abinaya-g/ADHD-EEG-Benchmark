# Manuscript Audit

Performed before writing any manuscript text, per the standing project rule:
nothing is stated as a "final result" unless it was verified from an actual
output file or a printed execution log from a real (non-synthetic) run — not
inferred from what the code is supposed to do, and not copied from the
desk-rejected manuscript.

## 1. Repository state

- Repository: `https://github.com/abinaya-g/ADHD-EEG-Benchmark`
- Branch: `claude/adhd-eeg-manuscript-revision-boeb24`
- Current commit: `4a1006ae429ca5d3ce5f61faf16395689904bcf3` (2026-09-25)
- `git status`: clean (no uncommitted changes) at time of audit
- Note: this local checkout only contains **code and documentation**. The
  repository does **not** contain the actual experiment output files
  (predictions, tables, figures) — `results/`, `tables/`, and `figures/` in
  the repo hold only `.gitkeep` placeholders and synthetic smoke-test CSVs
  (`SYNTHETIC_predictions.csv`, `SYNTHETIC_coral.csv`). All real numerical
  results exist only on the user's Google Drive (produced by running
  `run_all_experiments.py` on Kaggle/Colab against the actual dataset) and
  were shared with me piecemeal as individual CSV uploads during this
  conversation, not as a complete result set. **This is the single biggest
  constraint on what this manuscript can currently contain — see §8.**

## 2. Repository structure (as of this commit)

```
run_all_experiments.py          orchestration script (CLI, phase checkpointing, table/figure generation)
src/
  config.py                     all hyperparameters, seeds, CV design, paths
  data.py                       dataset loading, per-epoch normalization, 128->512Hz interpolation
  models.py                     CNN, EEGNet, ShallowConvNet, DeepConvNet, classical classifiers
  evaluation.py                 nested CV driver, subject-level aggregation, metrics
  coral.py                      CORAL domain adaptation
  statistics.py                 bootstrap CI, paired Wilcoxon tests, Holm correction
  reporting.py                  TABLE_1/4/5/6 builders
  sanity_checks.py              14 automated leakage/structural checks
  visualization.py              figure generators
  explainability.py             Integrated Gradients
tests/test_synthetic_pipeline.py   end-to-end self-test on synthetic surrogate data (not real results)
AUDIT_REPORT.md, FINAL_REPORT.md, EXPERIMENT_DECISION.md,
FINAL_EXPERIMENT_REPORT.md, FINAL_EXPERIMENT_AUDIT_V2.md   prior audit trail (this session's predecessor documents)
original_notebooks/manuscript_desk_rejected.pdf   the rejected manuscript, for context only
```

## 3. Keyword search across the repository (.py/.md/.ipynb)

| Keyword | Files matched |
|---|---|
| GroupKFold | 10 |
| StratifiedGroupKFold | **0 — not used anywhere** |
| nested | 12 |
| subject_id | 12 |
| outer fold | 11 |
| inner fold | 5 |
| early stopping | 8 |
| bootstrap | 13 |
| confidence interval | 7 |
| CORAL | 15 |
| 128 Hz | 11 |
| 512 Hz | 9 |
| interpolation | 13 |
| resampling | 5 |
| leakage | 14 |
| seed | 21 |
| Integrated Gradients | 9 |

Important correction for the Methods section: the codebase does **not** use
scikit-learn's `StratifiedGroupKFold`. The outer/inner splits are produced by
a custom function, `evaluation.make_grouped_shuffled_folds`, which shuffles
the unique subject list with a given seed and distributes subjects
round-robin by class label to keep folds approximately class-balanced, then
returns `GroupKFold`-style `(train_idx, test_idx)` pairs. This must be
described precisely as "a class-balance-aware, seed-shuffled adaptation of
group k-fold splitting," not as `StratifiedGroupKFold`.

## 4. Dataset — verified

From an actual `Loaded X=... ADHD_epochs=... Control_epochs=...` line
printed by a real (non-smoke-test) run of `run_all_experiments.py`:

```
Loaded X=(508, 19, 3840, 1) y=(508,) unique_subjects=121 ADHD_epochs=288 Control_epochs=220
```

This **matches** the values used in the desk-rejected manuscript: 121
subjects (61 ADHD / 60 Control by `src/config.py:DATA_FOLDERS` folder
labeling), 508 epochs (288 ADHD / 220 Control), 19 channels, 3840
samples/epoch (30 s at an assumed 128 Hz). The sampling rate is explicitly
an **assumption**, not verified metadata — `src/data.py` and `README.md`
both state the released `.mat` files carry no sampling-rate field; 128 Hz is
inferred from file duration and channel count matching the original
notebooks' own assumption. This must be stated as an assumption in the new
manuscript exactly as it was (correctly) caveated in the old one — this is
not a new finding, but it is not resolved either, and must not be
silently upgraded to a verified fact.

## 5. Validation design — verified

From `src/config.py`: `OUTER_FOLDS=5`, `INNER_FOLDS=4`, `REPEATS=5`,
`RANDOM_SEEDS=[42,43,44,45,46]`. Confirmed by every run's own printed config
line (`outer_folds=5 inner_folds=4 repeats=5 seeds=[42, 43, 44, 45, 46]`).

Pipeline, verified against `src/evaluation.py:run_nested_repeat` and
`select_n_epochs_via_inner_cv`:

```
Outer-train subjects (GroupKFold-style split, class-balanced, seeded)
        -> Inner GroupKFold (4 folds) on outer-train subjects ONLY
        -> Per inner fold: train with early stopping (patience=8) on inner-val subjects
        -> Take the MEDIAN selected epoch count across the 4 inner folds
        -> Refit ONE model on ALL outer-train subjects for exactly that many epochs
           (no validation_data argument at all in this final fit)
        -> Predict once on the outer-test subjects (never seen by this model before)
        -> Epoch-level predictions -> subject-level aggregation -> subject-level metrics
```

Verified leakage-prevention claims (all checked against actual code, not
assumed from GroupKFold's presence alone):

| Requirement | Verified how |
|---|---|
| No subject in both outer-train and outer-test | `sanity_checks.py` check 2, runtime-verified on every real run (`PASS`) |
| Inner validation drawn only from outer-train subjects | Structural: `select_n_epochs_via_inner_cv` receives only `X_tr/y_tr/groups_tr` |
| Outer-test never used for early stopping | Structural: final `model.fit()` call has no `validation_data` argument |
| Outer-test labels never used for model selection | Structural: `y[outer_test_idx]` only referenced after `.fit()` returns |
| Preprocessing not fit on outer-test | `data.normalize_epoch` is per-epoch, per-channel — no cross-epoch statistic exists to leak |
| CNN-derived features for classical classifiers don't leak | Structural: `clf.fit(train_feats, y_tr)` always precedes `clf.predict(test_feats)` |
| Subject IDs attached to every prediction | `sanity_checks.py` check 9, runtime-verified |

All 14 sanity checks passed (`14/14 passed`) on the final combined run that
produced the results in §7/§8 below — this was confirmed from the actual
console log of that run, not assumed.

## 6. Subject-level evaluation — verified

`src/evaluation.py:aggregate_subject_level` computes, per subject: mean
predicted probability across that subject's epochs (`mean_proba`), and a
majority-vote alternative (`vote_fraction`). The **primary** subject-level
label is `pred_mean_proba = (mean_proba >= 0.5)`. Metrics (accuracy,
balanced accuracy, sensitivity, specificity, precision, F1, MCC, AUC) are
then computed once per subject via `evaluation.compute_metrics`, called on
the aggregated subject-level table — never on pooled epochs for the primary
reported numbers. Epoch-level pooled metrics are computed separately (in
`run_all_experiments.py`'s "Building tables" step) and must be labeled as a
secondary/diagnostic quantity, not the headline result.

## 7. Confidence intervals and statistical testing — verified

- Bootstrap: `statistics.bootstrap_subject_level_ci` resamples rows of the
  **subject-level** table (one row per subject) with replacement,
  `n_boot=2000` (from `config.N_BOOTSTRAP`), 95% CI via the percentile
  method (`config.CI_ALPHA=0.05`). Never resamples epochs.
- Paired comparisons: `statistics.paired_model_comparison` — Wilcoxon
  signed-rank test on paired per-(repeat, outer_fold) **subject-level**
  metric values, with `n_paired_folds` explicitly counted as the number of
  shared `(repeat, outer_fold)` keys between the two groups being compared.
  Holm-Bonferroni step-down correction (`statistics.holm_correction`)
  applied within each family of comparisons.
- **Correct statistical-unit description for the write-up**: for two models
  that both ran the full 5-repeat × 5-outer-fold primary design, a
  comparison has `n_paired_folds=25` — verified directly in
  `TABLE_5_MODEL_COMPARISON_STATISTICS.csv` (e.g. `CNN|128Hz` vs
  `DeepConvNet|128Hz`, `n_paired_folds=25`). These 25 values are **paired
  outer-fold performance estimates across 5 repetitions**, not 25
  independent subject cohorts — the same 121 subjects are re-partitioned
  five times with different seeds. For a comparison involving the
  resolution-sensitivity condition (`128to512_interp`, which only has
  repeat 0), `n_paired_folds=5` — verified in
  `TABLE_6_128HZ_VS_128TO512_paired_test.csv`. These two conditions must
  never be conflated in the manuscript text.

## 8. What has and has not been directly verified from an actual result file

This is the critical gate for what can go into Results/Tables/Figures
without fabrication.

**Directly inspected, real files, from the final combined run (34,544
prediction rows, all 14 sanity checks passing):**

| File | Rows | Status |
|---|---|---|
| `TABLE_5_MODEL_COMPARISON_STATISTICS.csv` | 19 | Inspected directly |
| `TABLE_6_128HZ_VS_128TO512.csv` | 4 | Inspected directly |
| `TABLE_6_128HZ_VS_128TO512_paired_test.csv` | 2 | Inspected directly |
| Console-printed pooled epoch-level table (20 model×preprocessing rows, from the "Building tables" step) | 20 | Printed log, real run — usable as the epoch-level secondary table, but not a substitute for the actual saved `TABLE_2`/`TABLE_3` CSVs |

**NOT yet provided for this final combined run, and therefore NOT used in
any Results text, table, or figure below:**

- `TABLE_1_DATASET_SUMMARY.csv` (final run) — dataset/CV-config summary row
- `TABLE_2_EPOCH_LEVEL_RESULTS.csv` (final run) — the saved epoch-level table (the console printout above covers the same numbers but I have not seen the file itself)
- `TABLE_3_SUBJECT_LEVEL_RESULTS.csv` (final run) — **the primary results table for all 12 primary-run models**; this is essential and currently missing
- `TABLE_4_CONFIDENCE_INTERVALS.csv` (final run, reported as 160 rows in the console log but not inspected directly)
- The generated figures (`fig3`–`fig10`) from the final run
- `table_explainability_channel_importance.csv` / `table_explainability_temporal_importance.csv` from the final run
- The exact Python/TensorFlow/scikit-learn/NumPy/SciPy versions actually used in the Colab session that produced these results (`requirements.txt` pins `tensorflow==2.21.0`, but `README.md` explicitly instructs **not** installing this file on Kaggle/Colab and using the platform's preinstalled TensorFlow instead — so the pinned version cannot be reported as the version actually used without confirming it, e.g. via a printed `tf.__version__`)
- The current state of `ADHD_new.ipynb` itself (only earlier, partial snapshots of this notebook were inspected in prior turns of this session, not re-verified against the final combined run)
- An authoritative citation for the original dataset paper (Kaggle dataset `abinayajone/adhd-eeg-dataset`) — I do not have a verified citation for this in hand and will not fabricate one

## 9. Models — completeness audit

| Model | Repetitions confirmed | Outer folds | Predictions (128Hz) | Source |
|---|---|---|---|---|
| CNN | 5/5 | 5 | 2540 | Final run console table |
| CNN+LR | 5/5 | 5 | 2540 | Final run console table |
| CNN+NLSVM | 5/5 | 5 | 2540 | Final run console table |
| CNN+RF | 5/5 | 5 | 2540 | Final run console table |
| CNN+GNB | 5/5 | 5 | 2540 | Final run console table |
| CNN+KNN | 5/5 | 5 | 2540 | Final run console table |
| CNN+LinearSVM | 5/5 | 5 | 2540 | Final run console table |
| CNN+LR_noCORAL | 5/5 | 5 | 2540 | Final run console table |
| CNN+LR_withCORAL | 5/5 | 5 | 2540 | Final run console table |
| EEGNet | 5/5 | 5 | 2540 | Final run console table |
| ShallowConvNet | 5/5 | 5 | 2540 | Final run console table |
| DeepConvNet | 5/5 | 5 | 2540 | Final run console table (fixed via `--only-architecture DeepConvNet --repeat-indices 4` resume this session, after an earlier session's interrupted run left it at 4/5) |

All 12 primary-run models are complete at 5/5 repetitions in the final
combined result set. This is genuinely verified (not assumed) via the `n`
column of the console-printed pooled table, which shows `2540` for every
one of these rows.

Resolution-sensitivity condition (`128to512_interp`, repeat 0 only, by
design — not a completeness gap): CNN and EEGNet only, 508 predictions
each, matching the single-repeat design stated in `run_all_experiments.py`.

## 10. CORAL — verified

`src/coral.py:coral_transform` whitens source (outer-train) features by the
inverse square root of their own covariance, then re-colors with the square
root of the **target** (outer-test) feature covariance — computed from
outer-test features only, never outer-test labels. `evaluation.py`'s CORAL
block confirms this: `dist_before`/`dist_after` (covariance distances) are
computed and saved, and the downstream classifier (`CNN+LR_withCORAL`) is
still trained only on `y_tr` (outer-train labels). This is a legitimate
transductive/unsupervised domain-adaptation setting and must be labeled as
such, not as standard inductive subject-independent evaluation. Per the
module's own docstring, the CORAL mechanics were reused from the original
notebook and were **not** independently re-verified against Sun & Saenko
(2016) directly — flag this in Methods rather than asserting an exact
reproduction.

Result available: `CNN+LR_noCORAL` (0.628 epoch-level pooled accuracy) vs.
`CNN+LR_withCORAL` (0.618) — CORAL did not improve pooled accuracy in this
run. No direct paired significance test between these two specific groups
exists in `TABLE_5` (its reference row is `CNN|128Hz`, not
`CNN+LR_noCORAL|128Hz`) — this must be reported as an observed difference,
not a statistically tested one, unless a supplementary comparison is run.

## 11. Resolution-sensitivity experiment — verified complete

Confirmed via direct inspection (`TABLE_6_128HZ_VS_128TO512.csv` and its
paired-test file) — see `FINAL_EXPERIMENT_AUDIT_V2.md` §3 for the full
verified numbers. Both CNN and EEGNet show no statistically significant
difference between native 128 Hz and FFT-interpolated 128→512 Hz
representations (`n_paired_folds=5` for each, `p=0.50` and `p=0.19`
respectively). Correctly and consistently labeled throughout the codebase
as "128→512 Hz interpolation/resampling," never as a reconstruction of a
genuine 512 Hz acquisition.

## 12. Architecture-fidelity caveat (important for Methods honesty)

`src/models.py`'s own docstring states EEGNet, ShallowConvNet, and
DeepConvNet are "transparent re-implementations... written here because the
`braindecode`/EEGModels package availability could not be verified in
advance," implemented "from the architecture descriptions as I recall them,
not from a fresh read of the papers." **This means the manuscript must not
claim these are verified exact reproductions of Lawhern et al. (2018) /
Schirrmeister et al. (2017)** — they should be described as
"reimplementations following the published architecture descriptions,"
with layer widths/kernel sizes as actually coded (verified directly from
`build_eegnet`/`build_shallow_convnet`/`build_deep_convnet`), and this
caveat carried into the Limitations section.

## 13. Verdict of this audit

**NOT YET READY to write a complete, fully evidence-traced manuscript.**
The methodology sections (dataset, CV design, leakage controls,
statistics, CORAL, resolution experiment) are fully verifiable now from
code plus the files already inspected. The Results section, however,
cannot yet be written in full: the primary subject-level results table
(`TABLE_3`, all 12 models) and the confidence-interval table (`TABLE_4`,
160 rows) for the final combined run have not been provided as files, only
described by a partial console printout (epoch-level only) and a row
count. Writing subject-level numbers for all 12 models from memory of an
earlier, different (partial, pre-DeepConvNet-fix) snapshot would violate
the explicit instruction not to reuse old/incomplete numbers as final ones.

**Requested next step**: please provide, from the same final run that
produced `TABLE_5`/`TABLE_6` (34,544-row combined predictions):
1. `TABLE_1_DATASET_SUMMARY.csv`
2. `TABLE_3_SUBJECT_LEVEL_RESULTS.csv`
3. `TABLE_4_CONFIDENCE_INTERVALS.csv`
4. The figures directory (`fig3`–`fig10` PNGs), if figures are wanted in the manuscript
5. If available: the exact `tf.__version__`/`sklearn.__version__` printed in that Colab session, for the Reproducibility section

Once these are in hand, the manuscript, tables, figures, and the
remaining audit files (`NUMERICAL_CONSISTENCY_REPORT.md`,
`CLAIM_EVIDENCE_AUDIT.md`, `REPRODUCIBILITY.md`) can be completed against
real, complete, final-run data rather than partial figures.

---

## Addendum: full scientific re-audit (post-manuscript)

This section records a later, separate audit pass performed after the
manuscript above was written and the "READY FOR MANUSCRIPT WRITING" state
referenced elsewhere in this repository's audit trail (`FINAL_EXPERIMENT_AUDIT_V2.md`)
was reached. It does not replace the record above, which remains an accurate
account of the pre-writing state.

**What was found.** The raw prediction-level file this manuscript's tables
were built from was not present in this git repository at the start of this
pass (only the pre-aggregated `manuscript/TABLES/*.csv` summary tables were
tracked) — the same gap flagged in the "Repository state" section above had
not, in fact, been fully closed. The file was located, verified byte-for-byte
against the manuscript's own stated row count (34,544 rows) and independently
re-verified to reproduce `TABLE_2` and `TABLE_3` to within 1e-5, then
committed to this repository at `results_final/predictions/all_predictions.csv`
so this gap is now closed and every table in this manuscript is regenerable
from a file actually present in version control.

**What was wrong and fixed.** The model-comparison procedure behind Table 5
used a Wilcoxon signed-rank test on 25 per-(repetition, outer-fold) values
that reuse the same 121 subjects across repetitions — a pseudo-replicated
test. Rebuilding the comparison as a genuine subject-level (n=121) paired
bootstrap and permutation test changed the conclusion for three comparisons
(CNN+KNN, CNN+NLSVM, CNN+RF: previously reported as statistically
significant, no longer significant under the corrected test). Full detail,
including the corrected Table 5 and every affected manuscript section, is in
`manuscript/STATISTICAL_AUDIT.md`.

**What was independently re-verified and found correct, requiring no
change**: subject-level aggregation (Table 3/4's methodology), DeepConvNet's
repetition completeness (5 repetitions × 5 outer folds, fully present, no
subject-fold leakage, contrary to this audit's initial concern that an
earlier archive might have had fewer DeepConvNet repetitions), Table 6's
resolution-sensitivity paired test (not pseudo-replicated, since its 5 outer
folds come from a single, non-reused repetition), and the CORAL section's
transductive framing (already correct; restated explicitly as a Limitations
item at this pass's request).

**Current verdict: the manuscript is scientifically corrected but not yet
verified compiled.** The `.tex` output has been checked structurally (brace
balance, environment matching, no stray placeholders) but this environment
has no LaTeX toolchain to compile it; the `.docx` has been checked to open
correctly and embed all figures. See `manuscript/STATISTICAL_AUDIT.md` and
`manuscript/NUMERICAL_CONSISTENCY_REPORT.md` for the full itemized record of
this pass.
