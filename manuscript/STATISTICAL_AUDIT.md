# Statistical Audit

This document records a full re-audit of the model-comparison statistics used in `manuscript.md` (Table 5, Section 4.6), performed against the raw, epoch-level prediction file `results_final/predictions/all_predictions.csv` (34,544 rows), which was independently verified to be the exact source of `TABLE_2_EPOCH_LEVEL_RESULTS.csv` and `TABLE_3_SUBJECT_LEVEL_RESULTS.csv` (every value re-extracted and matched to within 1e-5).

## 1. The problem found

The manuscript's prior model-comparison procedure (Table 5) used a Wilcoxon signed-rank test on **25 "paired" observations per comparison** — one balanced-accuracy value per (repetition, outer fold), for 5 repetitions × 5 outer folds. This treats those 25 values as 25 independent paired samples.

They are not independent. Every repetition re-partitions the **same 121 subjects** into a new set of 5 outer folds; a subject is never split across folds within one repetition (verified, Section 3 below), but across the 5 repetitions, that same subject reappears in the "held-out" role 5 times, each time contributing correlated information (a subject who is unusually easy or hard to classify pulls in the same direction in every repetition it appears in). Treating those 25 values as independent replicates is a textbook case of pseudo-replication: it does not increase the true amount of independent information beyond n = 121 subjects, but it inflates the apparent sample size used by the significance test, which mechanically shrinks p-values below what the true sampling uncertainty supports.

This is exactly the concern raised for this audit, and it is a legitimate, material problem: it is not a hypothetical risk, it changed which comparisons were reported as statistically significant (Section 4 below).

## 2. What Table 4 does NOT have this problem

Table 4's 95% confidence intervals were already computed by `statistics.bootstrap_subject_level_ci`, which resamples the subject-level table (one row per subject) with replacement — this is a legitimate subject-level bootstrap, not the flawed procedure. Only the paired **hypothesis-test** procedure behind Table 5 (`statistics.compare_all_models` / `paired_model_comparison`, using per-fold values) had the independence problem. This was confirmed by direct code inspection of `src/statistics.py`.

Table 6's paired test (CNN and EEGNet, 128 Hz vs. 128-to-512 Hz interpolation) does **not** have this problem: the interpolation condition was run for a single repetition only, so its 5 outer folds are a single, non-reused subject partition (independently verified: every subject's outer-fold assignment in the interpolation condition is identical to its assignment in repetition 0 of the 128 Hz condition, and no subject appears in more than one of the 5 folds). The original per-fold Wilcoxon procedure was therefore retained for Table 6.

## 3. The corrected procedure

Implemented in `src/subject_level_paired_test.py`, reading directly from `results_final/predictions/all_predictions.csv`.

1. **One paired observation per subject.** For each model, each subject's prediction is the same aggregation already used for Table 3/Table 4: that subject's predicted probability, pooled (mean) across *all* epochs and *all* 5 repetitions, thresholded at 0.5 (`evaluation.aggregate_subject_level`'s convention). This was independently re-derived from the raw file and matched Table 3's 12 model rows exactly (n = 121, accuracy and balanced accuracy to 1e-5 for every model).
2. **Verified pairing.** All 12 models share the identical 121-subject set, in the identical order, with identical `y_true` labels — confirmed programmatically before any comparison was run.
3. **Paired bootstrap.** 20,000 resamples of the 121 subject indices, with replacement; the *same* resampled indices are applied to both models being compared (preserving the pairing), balanced accuracy is recomputed for each model on each resample, and the difference distribution yields: mean/median diff, 95% percentile CI, and a two-sided p-value (2 × min[P(diff≤0), P(diff≥0)], capped at 1 — equivalent to inverting the percentile CI).
4. **Paired permutation test.** 20,000 permutations; independently for each subject, the two models' predictions are swapped with probability 0.5 (testing exchangeability of "which model produced this subject's prediction"), and the two-sided p-value is the proportion of permuted |differences| at least as large as the observed one.
5. **Multiplicity correction.** Both the bootstrap and permutation p-values are Holm–Bonferroni corrected within the 11-comparison family (every other model vs. the plain CNN, at 128 Hz).
6. **Effect size.** Reported as the observed mean difference divided by the bootstrap standard deviation of that difference (a bootstrap-based standardized effect size); this is explicitly *not* labeled Cohen's d_z, since balanced accuracy is not a simple per-subject scalar that decomposes into per-observation paired differences the way Cohen's d_z requires.

Robustness check: the full procedure was re-run with an independent random seed (999 vs. the canonical 12345); the qualitative conclusion (which comparisons survive Holm correction) was identical, and no CI or p-value changed by more than what is expected from Monte Carlo noise at 20,000 resamples.

## 4. What changed as a result

| Comparison | Old (Wilcoxon, n=25, Holm p) | New (bootstrap, n=121, Holm p) | New (permutation, n=121, Holm p) | Conclusion changed? |
|---|---|---|---|---|
| CNN vs. EEGNet | 0.0002 | <0.0001 | 0.0016 | No — significant both ways |
| CNN vs. CNN+LR / CNN+LR_noCORAL | 0.0033 | 0.0110 | 0.0185 | No — significant both ways |
| CNN vs. ShallowConvNet | 0.0050 | 0.0110 | 0.0140 | No — significant both ways |
| CNN vs. CNN+LR_withCORAL | 0.0033 | 0.0133 | 0.0242 | No — significant both ways |
| CNN vs. CNN+KNN | 0.0227 | 0.1284 | 0.3240 | **Yes — was significant, no longer** |
| CNN vs. CNN+NLSVM | 0.0033 | 0.1284 | 0.3240 | **Yes — was significant, no longer** |
| CNN vs. CNN+RF | 0.0031 | 0.2080 | 0.3876 | **Yes — was significant, no longer** |
| CNN vs. CNN+LinearSVM | 0.1585 | 0.3021 | 0.5253 | No — not significant either way |
| CNN vs. DeepConvNet | 1.0000 | 0.9082 | 0.7734 | No — not significant either way |
| CNN vs. CNN+GNB | 0.5588 | 0.9085 | 0.9023 | No — not significant either way |

Three comparisons (CNN+KNN, CNN+NLSVM, CNN+RF) that were reported as statistically significant under the flawed procedure are **not** statistically significant under the corrected, subject-level procedure, by either the bootstrap or the permutation test. This is the expected direction of the correction: pseudo-replication inflates apparent significance, so removing it should — and here does — move some marginal comparisons from "significant" to "not significant," never the reverse. The four comparisons that remain significant (EEGNet, ShallowConvNet, CNN+LR, CNN+LR_withCORAL) do so under both the bootstrap and the permutation test, which gives some additional confidence that these specific four are not an artifact of either test's particular assumptions.

## 5. Manuscript sections updated as a consequence

- Abstract (Results paragraph, Methods paragraph)
- Section 3.17 (Statistical comparisons — full rewrite of the primary-test description)
- Section 4.6 (Table 5 replaced; interpretive prose rewritten)
- Figure 5's caption (reworded to state explicitly that the 25 per-fold points are descriptive, not the basis of inference)
- Section 5.1 (Principal findings — "in most cases, statistically significantly better" walked back to the actual two-of-several pattern)
- Section 5.3 (CNN vs. classical hybrid classifiers — RF reference softened since it no longer reaches significance)
- Section 7 (Conclusion — DeepConvNet's "does not outperform… at all" corrected: its point estimate is numerically higher than the plain CNN's, 0.589 vs. 0.547, just not significantly so)
- Section 6 (Limitations — three new items added: correlation among repeated-CV estimates, limitations of the inferential procedure itself, and the transductive nature of CORAL restated explicitly as a limitation)

## 6. What was *not* changed

- Table 3 and Table 4 (subject-level point estimates and confidence intervals) — unaffected, since their underlying computation was already subject-level.
- Table 6 (resolution sensitivity) — unaffected; its Wilcoxon procedure was independently verified not to have the pseudo-replication problem (Section 2 above).
- Figures 1–4, 6, 7a/7b — no change to the plotted values; only Figure 5's caption changed, to correctly frame what the plotted distribution is and is not evidence for.
- CORAL's own methods description (Section 3.18) — already correctly framed as transductive before this audit pass; this audit added an explicit restatement of that point in the Limitations section (Section 6) as the user's audit brief specifically requested.

## 7. Honesty note

This document reports a genuine finding, not a hypothetical one: an earlier draft of this manuscript did in fact claim statistical significance (Holm-adjusted p < 0.03) for three model comparisons (CNN+KNN, CNN+NLSVM, CNN+RF) that do not survive a defensible subject-level test. That earlier claim is now understood to have been an artifact of treating correlated repeated-CV fold estimates as independent observations. We record this explicitly, per this project's practice of documenting what was wrong and how it was caught, not only the corrected number.
