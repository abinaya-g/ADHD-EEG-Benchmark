# Experiment Decision Record (Phase 13)

Classification of every candidate experiment against the review brief's
"preferred final experimental package." The rule applied throughout: an
experiment that is already correctly implemented and was already validly
executed is **not** rerun or expanded merely because it would be
interesting to — see `AUDIT_REPORT.md` for what "validly executed" means
for each row (in most cases: yes, once, on real data, but the output was
lost to Kaggle's ephemeral storage and needs one re-execution — that is a
persistence problem, not grounds for redesigning the experiment).

## A. REQUIRED

These support the manuscript's central claim (robust, leakage-free,
subject-independent evaluation) and are all already implemented and
validated; only re-execution + persistence remains.

| Experiment | Status |
|---|---|
| Nested subject-independent GroupKFold CV (5 outer × 4 inner) | Implemented, validated (14/14 sanity checks, real data). Re-execute once. |
| Repeated evaluation with 5 independently controlled seeds | Implemented, validated. Re-execute once (same run as above). |
| Subject-level performance (mean-probability primary, majority-vote secondary) | Implemented, validated. Re-execute once (same run). |
| Subject-level 95% bootstrap CI | Implemented **this session** (was a real gap — see AUDIT_REPORT.md). Needs one validation run (in progress) + the real-data re-execution. |
| 128 Hz baseline (CNN + LR/NLSVM/RF/GNB/KNN/LinearSVM) | Implemented, validated. Re-execute once. |
| Leakage sanity checks (14 automated checks) | Implemented, validated on both synthetic and real data. Runs automatically every time; nothing further needed. |

## B. RECOMMENDED

Scientifically well-motivated, already implemented and validated, no
redesign needed — keep in the final package, re-execute alongside the
REQUIRED set (same run, negligible marginal cost since they share the
nested-CV infrastructure).

| Experiment | Status |
|---|---|
| EEGNet / ShallowConvNet / DeepConvNet deep-learning baselines | Already implemented and executed on real data before this session (per the brief: "if already includes them, retain"). Answers "is poor generalization specific to this CNN, or does it persist across established architectures?" — real run showed it does NOT persist uniformly (EEGNet clearly outperformed the custom CNN), a genuinely informative result. Retained, not expanded. |
| CORAL (transductive, explicitly labelled) | Already implemented, correctly scoped (no test labels, unsupervised covariance alignment only), already executed once. Retained as the single controlled domain-adaptation sensitivity check the brief asks for — not expanded into multiple CORAL variants. |
| 128→512 Hz interpolation sensitivity experiment | Already implemented, correctly labelled as interpolation (never "512 Hz data"), executed once on real data at a single repeat (deliberately reusing the primary run's seed for a valid paired comparison). This session added the missing `TABLE_6` + paired statistical test that made this comparison actually reportable. |
| Paired statistical model comparison (Holm-corrected, rank-biserial effect size, subject-level per-fold pairing) | Implemented **this session** (was a real gap — the only prior comparison logic used epoch-level pooling and was never automated). |
| CNN ablation: `A_full` (baseline), `B_no_spatial` (effect of spatial conv), `C_no_temporal` (effect of temporal conv) | These three directly answer the brief's named questions ("effect of temporal convolution," "effect of architectural components," and — via CNN-alone vs CNN+classifier comparison already in the REQUIRED set — "effect of the learned representation"). Already implemented and executed once (real run showed `B_no_spatial` dramatically outperforming the full architecture — a genuinely notable, reportable finding, though single-seed and provisional). |

## C. OPTIONAL

Already implemented (so removing them would waste existing, working code
and a previously-successful real execution), but not central to the
manuscript's claims. Keep the code and the already-produced numbers if the
re-execution happens to include them (they're cheap relative to the
REQUIRED/RECOMMENDED set); do not treat their absence as blocking
"READY FOR MANUSCRIPT WRITING."

| Experiment | Status |
|---|---|
| CNN ablation: `G_no_batchnorm`, `H_no_pooling` | Implemented and executed once; answer secondary architectural questions (BN, pooling) not explicitly requested by the brief's named list. Genuinely cheap to keep (same phase, same script) but not required for the central claim. |
| Integrated Gradients explainability | Implemented **this session's wiring** (was a dead CLI flag before). The brief explicitly calls explainability "secondary to the main validation study" and warns against strong claims from attribution maps. Current implementation trains one visualization-only model on the full dataset (mirroring the ORIGINAL notebook's own precedent for filter visualization) — it does not yet compute per-fold attribution stability (mean ± SD across outer folds), which would be needed for the brief's stronger "if unstable across folds, report this" instruction. Sufficient for a supplementary figure; not sufficient for any claim about attribution *stability*. |
| Majority-vote subject-level aggregation | Implemented and saved alongside mean-probability in every subject-level table, for internal comparison. The brief specifies mean-probability as primary; majority-vote is reported as a secondary check, not a separate required analysis. |

## D. UNNECESSARY (explicitly not done, on purpose)

| Candidate | Why it's excluded |
|---|---|
| True 512 Hz reproduction of the original acquisition | Impossible from the released files — no sampling-rate metadata exists, and every file's sample count is an exact multiple of 128 Hz × 30 s, not 512 Hz × 30 s (Appendix A5). Claiming otherwise would be fabrication. The interpolation sensitivity experiment (in RECOMMENDED) is the correct substitute, explicitly labelled as such. |
| Additional CORAL variants (e.g. Deep CORAL, multiple alignment strategies) | The brief explicitly asks for "a controlled domain-adaptation sensitivity experiment, not a large domain-adaptation study." One correctly-scoped CORAL comparison already exists. |
| Additional deep-learning architectures beyond EEGNet/ShallowConvNet/DeepConvNet | The brief explicitly says "do not add additional architectures merely to increase the number of experiments." |
| Additional classical classifiers beyond the 7 named (LR, NLSVM, RF, GNB, KNN, LinearSVM, + CNN alone) | Already exactly matches the brief's specified core set — nothing added, nothing removed. |
| Large-scale hyperparameter search | Explicitly excluded by the task's opening instruction. Hyperparameters (learning rate, batch size, patience, epochs) are fixed, documented in `src/config.py`, and inherited from the original notebooks where applicable. |
| SHAP / Grad-CAM / additional explainability methods | Integrated Gradients alone satisfies "if already implemented, audit it" — no new explainability method was added, per the brief's "explainability is secondary" instruction. |
| Repeating the ablation/resolution experiments across all 5 seeds | Not requested, and would materially increase compute (5x) for experiments the brief frames as sensitivity checks, not primary comparisons. Flagged as a limitation (single-seed, provisional) in `AUDIT_REPORT.md` and `FINAL_EXPERIMENT_REPORT.md` instead of silently expanding scope. |

## Net effect on scope

**Zero new models, zero new architectures, zero hyperparameter search were
added.** The only code changes made in response to this audit are:
automating three tables that already had the underlying logic implemented
but not wired into the CLI script (`TABLE_1` dataset summary, `TABLE_4`
confidence intervals, `TABLE_5` model-comparison statistics, `TABLE_6`
128-vs-interpolation paired comparison), and making the previously-dead
`--skip-explainability` flag actually control something. Every REQUIRED
and RECOMMENDED experiment above was already designed and had already been
executed successfully once on real data before this audit began — what's
missing is not new science, it's re-running that already-correct design
one more time somewhere the output survives, and letting the newly-wired
tables summarize it.
