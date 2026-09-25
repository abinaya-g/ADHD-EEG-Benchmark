# Claim-to-Evidence Audit

For every major claim in `manuscript.md`, this table lists the evidence it
rests on, the statistical support (if any), whether causal language is used
and whether that is justified, and the final wording adopted. Claims are
grouped by topic, matching the areas the drafting brief asked to be checked
with particular care: leakage, superiority, generalization, clinical
relevance, the 512 Hz interpolation, CORAL, and architecture comparisons.

## Leakage

| Claim | Evidence | Statistical support | Causal language? | Final wording |
|---|---|---|---|---|
| "Outer-test subjects never contributed to any training, validation, or model-selection step" | `src/evaluation.py` structural design (no `validation_data` in final fit; inner CV restricted to `X_tr/y_tr/groups_tr`); `leakage_sanity_checks.csv`, all 14 checks passed | Not a statistical claim — a structural/implementation claim, verified by code inspection and a runtime audit | No causal language; this is a description of what the pipeline does, not a claim about its effect on results | Stated plainly in Methods 3.12 and re-confirmed in Results 4.2, with the audit's own limits stated ("the checks confirm implementation matches design, not that the design itself is beyond scrutiny") |
| "This effect [record-wise leakage] has been documented outside the EEG-ADHD literature" | External citations [7,8,9] | Those papers' own empirical/simulation results | No causal claim about *this* dataset's leakage, only that the general phenomenon is documented elsewhere | Introduction and Related Work cite this as established methodological literature, not as evidence about this specific dataset |
| "We cannot state that Amini et al. [11] or Hassan and Singhal [12]'s pipelines contain subject leakage" | No access to their training code or fold logs | None — explicitly stated as unverifiable | Explicitly avoided: the manuscript states we *cannot* make this causal claim | Section 2.3 and Section 5.8 both state this refusal to speculate explicitly, in the same sentence as raising the question |

## Superiority (model comparisons)

| Claim | Evidence | Statistical support | Causal language? | Final wording |
|---|---|---|---|---|
| "EEGNet ... significantly exceeding the plain CNN" | TABLE_5_MODEL_COMPARISON_STATISTICS.csv, row CNN\|128Hz vs EEGNet\|128Hz | Wilcoxon signed-rank, n=25 paired folds, p_holm=0.0002 | No — "significantly exceeding" is the correct statistical term given the test result, not a causal claim about *why* | Used consistently as "significantly [exceeds/differs from]" tied to a specific p-value and paired-fold count, never as an unqualified "is better" |
| "Six of seven CNN-derived hybrid classifiers ... outperformed the plain CNN significantly" | TABLE_5, rows for LR, LR_noCORAL, LR_withCORAL, NLSVM, RF, KNN | Same test, p_holm ranging 0.0031–0.0227 | No | Abstract and Results state the exact count (6 of 7) and name the one exception (GNB) is not part of this significant set alongside LinearSVM — see next row |
| "CNN+LinearSVM and CNN+GNB do not reach significance after correction, despite ... numerically higher point estimate" | TABLE_5, p_holm=0.1585 (LinearSVM), 0.5588 (GNB) | Same test | Explicitly framed to avoid implying a numerically larger mean is evidence of superiority | Results 4.6 states this contrast directly as the illustrative case for "a numerically larger mean is not, by itself, treated as evidence of superiority" |
| "DeepConvNet does not differ significantly from the plain CNN" | TABLE_5, p_holm=1.0, mean diff −0.006 (smallest in the table) | Same test | No — stated as "did not differ significantly," never as "is equivalent to" | Results 4.6 and Discussion 5.4 both use the "no significant difference detected" phrasing, consistent with Methods 3.17's explicit rule against reading non-significance as equivalence |

## Generalization

| Claim | Evidence | Statistical support | Causal language? | Final wording |
|---|---|---|---|---|
| "generalizes across held-out children in the 0.55–0.79 balanced-accuracy range" | TABLE_3, full range of mean_probability balanced_accuracy values at 128Hz (0.542–0.793) | Point estimates + bootstrap CIs (Table 4) | No causal claim, a descriptive range | Abstract and Conclusion state this range and its bounds explicitly, tied to Table 3/4 |
| "Internal cross-validation ... estimates generalization to new subjects from the same recording protocol and population, not to a genuinely external population" | Methodological reasoning from the CV design itself (Section 3.12) plus [7,9] | N/A — a definitional statement about what nested CV does and does not estimate | Explicitly limits the generalization claim's scope | Limitations section, stated as a named limitation, not folded into the Results as if already addressed |

## Clinical relevance

| Claim | Evidence | Statistical support | Causal language? | Final wording |
|---|---|---|---|---|
| "No model ... reaches a level of ... performance that we would characterize as clinically actionable" | Best balanced accuracy obtained = 0.793 (EEGNet), with 95% CI 0.723–0.862 | Same CI evidence | No overclaiming; this is a refusal of a claim, not an assertion of one | Discussion 5.10, phrased as declining to make a clinical-utility claim, consistent with the brief's explicit instruction never to claim "ready for clinical diagnosis" |
| "not a validated instrument for individual clinical decision-making" | Same as above, plus absence of external validation (Limitations) | N/A | No | Discussion 5.10 and Conclusion both state this explicitly |

## 512 Hz interpolation

| Claim | Evidence | Statistical support | Causal language? | Final wording |
|---|---|---|---|---|
| "This is interpolation of the existing 128 Hz samples, not a reconstruction of a genuine 512 Hz acquisition" | `src/data.py:resample_pipeline_b` implementation (scipy.signal.resample of existing samples) | N/A — a description of what the function mathematically does | Explicit denial of a causal/representational claim that the literature sometimes implies | Stated verbatim in Methods 3.19, repeated in Limitations, used consistently as "128-to-512 Hz interpolation" throughout, never "512 Hz acquisition" or "512 Hz pipeline" |
| "Neither comparison reaches significance ... we do not interpret either numerical difference as evidence that interpolation helps or harms classification, nor as evidence about the original recording's true sampling rate" | TABLE_6 paired test, p=0.50 (CNN), p=0.19 (EEGNet), n_paired_folds=5 each | Wilcoxon signed-rank | Explicitly refuses two possible causal readings of a null result | Results 4.8 and Discussion 5.6 both state this refusal directly, including the specific reasoning ("interpolation cannot recover information not present in the samples being interpolated") |

## CORAL

| Claim | Evidence | Statistical support | Causal language? | Final wording |
|---|---|---|---|---|
| "CORAL-aligned features did not outperform the unaligned features" | TABLE_3: CNN+LR_withCORAL=0.735 vs CNN+LR=0.743 | Point estimates and overlapping 95% CIs (0.651–0.816 vs 0.662–0.819); **no dedicated paired significance test between these two specific configurations exists in TABLE_5** (its reference model is the plain CNN, not CNN+LR_noCORAL) | Explicitly flagged as an *observed*, not statistically confirmed, difference | Results 4.7 states this distinction explicitly: "We did not run a dedicated paired significance test between these two specific configurations ... this should be read as an observed, not a statistically confirmed, difference" |
| "computing cov_target from outer-test subjects' unlabelled features ... is a transductive unsupervised domain-adaptation setting, not standard inductive subject-independent evaluation" | `src/coral.py` implementation (whitens source, re-colors with target covariance; no target label used) | N/A — structural claim about the algorithm, verified by code inspection | No causal claim; a classification of the experimental setting | Methods 3.18 states this explicitly and repeats the same framing in the Related Work discussion of CORAL in the wider literature |

## Architecture comparisons

| Claim | Evidence | Statistical support | Causal language? | Final wording |
|---|---|---|---|---|
| "EEGNet, ShallowConvNet, and DeepConvNet were reimplemented ... not verified line by line against the original authors' own code" | `src/models.py` module docstring, which states this explicitly as a known limitation of the implementation | N/A | No claim of exact reproduction is made | Methods 3.9–3.11 caveat and the Limitations section both state this; the manuscript describes these as "reimplementations following the published architecture descriptions," never as "exact replications" |
| "DeepConvNet's greater parameter count is a plausible, though not confirmed, contributor to its comparatively weak generalization" | DeepConvNet's known architecture (4 convolutional blocks, up to 200 filters) vs the dataset size (121 subjects) | No direct statistical test of a parameter-count-vs-performance relationship was run | Explicitly hedged ("plausible," "not confirmed") rather than asserted | Discussion 5.4, worded as a hypothesis for future work, not a finding |
| "A plausible explanation, which this study's design cannot fully adjudicate, is that the CNN's small dataset makes its jointly trained classification head prone to ... overfitting" | Same dataset-size reasoning | No direct test | Explicitly hedged | Discussion 5.3, same hedging pattern |

## Overall audit outcome

No claim in the manuscript was found, on this pass, to assert statistical
superiority from a numerically larger point estimate alone, to treat a
non-significant p-value as proof of equivalence, to describe the
128-to-512 Hz interpolation as a genuine higher-resolution acquisition, to
claim an exact reproduction of EEGNet/ShallowConvNet/DeepConvNet beyond what
was actually verified, or to make an unsupported clinical-readiness claim.
Two claims (rows under "Superiority" and "CORAL") required a wording
correction during this audit pass to make the distinction between "observed
difference" and "statistically tested difference" explicit; both are shown
corrected above.
