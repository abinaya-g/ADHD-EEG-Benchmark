# SPEC_REVIEW_REPORT — N1 specification, rev. 1 → rev. 2

## 1. Current status

- **Documents revised:**
  - `IMPLEMENTATION_SPEC.md` (now rev. 2);
  - `IMPLEMENTATION_RISK_REGISTER.md` (now rev. 2).
- **Novelty classification:** unchanged — **B, "promising, but important
  unresolved overlap/evidence gaps remain"**. It has not been upgraded.
- **Implementation:** blocked until two conditions hold:
  1. the literature-verification gate (spec §16, Gate 0) is resolved
     separately;
  2. the revised specification has passed independent review.
- **Core algorithm preserved:**
  - E3 and E4;
  - Stage 1 (SPRT-style activation/suppression);
  - Stage 2 (HI/HB/HR by BIC);
  - HBA/MPA backbone;
  - selectors: JOINT, POOLED, DECOUPLED, HBA-only, MPA-only;
  - native control;
  - MPHBS baseline;
  - fixed-structure baselines.
- **No new algorithmic mechanism was added.** The two additions are
  analysis and test devices only; neither can influence a run:
  - JOINT-HALF, an offline sample-matched analysis;
  - the T-SPRT calibration harness.

## 2–6. Corrections

Columns:
- **Alg. behaviour?** — does the correction change what the algorithm does
  during a run?
- **Interpretation?** — does it change how experimental results are
  interpreted?

| # | correction | exact sections changed | reason | Alg. behaviour? | Interpretation? |
|---|---|---|---|---|---|
| 1 | **JOINT vs POOLED made the primary E4 control.** Required pattern: JOINT > POOLED ∧ JOINT > HBA-only ∧ JOINT > MPA-only. JOINT > single-optimizer alone is declared insufficient. Added the sample-matched JOINT-HALF analysis (C9) and unit test T-E4-OBS (JOINT and POOLED see identical observations) | spec header (E4 definition); §3.5 (window logging); §4.4 (selector table with roles; POOLED defined as one common pooled model on identical observations; JOINT-HALF); §5 table; §6.1 (P driving selector); §6.2; §11 (P = PRIMARY E4 SAMPLE-SIZE CONTROL); §13.1 (T-E4-OBS); §13.2; §14.3 (C2 = JOINT > POOLED primary; C3; C9); §14.6 (new E4 analysis, Q1–Q5, intersection-union decision rule, ceiling rule); §16 Gate 9. Register: R-E4-SAMPLE card; R-BIC-8 marked superseded; R-E4-CEIL and R-E4-HALF added | JOINT has about 2× the observations of each single-optimizer selector, so its advantage could be a pure sample-size effect | **No.** POOLED, JOINT and the single-optimizer selectors already existed. The only additions are window logging (no effect on the run) and an offline analysis | **Yes.** E4 now requires JOINT > POOLED; a JOINT advantage over single sources alone no longer counts |
| 2 | **S3 made a hard diagnostic gate for BIC power.** The BIC formulation is kept unchanged. Required diagnostics were added | §3.5 (ΔBIC and sample-count logging); §4.2 (pre-registration note, with the q = 230 example at D = 20); §12 (S3 role); §14.2 (BIC-power diagnostics); §16 Gate 7 with its failure rule. Register: R-BIC-POWER card with a stop condition | a full covariance has many parameters, and BIC may make HR unreachable | **No.** Logging only | **Yes.** An absent HR selection on S3 with sufficient samples is a STOP condition that needs diagnosis before full experiments |
| 3 | **SPRT terminology and assumptions.** Now "SPRT-style prequential sequential likelihood-ratio decision rule"; α and β are nominal. Documented: p0 estimation, dependence, truncation, reset, accumulation, boundaries and minimum samples. Added T-SPRT calibration with two levels (false activation, false suppression, stopping time). Renamed the unit test to T-SPRT-UNIT | §1 (Stage 1); §1.1 and §2 wording; §4.8 (title; components table); §5 table; §6; §7 (thresholds as nominal); §7.1 (new: assumptions, no guarantee); §7.2 (new: T-SPRT calibration); §13.1; §14.3 C5; §15.4 stage 3b; §16 Gate 8. Register: R-SPRT-CALIBRATION card; R-SPRT-2 absorbed; R-DES-8 added | Wald guarantees need i.i.d. data with known hypotheses, no truncation and no reset; none of these holds here | **No.** Decision logic, thresholds and parameters are unchanged. Calibration is a separate harness | **Yes.** No α/β guarantee is claimed, and H0/S7 results can be interpreted only after calibration |
| 4 | **S7 is no longer "ground-truth H0".** Now a "controlled null-transfer scenario" with an "offline reference label based on batch exchange-vs-native evidence", explicitly **not** an independent ground-truth oracle | §5 table; §12 (roles; S7 row; label rules); §14.2; §14.3 (C5, C7 wording); §14.5 (reporting). Register: R-S7-NULL-CIRCULARITY card; R-LK-6 kept for traceability; R-DES-2 | the reference is built from the same kind of evidence and the same control model as the online rule, so agreement is partly circular | **No** | **Yes.** C5 means agreement between sequential and batch estimates, not correctness; S7 cannot support claims of a universally correct H0 detector |
| 5 | **S8 is a stochastic placebo / robustness condition.** Structure-recovery tests are now split into PRIMARY (S1–S3) and SECONDARY/CONTROL (S4–S8). S8 has no HI/HB/HR label and is evaluated by *confident-selection* avoidance | §12 (roles table; S8 row; S8 measures); §14.3 (C1–C4 and C9 restricted to S1–S3; C8 redefined as confident HB/HR on S8 < S3; S4 moved to a secondary family); §14.5. Register: R-S8-STOCHASTIC-PLACEBO card; R-DES-1 updated | S8 redraws the rotation at every evaluation, so no stable structure exists to recover | **No** | **Yes.** Confirmatory families are restricted to S1–S3 and shrink: chance test 8→6 tests; JOINT vs POOLED from 8 two-sided to 6 one-sided (C2); JOINT vs single-source 16→12 (C3); DECOUPLED 6→4 (C4). S4 becomes secondary |
| 6 | **A10 renamed** to "neutral success-rate representation-selection control", described as conceptually inspired by success-based coordinate-system selection and explicitly *not* ACoS | §6.1; §11 (A10 row); §13.2. Register: R-BASE-2 | the published ACoS algorithm has not been reproduced from its paper or code | **No** | Minor: A10 results cannot be described as an ACoS comparison |
| 7 | **Change-point detector removed from the plan** (previously "not in v1; may be added after a gate") | §6.3 (rewritten); §13.3 (rewritten: no additional mechanisms). Register: R-INT-9 raised to C; R-BIC-3 mitigation text | keep the method simple; validate E3/E4, BIC, calibration, FE and leakage first | **No** (it was never part of v1) | No |
| 8 | **Core algorithm preserved** | no algorithmic section changed: §4.1–4.3, §4.5–4.7, §4.9 and §8–§10 keep their rev. 1 content | instruction | No | No |
| 9 | **Ablation / baseline table**: every variant now has an explicit purpose. P is marked as the PRIMARY E4 SAMPLE-SIZE CONTROL | §11 (now 3 columns) | instruction | No | Clarifies the purposes |
| 10 | **E4 analysis defined before experiments** (Q1–Q5; the intended interpretation; no novelty conversion) | §14.6 (new) | instruction | No | **Yes** (see correction 1) |
| 11 | **Risk register rebuilt**: Part A has full cards for every critical risk; Part B lists the rest. The five required risks were added. R-BASE-5 was raised to C, as was R-INT-9. New entries: R-LIT-GATE, R-E4-CEIL, R-E4-HALF, R-DES-8. No rev. 1 risk was deleted | `IMPLEMENTATION_RISK_REGISTER.md` (whole file) | instruction | No | No |
| 12 | **Experimental gates**: Gate 0 (literature) plus Gates 1–10, each with an a-priori operational criterion. The stage table now includes a blocked row and stage 3b | §16 (new); §15.4 | instruction | No | Progress conditions made explicit |
| 13 | **Novelty status stays B**; the outstanding verification list is recorded | spec header; §16 Gate 0 list. Register: R-LIT-GATE, R-INT-5 | instruction | No | No |

Two points are worth a reviewer's attention.

**a) Gate thresholds are my a-priori choices.** They are listed here so the
independent review can accept or change them *before* any result exists:

| gate | threshold |
|---|---|
| Gate 5/6 | modal label correct in ≥ 4 of 5 smoke runs per D |
| Gate 7 | HR selected in ≥ 5% of evidence-bearing epochs, with ≥ W samples per population in ≥ 50% of epochs |
| Gate 9 | ≥ 90% valid epochs |
| T-SPRT calibration | "grossly off" = more than 3× nominal |
| §14.6 | ceiling = accuracy ≥ 0.95 for all selectors |
| T-SPRT calibration | σ_u ∈ {0, 0.5} |

**b) T-SPRT calibration Level 2 uses test-only harness modifications.**
These are a native-equivalent candidate as the null and the known optimum
as the donor. They are fixtures, not N1 variants. Whether this is
acceptable is a review question.

## 7. Remaining unresolved risks

1. **R-BIC-POWER:** HR may be unreachable at W = max(60, 5D). This is
   unknown until Gate 7 is evaluated.
2. **R-BIC-4:** successful-step covariance may reflect operator geometry
   rather than landscape structure. This is the premise of Stage 2 and
   E4, and it is untested.
3. **R-SPRT-CALIBRATION:** the realised error behaviour of the SPRT-style
   rule is unknown.
4. **R-E4-SAMPLE / R-BIC-10:** JOINT may legitimately lose to POOLED when
   the two optimizers see different structure. E4 could then be
   unsupported for reasons that are informative but negative.
5. **R-S7-NULL-CIRCULARITY:** there is no independent ground truth for H0,
   and none is available by construction.
6. **R-SPRT-3:** native-control confounding (operator type, tournament
   selection of receivers).
7. **R-DES-6:** the mechanism may be inactive on CEC2022 and FIR, as
   happened with LA-KAIE.
8. **R-BASE-ASYM:** N1 uses a single backbone, while MPHBS uses domain
   configurations.
9. **R-ENG-3:** logging the windows every epoch, which JOINT-HALF and the
   BIC diagnostics need, increases log size.

## 8. Remaining novelty-verification gaps

There is still no full-text verification for any of the items below. The
evidence log was compiled on 2026-09-28; the access conditions are
recorded in `N1_NOVELTY_EVIDENCE_LOG.md`.

- the ensemble knowledge-transfer framework with AIE + MAS
  (S2210650223001670) and related multitasking transfer-selection methods;
- LCC (arXiv 2504.17578) and LH-CC (doi 10.1145/3795095.3805054);
- ACoS (doi 10.1109/tcyb.2018.2802912) and its ASOC follow-up;
- GOMEA / FOS-related structural selection. The library code has been
  checked; the papers, including *Predetermined versus learned linkage
  models* (GECCO 2012), have not;
- dd-CMA-ES. The code has been checked; the paper has not;
- statistical transfer-suppression methods: OKTPO-MFEA, MGAD, and Bayesian
  competitive knowledge transfer (arXiv 2510.23407);
- island / multi-population EDA model-migration methods;
- relevant KBS 2023–2026 literature (evidence log S14).

Items that remain inaccessible are recorded as unresolved evidence gaps,
not as absence of overlap.

## 9–11. Statements

**NO CODE IMPLEMENTATION HAS BEEN STARTED.**

**NO EXPERIMENTS HAVE BEEN RUN.**

**NO MANUSCRIPT TEXT HAS BEEN WRITTEN.**

In this step, no Python was executed and no tests were run. Only these
three Markdown files were edited or created.
