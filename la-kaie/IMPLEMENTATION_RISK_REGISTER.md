# IMPLEMENTATION_RISK_REGISTER — Revised N1 (rev. 3)

This register goes with `IMPLEMENTATION_SPEC.md` rev. 3; section references
(§) point to that spec. The changes are listed in `SPEC_REVIEW_REPORT.md`:
Part I covers rev. 2 and Part II covers rev. 3. No earlier risk was
deleted.

Rev. 3 reframes the research questions:
- **RQ1 (primary)** is the former E4: complementary structural evidence.
- **RQ2 (secondary)** covers exchange decisions: RQ2a is the representation
  decision; RQ2b is the former E3 (activation).
- In this register, "E4" and "RQ1" refer to the same question.

## Severity

| code | meaning |
|---|---|
| **C** | critical: triggering it means **STOP and report to the user** |
| H | high |
| M | medium |
| L | low |

A critical risk must never be silently patched, and the algorithm,
hypotheses and baselines must not be changed in response. Every **C** risk
has a full card in Part A, with its description, why it matters, detection,
mitigation and stop condition. Part B lists the non-critical risks.

---

## Part A — Critical risk cards

### Literature and scope

**R-LIT-GATE — implementation starts before the literature-verification gate is resolved**
- *Why it matters:* the novelty status is B, and the closest methods have
  not been verified from full text:
  - AIE + MAS;
  - LCC / LH-CC;
  - ACoS;
  - GOMEA / FOS;
  - dd-CMA-ES;
  - statistical transfer suppression;
  - island EDAs;
  - KBS 2023–2026.

  Implementing now risks investing in a design whose distinguishing
  elements may already exist.
- *Detection:* spec §16 Gate 0; no `src/n1/` exists before Gate 0 is
  closed.
- *Mitigation:* implementation is blocked (spec header). Inaccessible papers
  are recorded as unresolved evidence gaps.
- *Stop condition:* any code, test run or experiment before Gate 0 is
  resolved and the spec has passed independent review.

**R-INT-5 — novelty claimed from experimental improvement, or standard components marketed as novel**
- *Why it matters:* the evidence supports only class B, and the components
  (BIC, covariance, eigenbasis, linkage, …) are standard.
- *Detection:* review of every generated report.
- *Mitigation:* the scope statement in the spec header; the §14.6 rule that
  E4 outcomes are not converted into novelty claims.
- *Stop condition:* any report containing a novelty or superiority claim.
  Stop, and correct the report.

### E4 validity

**R-E4-SAMPLE — JOINT may appear superior because it receives more structural observations rather than because heterogeneous optimizer evidence is genuinely complementary**
- *Why it matters:* JOINT uses about 2W observations, while HBA-only and
  MPA-only use W each. A JOINT advantage over the single-optimizer
  selectors is therefore expected from sample size alone, and would not
  demonstrate E4.
- *Detection:* C2 (JOINT vs POOLED), C9 (sample-matched JOINT-HALF), and
  T-E4-OBS (identical observations for JOINT and POOLED).
- *Mitigation:* POOLED uses exactly the same combined observations as JOINT
  but removes the optimizer-specific parameterisation and structure. JOINT
  vs POOLED is therefore the **primary E4 control** (spec §4.4, §14.6).
  E4 is supported only under the intersection-union pattern:
  - JOINT > POOLED;
  - JOINT > HBA-only;
  - JOINT > MPA-only;
  - JOINT-HALF > both single-optimizer selectors.
- *Stop condition:*
  - any analysis or report that presents JOINT > HBA-only / MPA-only alone
    as evidence for E4;
  - Gate 9 failing (T-E4-OBS fails, or JOINT or POOLED is operationally
    invalid).

**R-BIC-4 — the covariance of successful steps reflects operator geometry, not landscape structure**
- *Why it matters:* E4 and Stage 2 assume that native successful
  displacements carry landscape structure. HBA/MPA move rules, which are
  directed toward the best, could impose structure of their own.
- *Detection:* Gate 5 (S1 behaves as HI); Gate 6 (S2 behaves as HB).
- *Mitigation:* none by design. This premise is what C1 tests.
- *Stop condition:* Gate 5 or Gate 6 fails in smoke.

### BIC power

**R-BIC-POWER — BIC may systematically favour HI/HB over HR because HR has substantially greater parameter complexity**
- *Why it matters:*
  - At D = 20, HR has q = D + D(D+1)/2 = 20 + 210 = 230 parameters, against
    n = W = 100 samples per population. The BIC penalty is
    0.5·230·ln 100 ≈ 530 nats per population.
  - HR may then be unreachable even on S3 (the dense rotation). If so, the
    HR hypothesis is untestable, and the HI/HB/HR comparisons are biased.
- *Detection:* S3 is a **hard diagnostic gate** (spec §16, Gate 7). The
  logged diagnostics are:
  - the number of samples used for structural fitting;
  - BIC of HI, HB and HR;
  - ΔBIC;
  - the selected-model frequencies;
  - sample counts;
  - whether HR is ever selected on S3.
- *Mitigation:*
  - The pre-registered BIC formulation is kept **unchanged** for the smoke
    specification.
  - T-BIC reports selection rates on synthetic Gaussian data at n = W. This
    shows whether the problem is statistical (power) or pipeline-related.
- *Stop condition:* HR selection on S3 is approximately absent (below 5% of
  evidence-bearing epochs) despite sufficient successful native-displacement
  samples.
  - That is evidence that the BIC/model-complexity formulation lacks
    adequate power for the intended HR detection task.
  - STOP; diagnose before any full experiment.
  - A remedy requires a documented proposal and user approval.
  - BIC is never tuned or redefined after full benchmark results.

### SPRT-style rule

**R-SPRT-CALIBRATION — the nominal thresholds may not correspond to classical error probabilities**
- *Why it matters:* the observations are adaptive and dependent, p0 is
  estimated online, and there are truncation and resets. Wald's α/β
  guarantees therefore do not apply. Interpreting H0/S7 behaviour through α
  and β would overstate reliability.
- *Detection:* T-SPRT calibration (spec §7.2). It reports:
  - the empirical false activation rate under controlled no-transfer;
  - the empirical false suppression rate under controlled positive
    transfer;
  - the stopping-time distribution;
  - Monte Carlo standard errors.
- *Mitigation:*
  - The terminology is "SPRT-style prequential sequential likelihood-ratio
    decision rule".
  - No formal type-I/II guarantee is claimed (§7.1).
  - Calibration is required before S7 or any H0 result is interpreted
    (Gate 8).
- *Stop condition:*
  - calibration not completed before interpreting H0/S7;
  - a false activation or false suppression rate above 3× nominal in
    Level 1 with σ_u = 0 (the "grossly off" condition of §7.2);
  - any report claiming a formal α/β guarantee.

**R-SPRT-1 — implementation bug in the LLR / boundary logic**
- *Why it matters:* every E3 result depends on this logic.
- *Detection:* T-SPRT-UNIT, under idealised Wald assumptions.
- *Mitigation:* the unit test runs before any experiment.
- *Stop condition:* T-SPRT-UNIT fails.

**R-SPRT-4 — exchange step lengths lie outside the native support, so the control model extrapolates**
- *Why it matters:* p0 is then unreliable, and E3 decisions become
  artefacts.
- *Detection:* the out-of-support fraction is logged (§4.8).
- *Mitigation:* clipping to [q1%, q99%].
- *Stop condition:* more than 50% of exchanges are out of support in the
  smoke run. Stop and report.

### S7 and S8 interpretation

**R-S7-NULL-CIRCULARITY — S7 treated as ground-truth H0**
- *Why it matters:*
  - S7 is a *controlled null-transfer scenario*: its construction makes
    useful exchange unlikely, but it does not make it mathematically
    impossible (for example, basin hopping could help).
  - Its reference label comes from batch exchange-vs-native evidence, using
    the same native-control model as the online rule. Agreement is
    therefore partly circular: it measures consistency between a sequential
    and a batch estimate, not correctness.
- *Detection:* review of the analysis code and reports for "ground truth",
  "oracle" or "correct H0" wording applied to S7.
- *Mitigation:*
  - The spec (§12, §14.5) uses the terms "controlled null-transfer
    scenario" and "offline reference label based on batch exchange-vs-native
    evidence".
  - C5 is interpreted only as agreement.
  - C5 is interpreted only after T-SPRT calibration.
- *Stop condition:* any report or claim that uses S7 as proof of a
  universally correct H0 detector, or calls its reference label ground
  truth.

**R-S8-STOCHASTIC-PLACEBO — S8 treated as a structure-recovery benchmark**
- *Why it matters:*
  - S8 redraws the rotation at every evaluation, so it has no stable
    coordinate structure and no HI/HB/HR label.
  - Scoring "accuracy" on S8, or pooling it with S1–S3, would be
    meaningless.
- *Detection:* S8 does not appear in the confirmatory structure tests
  C1–C4 or C9 (checked in the analysis script).
- *Mitigation:*
  - S8 is defined as a "stochastic placebo / robustness condition with no
    stable coordinate structure" (§12).
  - It is evaluated only by whether the mechanism avoids *confident*
    exploitation (C8, the PLL gap, switch frequency).
- *Stop condition:* S8 is reported with a structural label or included in
  structure-recovery accuracy.

### Budget and FE integrity

**R-FE-1 — hidden objective evaluations in the evidence, selection or diagnostic code**
- *Why it matters:* it breaks FE fairness and the declared budget.
- *Detection:* T-FE (a raising guard); the end-of-run FE audit (§5.5);
  Gate 3.
- *Mitigation:* the objective handle is never passed to the evidence
  modules.
- *Stop condition:* any audit mismatch or guard trigger.

**R-FE-2 — probes or exchanges exceed the declared budget**
- *Why it matters:* the budget is violated.
- *Detection:* `BudgetExceeded` is never caught; FE audit; Gate 2.
- *Mitigation:* the stop rule (§5.4).
- *Stop condition:* any exceedance.

**R-FE-5 — structural labels, reference labels or JOINT-HALF computed with objective calls**
- *Why it matters:* these would be hidden FE and possible leakage.
- *Detection:* the offline scripts read stored data only; T-FE.
- *Mitigation:* §5 table (0 FE, offline).
- *Stop condition:* any objective call in the offline code.

### Leakage and prequential validity

**R-LK-1 — CUR influences the HB partition that scores it**
- *Why it matters:* the same data would both choose and validate the
  partition.
- *Detection:* T-PREQ; Gate 4.
- *Mitigation:* the partition comes from OLD only (§4.3).
- *Stop condition:* T-PREQ fails.

**R-LK-2 — p0 computed after observing y, or the control model trained on the outcome it predicts**
- *Why it matters:* it violates predict → observe → update.
- *Detection:* the log-order assertion; T-PREQ; Gate 4.
- *Mitigation:* refit at step 4; p0 logged before evaluation.
- *Stop condition:* any violation.

**R-LK-3 — exchange-accepted moves enter the structural windows**
- *Why it matters:* the chosen structure would confirm itself.
- *Detection:* `op` tags; an assertion that no `op=EXCH` rows exist;
  Gate 4.
- *Mitigation:* only native operators are recorded (§3.2).
- *Stop condition:* any `op=EXCH` row in a window.

**R-LK-4 — shadow selectors perturb A0 trajectories**
- *Why it matters:* this would invalidate the trajectory-neutral E4 design,
  including JOINT vs POOLED.
- *Detection:* T-SHADOW (bit-identical A0); Gate 3.
- *Mitigation:* RNG isolation (§9).
- *Stop condition:* T-SHADOW fails.

**R-LK-5 — labels or reference labels visible to the algorithm**
- *Why it matters:* information leaks from the evaluation into the method.
- *Detection:* code review; labels are computed in a separate offline
  script.
- *Mitigation:* §12.
- *Stop condition:* any label accessible at run time.

### Mapping and baselines

**R-MAP-1 — mapping bug (the full mask does not give x_d)**
- *Why it matters:* structure comparisons become invalid.
- *Detection:* T-MAP.
- *Mitigation:* unit test.
- *Stop condition:* T-MAP fails.

**R-BASE-1 — MPHBS altered**
- *Why it matters:* the published reference is no longer faithful.
- *Detection:*
  - T-MPHBS;
  - the reproduction FE totals (199,930 / 149,857);
  - comparison with `baseline_validation_results.md`.
- *Mitigation:* MPHBS code is frozen, and N1 lives in `src/n1/`.
- *Stop condition:* any MPHBS change or failed re-verification.

**R-BASE-5 — initial populations differ between A4 and the N1 arms** (raised from M to C in rev. 2, because the spec requires identity)
- *Why it matters:* paired comparisons with MPHBS lose their common-seed
  basis.
- *Detection:* T-INIT.
- *Mitigation:* `rng_init = default_rng(seed)`, as in MPHBS (§15.1).
- *Stop condition:* T-INIT fails.

### RQ1 framing (rev. 3)

**R-RQ1-TRANSFER — RQ1 is reported as evidence "for transferable information" when it only shows structure identification**
- *Why it matters:*
  - RQ1's central test (JOINT vs POOLED accuracy against the known
    structure of S1–S3) measures *identification of structure*.
  - The known structure need not be the representation that makes
    exchange useful.
  - Claiming "transferable" without evidence would overstate the result.
- *Detection:* check reports for the transferability qualifier on cells
  where the T-REL check (§14.7) failed or could not be assessed.
- *Mitigation:*
  - T-REL, a pre-declared transfer-relevance check using the existing
    fixed-structure arms A1–A3.
  - The qualifier "for transferable information" is permitted only on
    cells where T-REL holds.
- *Stop condition:* any report attaching the transferability qualifier
  without T-REL support. Stop and correct the report.

**R-RQ2-ATTRIBUTION — an RQ2 benefit is attributed to complementary evidence when RQ1 is not supported, or when A7 does not beat P**
- *Why it matters:* an exchange benefit may come from the amount of
  evidence or from exchange itself, not from complementarity.
- *Detection:* the §14.7 conditional-interpretation table is applied to
  every RQ2 statement.
- *Mitigation:* the permitted statements are fixed in advance (§14.7).
  C10 (A7 vs P) is the attribution test.
- *Stop condition:* any RQ2 statement that falls outside the permitted
  table.

**R-FRAME-TIMING — the reframing is mistaken for post-hoc hypothesis change**
- *Why it matters:* integrity rule R-INT-3.
- *Detection:* git history.
- *Mitigation:*
  - The reframing (rev. 3) was made **before any N1 result exists**. No
    N1 code or experiment exists.
  - It was motivated by literature evidence (Gate 0), not by results.
  - It is logged in `SPEC_REVIEW_REPORT.md` Part II and must be frozen
    with the spec (§15.6).
- *Stop condition:* any further change of RQ1 or RQ2 after the
  `n1-freeze-synthetic` tag.

### Research integrity

**R-INT-1 — hyperparameters changed after full-test results**
- *Why it matters:* this is selective tuning.
- *Detection:* freeze tags; `DEVELOPMENT_CHANGES.md`; Gate 10.
- *Mitigation:* the freeze protocol (§15.6).
- *Stop condition:* any post-freeze change that is not a logged,
  failing-test bug fix.

**R-INT-2 — functions, runs or seeds removed or selected**
- *Why it matters:* this is selective reporting.
- *Detection:* the analysis asserts 30 runs per cell.
- *Mitigation:* §14.5.
- *Stop condition:* any removal.

**R-INT-3 — hypothesis definitions (E3, E4, HI/HB/HR, the E4 decision rule) changed after results**
- *Why it matters:* this is HARKing.
- *Detection:* the spec is under git, with freeze tags.
- *Mitigation:* §14.6 is fixed in advance.
- *Stop condition:* any change after results.

**R-INT-4 — a baseline modified to favour N1**
- *Why it matters:* the comparison becomes unfair.
- *Detection:* R-BASE-1 detection.
- *Mitigation:* frozen code.
- *Stop condition:* any modification.

**R-INT-6 — manuscript prose written**
- *Why it matters:* this is an explicit prohibition.
- *Detection:* review.
- *Mitigation:* none is produced.
- *Stop condition:* any manuscript section.

**R-INT-7 — proceeding past Stage 3 without approval**
- *Why it matters:* this is an explicit prohibition.
- *Detection:* stage log.
- *Mitigation:* STOP at `MECHANISM_SMOKE_REPORT.md`.
- *Stop condition:* any Stage-4 run without approval.

**R-INT-8 — smoke results used to tune anything**
- *Why it matters:* smoke seeds would become tuning seeds.
- *Detection:* the `n1_smoke` seeds are disjoint from the experiment
  seeds; `DEVELOPMENT_CHANGES.md`.
- *Mitigation:* smoke is for defects only.
- *Stop condition:* any non-bug-fix change after smoke without user
  approval.

**R-INT-9 — an additional mechanism (e.g. a change-point detector) added** (raised from H to C in rev. 2)
- *Why it matters:* the spec forbids new mechanisms (§6.3, §13.3), and
  added complexity would confound validation.
- *Detection:* spec diff and code review.
- *Mitigation:* §6.3 removes the change-point detector from the plan.
- *Stop condition:* any mechanism not in §1–§8.

---

## Part B — Non-critical risks (H / M / L)

| id | failure mode | Sev | Detect | Mitigation | If triggered |
|---|---|---|---|---|---|
| R-FE-3 | arms end with different unused FE | M | `fe_unused` logged | bound < 3N + E_x | report |
| R-FE-4 | the per-iteration cost of N1 and MPHBS differs (Pbest evaluation) | L | declared (§5.7) | identical total budget | report |
| R-LK-6 | S7 / S1–S4 reference-label circularity (superseded in detail by R-S7-NULL-CIRCULARITY; kept for traceability) | H | inherent | see card | report as a limitation |
| R-SPRT-2 | dependence and estimated p0 invalidate nominal α/β (**absorbed into R-SPRT-CALIBRATION**; kept for traceability) | H | T-SPRT calibration | see card | report |
| R-SPRT-3 | native-control confounding (operator type, receiver selection by tournament) | H | NA fit per operator | step-length and population covariates | report; no covariates added after results |
| R-SPRT-5 | complete separation / degenerate NA buffer | M | `separation` flag | smoothed fallback (§9) | report frequency |
| R-SPRT-6 | truncation dominates decisions | M | truncation fraction; T-SPRT calibration stopping-time distribution | n_max fixed | report |
| R-SPRT-7 | never reactivates (too few probes) | M | time to reactivation | r_probe fixed | report as a limitation |
| R-SPRT-8 | ACTIVE/SUPPRESSED chattering | M | state flips per run | reset policy; n_min | report |
| R-SPRT-9 | sensitivity to δ | H | δ grid on synthetic only | pre-declared grid | report; primary δ unchanged |
| R-SPRT-10 | "informative" (odds at matched step length) ≠ "improves the final result" | H | C5 vs C6 | declared | report |
| R-BIC-2 | non-Gaussian displacements (Lévy, clipping) | H | Mahalanobis diagnostics | declared | report |
| R-BIC-3 | within-window non-stationarity | H | OLD/CUR scale; PLL vs BIC gap | standardisation (§4.1); **no change-point detector** | report |
| R-BIC-5 | MPA elite-directed moves give rank-1 structure | H | `CUR_M` spectrum | centring | report |
| R-BIC-6 | α_link too conservative, so HB degenerates on S2 | H | ARI; degeneracy frequency; Gate 6 | fixed | report (Gate 6 failure is C via R-BIC-4) |
| R-BIC-7 | degenerate HB changes the candidate set | M | flags | no double counting | report |
| R-BIC-8 | sample-size advantage of JOINT (**superseded by R-E4-SAMPLE**; kept for traceability) | — | — | see card | — |
| R-BIC-9 | the DECOUPLED placebo is imperfect (column permutation changes the Stouffer statistic) | M | C4 caveat | declared | report |
| R-BIC-10 | HBA and MPA see genuinely different structure, so JOINT's shared-structure assumption fails and JOINT < POOLED can occur for this reason | M | HBA-only vs MPA-only agreement | reported | report; not reinterpreted as E4 support |
| R-BIC-11 | hysteresis delays or locks switches | M | S6 latency; forced switches | κ fixed | report |
| R-BIC-12 | long warm-up | M | no-evidence epoch fraction | default HI | report |
| R-E4-CEIL | ceiling: all selectors ≈ 100% accurate on S1, so the E4 pattern cannot be assessed | M | per-cell accuracy | "ceiling" rule (§14.6) | report as not assessable; not counted as support |
| R-E4-HALF | JOINT-HALF uses the newest W/2 rows, a time-selection that differs from single-source windows | L | declared | offline only | report |
| R-MAP-2 | unequal expected transfer size across HI/HB/HR | H | coordinates changed and ‖x' − x_r‖ logged | p_x per unit fixed | report |
| R-MAP-3 | HR eigenbasis unstable under near-equal eigenvalues | M | eigen-gap | declared | report |
| R-MAP-4 | clipping distorts structured moves | L | clip frequency | same clip as the backbone | report |
| R-MAP-5 | repeated donor/receiver pairs | L | unique-pair count | tournament | report |
| R-BASE-ASYM | single N1 backbone vs MPHBS domain configurations | M | declared | not engineered away | limitation |
| R-BASE-2 | A10 misread as ACoS | M | naming review | renamed "neutral success-rate representation-selection control"; explicitly not ACoS (§11) | correct the wording |
| R-BASE-3 | S7 per-population initialisation favours some arms | M | applied to all arms | declared | report |
| R-BASE-4 | common random numbers diverge after the first exchange | L | inherent | the seed is the pairing unit | report |
| R-DES-1 | synthetic labels are not revealable from displacements (S5 weak label; S8 has no label) | H | S5 and S8 are secondary; S8 is a placebo | §12 roles | report |
| R-DES-2 | exchange in S7 is actually helpful (e.g. basin hopping) | H | reference label | S7 is a controlled scenario, not truth | report as is |
| R-DES-3 | S6 transition unobservable within budget | M | post-hoc transition | r0 fixed | C7 counts "no transition" separately |
| R-DES-4 | multiple testing | M | — | Holm per family; families fixed (§14.3) | — |
| R-DES-5 | Wilcoxon approximation and ties at n = 30 | L | — | exact p where available | — |
| R-DES-6 | mechanism inactive on CEC/FIR (as with LA-KAIE) | H | exchange FE share; state and selection logs | mechanism metrics reported with performance | no mechanism-benefit claim without activity evidence |
| R-DES-7 | passive (A0) accuracy does not transfer to the active regime | M | A7 vs P, A5, A6 | §14.6 Q3 | report both |
| R-DES-8 | T-SPRT calibration Level 2 harness (native-equivalent null, oracle donor) differs from real exchange conditions | M | Levels 1 and 2 reported side by side | declared as a harness | report |
| R-TREL-FALLBACK | fixed-structure arms fall back to HI (A2 on S1, where the partition is degenerate; A3 during warm-up), so T-REL comparisons are not assessable | M | fallback-epoch fraction logged | fallback rule, with more than 50% fallback meaning not assessable (§14.7) | report the excluded count |
| R-RQ2B-PRECEDENT | RQ2b (activation relative to native search) is precedented in concept (AEMTO, Gate 0 G1); presenting it as a research contribution would overstate it | H | report review | RQ2b is framed as a component analysis only (spec header) | correct the wording |
| R-ENG-1 | runtime (5 selectors × 2 populations per epoch; 13 arms) | M | smoke timing | U = 5; single thread per run | report; ask before reducing arms |
| R-ENG-2 | cross-machine non-determinism | L | pinned environment | `capture_environment` | report |
| R-ENG-3 | log size, increased by per-epoch window logging for JOINT-HALF and BIC-power diagnostics | M | disk check | float32 npz per run, compressed | report; ask before dropping window logs |
| R-ENG-4 | CEC2022 library state | L | validated wrapper | unchanged | — |

---

## Consistency checklist (spec rev. 3 ↔ register rev. 3)

- [x] Rev. 3 additions are present: R-RQ1-TRANSFER, R-RQ2-ATTRIBUTION and
  R-FRAME-TIMING (critical cards), and R-TREL-FALLBACK and
  R-RQ2B-PRECEDENT (Part B).
- [x] RQ1 still uses JOINT vs POOLED as the central test (spec §14.6). The
  risks R-E4-SAMPLE and R-BIC-POWER apply to RQ1 unchanged.

### Carried over from rev. 2

- [x] Every **C** risk has a card with its description, why it matters,
  detection, mitigation and stop condition (Part A).
- [x] The required risks are present:
  - R-E4-SAMPLE;
  - R-BIC-POWER;
  - R-SPRT-CALIBRATION;
  - R-S7-NULL-CIRCULARITY;
  - R-S8-STOCHASTIC-PLACEBO.
- [x] Every spec gate (§16, Gates 0–10) maps to at least one risk:

  | gate | risk(s) |
  |---|---|
  | 0 | R-LIT-GATE |
  | 1 | R-SPRT-1, R-MAP-1, R-BASE-5 |
  | 2 | R-FE-2 |
  | 3 | R-FE-1, R-LK-4 |
  | 4 | R-LK-1..3 |
  | 5, 6 | R-BIC-4 |
  | 7 | R-BIC-POWER |
  | 8 | R-SPRT-CALIBRATION |
  | 9 | R-E4-SAMPLE |
  | 10 | R-INT-1 |

- [x] The terminology is consistent with the spec: "SPRT-style prequential
  sequential likelihood-ratio decision rule"; "controlled null-transfer
  scenario"; "offline reference label based on batch exchange-vs-native
  evidence"; "stochastic placebo / robustness condition"; "neutral
  success-rate representation-selection control".
- [x] No risk from rev. 1 was deleted. Superseded entries (R-BIC-8, R-LK-6,
  R-SPRT-2) are kept and marked.
- [x] No new algorithmic mechanism is introduced. JOINT-HALF and the T-SPRT
  calibration harness are offline analysis and test devices only.
