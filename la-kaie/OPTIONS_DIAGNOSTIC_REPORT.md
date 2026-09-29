# Diagnostic report: option 2 (specification revision) vs option 3 (re-scope)

- Date: 2026-09-29.
- Status: analysis only. No code was written, no run was made, and no parameter was changed
  for this report.
- Inputs:
  - `MECHANISM_SMOKE_REPORT.md` (Gates 5 and 6 FAIL, plus the diagnostic appendix);
  - `results/n1_calibration/T_SPRT_CALIBRATION_REPORT.md`;
  - `MPHBS_REPRODUCTION_REPORT.md`;
  - `IMPLEMENTATION_SPEC.md` rev. 2.1;
  - `IMPLEMENTATION_RISK_REGISTER.md` rev. 2.
- Novelty status is unchanged (B). Nothing in this report is a novelty claim.

---

## 1. What has to be explained

These are facts from the smoke campaign: 5 runs per cell, passive A0 shadows, JOINT selector.

| function (known structure) | D | in-window BIC share HI/HB/HR | prequential PLL share HI/HB/HR | distinct iterations per W-row window |
|---|---|---|---|---|
| S1 (HI) | 10 | 0.47 / 0.25 / 0.28 | 0.68 / 0.27 / 0.05 | 11 |
| S1 (HI) | 20 | 0.02 / 0.08 / 0.89 | 0.92 / 0.05 / 0.02 | 9 |
| S2 (HB) | 10 | 0.00 / 0.10 / 0.89 | 0.53 / 0.19 / 0.28 | 6 |
| S2 (HB) | 20 | 0.01 / 0.08 / 0.91 | 0.92 / 0.04 / 0.04 | 9 |
| S3 (HR) | 10 | 0.00 / 0.00 / 1.00 | 0.49 / 0.01 / 0.50 | 16 |
| S3 (HR) | 20 | 0.00 / 0.00 / 1.00 | 0.79 / 0.01 / 0.20 | 16 |

Additional facts:
- The implementation is not implicated:
  - likelihood nesting held in 78,145 of 78,147 epochs;
  - the log-likelihood matches scipy;
  - on i.i.d. Gaussian data, BIC recovers HI and HB in 20/20 trials, and HR in 20/20 at
    D = 20 and 6/20 at D = 10.
- The estimated S2 partitions do not match the true blocks (ARI ≈ 0).
- The single-source selectors disagree on S1 at D = 10 (HI chosen: HBA-only 0.31, MPA-only
  0.65).

**Working diagnosis.** It has three parts, with different levels of support.

- **D1 — dependence (well supported).**
  - The W rows of a window come from only 6–16 iterations of the same 30 agents.
  - BIC counts them as W independent observations. That inflates the evidence for
    parameter-rich models.
- **D2 — transience (well supported).**
  - The structure found inside a window does not predict the next window. On S1 and S2 at
    D = 20, PLL prefers HI in 92% of epochs.
  - The detected correlation is therefore a property of the population's momentary
    geometry, not of the landscape.
- **D3 — operator geometry (plausible, not yet isolated).**
  - HBA and MPA move agents toward a shared best point (xprey or elite). A successful
    displacement therefore approximates "best point − current position". Its covariance
    then approximates the current population covariance, whatever the landscape.
  - This is the mechanism I consider most likely. I have **not** yet separated it from
    D1 and D2 by a controlled experiment.

A fourth fact matters for both options.
- **Neither criterion discriminates.**
  - The in-window BIC picks HR everywhere, so the S3 pass tells us nothing.
  - The prequential PLL picks HI almost everywhere, including S3 at D = 20 (0.79).
- On the current evidence stream, no criterion we have looked at separates S1, S2 and S3.
  **The problem is the evidence, not only the criterion.**

---

## 2. Option 2: revise the specification and keep the research question

### 2.1 Candidate revisions and what each addresses

| revision | targets | expected effect | main risk |
|---|---|---|---|
| **2a Dependence-aware evidence** (one row per agent per epoch; effective-sample-size-scaled BIC penalty; block bootstrap over iterations) | D1 | removes the spurious HR preference on S1 and S2 | also removes power. With ≈ 6–16 effective blocks per window, HR (230 parameters at D = 20) may become undetectable, so Gate 7 would fail instead. This is R-BIC-POWER from the other side |
| **2b Prequential PLL (or cross-validated likelihood) as the selection criterion** | D2 | selects only structure that generalises over time | the existing logs show it fails S3 (HR only 0.20 at D = 20) |
| **2c Whitening displacements by the current population geometry** (e.g. express steps in the population's own coordinate system, or model the residual after regressing on "best − x") | D3 | removes the component that merely copies the population shape | the population shape *is* partly landscape-driven: on an ill-conditioned rotated ellipsoid, a converging population elongates along the landscape axes. Whitening can therefore remove the signal that S3 needs. Its net effect is unknown |
| **2d Longer, slowly updated evidence** (exponential accumulation over many iterations, in the spirit of covariance-learning evolution strategies) | D1, D2 | averages out transient geometry | moves the method toward established covariance-adaptation machinery, which weakens the distinction from existing work (novelty status B, possibly lower). It also slows reaction to structure changes (S6) |
| **2e Different evidence source** (e.g. finite-difference or paired-perturbation probes instead of successful native moves) | D1–D3 | measures landscape structure directly | costs FE (contradicting "FE-free evidence", a core feature of N1). It also resembles existing linkage-learning / differential-grouping probes. This changes the method substantially |

### 2.2 Evidence for or against option 2 that already exists
- **Against:**
  - The one criterion change that can be read from existing logs (2b) fails S3.
  - No revision has been shown to pass S1, S2 **and** S3 together.
  - 2a and 2c each attack one failure mode while plausibly creating another (loss of power
    on S3; loss of signal on S3).
- **For:**
  - D1 and D2 are specific, testable and conventional statistical problems, not a
    conceptual dead end.
  - The existing infrastructure all stays valid, including the FE accounting, leakage
    controls, shadow design, JOINT/POOLED identity and calibration harness.

### 2.3 Integrity implications
- Any option-2 revision is a **method change made after seeing smoke results**. To keep the
  study credible:
  1. freeze a new specification (rev. 3.x) that states it is a post-smoke redesign and why;
  2. choose the revision **before** looking at any new outcome data;
  3. run a fresh smoke campaign on **new seeds**. The `n1_smoke` seeds are now "used";
  4. report the failed rev. 2.1 smoke result in any eventual write-up;
  5. if several revisions are screened, report all of them. Screening several and keeping
     the best is a forking-paths risk and must be declared.
- Using the existing smoke logs to *screen* candidate revisions offline is possible at zero
  FE, but it is exploratory. Only a fresh-seed confirmation would count.

### 2.4 Effect on the research claim
- E4 (JOINT vs POOLED) can only be tested once some evidence stream separates S1, S2 and S3.
  Until then the primary test has nothing to compare.
- Revisions 2d and 2e move N1 toward known covariance-adaptation or linkage-probing
  approaches. The remaining distinction from existing work — heterogeneous-optimizer
  complementarity — would then carry the whole claim.
- Gate 0 already found that exchange regulation (E3) is precedented. Option 2 therefore
  risks ending with a method whose distinguishing element is small.

### 2.5 Rough cost
- One specification revision and review cycle.
- Implementation changes confined to `src/n1/records.py` and `src/n1/selectors.py` (for
  2a–2d).
- A new smoke plus calibration run: about 20 minutes of compute at the observed ≈ 9 s per
  run.
- The dominant cost is the design and review time, plus the probability of a second gate
  failure. I would estimate that probability as substantial, given §2.2, but I cannot
  quantify it.

---

## 3. Option 3: re-scope to characterising the artefact

### 3.1 Candidate research question
> To what extent do the successful native displacements of population-based optimizers
> (here HBA and MPA) encode landscape structure, and to what extent operator and
> population geometry — and can the two be separated?

Possible sub-questions:
- **Q3.1** How does displacement covariance relate to (a) the population covariance,
  (b) the landscape Hessian structure (known for S1–S3), and (c) the operator form? Studied
  over time and across D.
- **Q3.2** Which operator features produce the shared-direction (rank-structured)
  component? Examples are elite- and xprey-directed moves, Lévy steps and FAD.
- **Q3.3** Do the two optimizers carry *different* biases? If so, does combining them
  cancel part of the artefact? This is where the JOINT / POOLED / single-source machinery
  would still apply, but as a measurement tool rather than a method.
- **Q3.4** How reliable are common structure-detection statistics (BIC, correlation-based
  partitions, PLL) when applied to optimizer-generated samples rather than i.i.d. samples?

### 3.2 Evidence already in hand
- The smoke logs already contain the key observations:
  - in-window vs prequential disagreement;
  - window dependence;
  - single-source disagreement;
  - placebo behaviour;
  - ARI ≈ 0 on S2.
- These are exactly what option 3 would study, and they are robust across all 5 runs per
  cell.
- The passive A0 design is directly suitable: it measures without changing the trajectory.

### 3.3 What would still be needed
1. A new pre-registered design with its own hypotheses. Examples:
   - displacement covariance tracks population covariance more closely than the landscape
     Hessian;
   - the effect scales with D;
   - operator-ablated variants reduce it.
2. Additional controlled conditions:
   - an isotropic sphere, where no structure should be found;
   - conditioning sweeps;
   - rotations at known angles;
   - possibly operator ablations (e.g. removing the elite-directed component).
3. Quantities not logged yet: the population covariance at each epoch, and a per-row agent
   id. Both are cheap to add.

### 3.4 Integrity implications
- The research question changes, so this is a new study. It must be declared as a
  re-scoping prompted by a negative mechanism result, not presented as the original plan.
- The existing smoke data are exploratory for the new question. Confirmatory claims need
  fresh seeds under the new pre-registration.

### 3.5 Effect on the research claim
- Contribution type: a characterisation or negative-result study, i.e. "a class of
  FE-free structural evidence is confounded by operator geometry". This kind of result
  can be useful to the field, but it is not a new optimizer.
- Venue fit is uncertain. Some optimization journals publish analysis papers; a
  knowledge-based-systems venue may expect a method. I have not checked current editorial
  policies, so treat this as a judgment, not a fact.
- **Novelty exposure is different, not absent.**
  - Landscape analysis from sampled points, and from algorithm search trajectories, is an
    established area. Exploratory landscape analysis is the best-known family, and
    trajectory-based variants have been studied.
  - I have not verified which specific works analyse displacement covariance of HBA- or
    MPA-type operators, so a new literature check would be required.
  - The Gate 0 access limits (publisher sites blocked) would apply again.

### 3.6 Rough cost
- Small implementation additions: extra logging plus a few synthetic controls.
- A new pre-registration.
- Analysis-heavy rather than algorithm-heavy.
- The probability of obtaining *a* reportable result is high, because the effect is
  already visible. The open question is whether it is interesting enough to publish.

---

## 4. Side-by-side comparison

| criterion | option 2 (revise the N1 specification) | option 3 (re-scope to the artefact) |
|---|---|---|
| addresses the diagnosed cause | partially: each revision targets one of D1–D3, and none is known to fix all three | studies the cause directly |
| existing evidence that it can succeed | weak: the one testable change (PLL) fails S3; the rest are untested | strong for *detecting* the effect; unknown for *explaining* it (D3 not yet isolated) |
| probability of passing its own gates | uncertain; a second gate failure is plausible | high for descriptive gates; unknown for mechanistic hypotheses |
| keeps the original research question (E3/E4) | yes | no (E4 becomes a measurement question) |
| novelty exposure | rises: revisions move toward covariance adaptation or linkage probing; E3 is already precedented | different: landscape analysis and trajectory analysis literature; needs a new check |
| integrity burden | post-smoke redesign; new seeds; forking-paths declaration | new study; re-scoping declaration; new seeds |
| reuse of the current code | high | high (passive design, selectors and logs reused as instruments) |
| compute cost | low | low to moderate (more control conditions) |
| main failure mode | a method that passes the gates but is close to existing methods, or one that fails the gates again | a correct but modest descriptive result |
| type of eventual claim | a method with pre-registered mechanism evidence | an evidence-quality or negative-result characterisation |

---

## 5. Assessment

This assessment is my judgment on the evidence above; it is not a decision.

1. The most consequential finding is §1's fourth fact: **the evidence stream, not only the
   criterion, fails to separate S1, S2 and S3.**
   - This favours option 3 on scientific grounds: it studies the thing that failed.
   - It also makes option 2's success depend on changing the evidence itself (2a, 2c or
     2e), not just the scoring.
2. **Option 2 is only worth pursuing if a cheap exploratory screen shows that some evidence
   transformation separates S1, S2 and S3 on the existing logs.** Candidates are thinning to
   one row per agent per epoch, whitening, and long accumulation.
   - The screen would be offline and zero-FE, and could be run in both directions.
   - If nothing separates them, option 2 is very unlikely to pass fresh-seed gates.
   - Such a screen is exploratory and must be labelled as such.
3. The two options are **not mutually exclusive in sequence.**
   - The option-3 measurements (Q3.1–Q3.3) are exactly the diagnostics that would tell us
     whether any option-2 revision can work.
   - A defensible path is: option-3 characterisation first, as its own pre-registered
     study. Only if it identifies a separable landscape component, a later option-2 method
     revision.
4. Carried-over caveats that apply to either option:
   - the unresolved CEC2022 F10 MPHBS discrepancy;
   - the above-nominal SPRT error rates (0.066 idealised replay; 0.147 with correlated
     outcomes; 0.25 false suppression in the pipeline harness).

No option is selected by this report. Both require a new, frozen specification and fresh
seeds before any confirmatory run.
