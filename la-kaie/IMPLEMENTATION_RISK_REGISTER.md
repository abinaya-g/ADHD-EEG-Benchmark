# IMPLEMENTATION_RISK_REGISTER — Revised N1

This register goes with `IMPLEMENTATION_SPEC.md`; section references (§) point
to that spec.

## Columns

| column | meaning |
|---|---|
| **Sev** | severity: **C** = critical, H = high, M = medium, L = low |
| **Detect** | how the failure mode would be noticed |
| **Mitigation** | what is fixed in the spec *before* results exist |
| **If triggered** | what happens when the risk materialises |

## Rule for critical risks

For every risk marked **C**, triggering it means **STOP and report to the
user**:
- It must not be silently patched.
- The algorithm, the hypotheses and the baselines must not be changed in
  response.

This applies to every critical failure condition in the stage instructions.

---

## 1. Budget and FE integrity

| id | failure mode | Sev | Detect | Mitigation | If triggered |
|---|---|---|---|---|---|
| R-FE-1 | hidden objective evaluations in the evidence, selection or diagnostic code | C | T-FE (raising guard); end-of-run FE audit (§5.5) | the objective handle is never passed to evidence modules | STOP; run invalid |
| R-FE-2 | probes or exchanges exceed the declared budget | C | `BudgetExceeded` is never caught; FE audit | stop rule (§5.4) | STOP |
| R-FE-3 | arms end with different unused FE, which biases the final error | M | `fe_unused` logged per run | bound < 3N + E_x (≤ 114 FE, ≈ 0.06% of CEC MaxFE); reported | report; no correction |
| R-FE-4 | the N1 vs MPHBS per-iteration cost differs (MPHBS spends 1 FE on Pbest; N1 does not) | L | declared (§5.7) | total budget is identical | report |
| R-FE-5 | oracle or ground-truth labels computed with objective calls | C | oracle code reads stored positions only; T-FE | offline only (§5 table) | STOP |

## 2. Leakage and prequential validity

| id | failure mode | Sev | Detect | Mitigation | If triggered |
|---|---|---|---|---|---|
| R-LK-1 | the CUR window influences the HB partition that scores it | C | T-PREQ (perturbing CUR leaves P unchanged) | partition from OLD only (§4.3) | STOP |
| R-LK-2 | p0 computed after observing y, or the native-control model fitted on data that includes the outcome it predicts | C | log-order assertion; T-PREQ | native-control model refitted at step 4, before exchanges; p0 logged before evaluation (§4.8) | STOP |
| R-LK-3 | exchange-accepted moves enter structural windows (self-confirmation of the chosen structure) | C | window records carry an `op` tag; assertion that no `op=EXCH` is present | only native operators are recorded (§3.2) | STOP |
| R-LK-4 | shadow selectors perturb A0 trajectories (RNG or state), which invalidates the passive E4 design | C | T-SHADOW (bit-identical A0) | RNG isolation (§9) | STOP |
| R-LK-5 | ground-truth labels (S6, S7 oracle) visible to the algorithm | C | code review; labels are computed in a separate offline script | §12 | STOP |
| R-LK-6 | the offline S7 oracle uses the same native-control model as the SPRT, so the agreement (C5) is partly circular | H | inherent | declared: C5 measures agreement of the sequential test with a batch hindsight estimate of the **same** quantity, not with an independent truth; S1–S4 and S7 outcomes are reported with this caveat | report as a limitation |

## 3. Statistical validity of the H0 test (E3)

| id | failure mode | Sev | Detect | Mitigation | If triggered |
|---|---|---|---|---|---|
| R-SPRT-1 | the SPRT is miscalibrated even under its own assumptions (implementation bug) | C | T-SPRT: empirical error > nominal + 3 Monte Carlo SE | unit test before experiments | STOP |
| R-SPRT-2 | dependence between outcomes within an iteration, and an estimated (not known) p0, make nominal α and β invalid in practice | H | empirical decision rates on S1–S4 vs oracle (C5) | declared approximate (§7); calibration reported | report; no retuning of α and β after results |
| R-SPRT-3 | native-control confounding: native and exchange successes differ for reasons other than informativeness (operator type, population age, receiver selection by tournament vs all agents) | H | compare the NA fit per operator; log the out-of-support fraction | step-length and population covariates (§4.7) | report; **no covariates are added after results** |
| R-SPRT-4 | exchange step lengths lie outside the native support, so the logistic model extrapolates | H | out-of-support fraction logged (§4.8) | clipping to [q1%, q99%] | if > 50% of exchanges are out of support in the smoke run → STOP and report (**C at smoke**) |
| R-SPRT-5 | complete separation or an all-failure NA buffer late in the run, giving degenerate p0 | M | `separation` flag | smoothed intercept-only fallback (§9) | report the frequency |
| R-SPRT-6 | the truncation rule (n_max) dominates decisions, so the test is effectively a fixed-sample rule | M | truncation fraction logged | n_max = 200 fixed | report; no change |
| R-SPRT-7 | never reactivates: when suppressed, 2 probes per iteration may be too few to detect recovery within the run | M | time-to-reactivation; S6 and oracle | r_probe fixed a priori | report as a limitation |
| R-SPRT-8 | oscillation between ACTIVE and SUPPRESSED (chattering) | M | number of state flips per run | reset policy; n_min | report |
| R-SPRT-9 | δ = log 2 is arbitrary; conclusions are sensitive to it | H | δ sensitivity on the synthetic suite only (§7) | pre-declared grid {log 1.5, log 2, log 3}; all reported | report; the primary δ is not changed |
| R-SPRT-10 | "informative" in the SPRT sense (success odds at matched step length) differs from "improves the final result" | H | C5 vs C6 comparison | declared distinction | report; not reinterpreted after results |

## 4. Structural evidence and model selection (E4)

| id | failure mode | Sev | Detect | Mitigation | If triggered |
|---|---|---|---|---|---|
| R-BIC-POWER | with W = max(60, 5D), the full model has D + D(D+1)/2 parameters (e.g. 230 at D=20 with n=100); the BIC penalty may make HR almost never selected even on S3 | **C at smoke** | T-BIC selection rates on Gaussian data; S3 accuracy in smoke | W fixed a priori | if HR accuracy on S3 in the smoke run is ≈ 0, **STOP and report**; do not change W, the criterion or the penalty without user approval |
| R-BIC-2 | successful displacements are not Gaussian (heavy tails from Lévy flights, MPA and FAD; boundary clipping); the likelihood is misspecified | H | Mahalanobis QQ diagnostics logged | declared; T-BIC on Gaussian data only establishes the implementation | report |
| R-BIC-3 | non-stationarity inside a window (the step scale shrinks during convergence) inflates apparent correlations | H | compare OLD and CUR scale; the PLL vs BIC gap | standardisation by the population spread (§4.1) | report; no change-point detector in v1 (§6.3) |
| R-BIC-4 | the covariance of **successful steps** reflects operator geometry (HBA/MPA move rules toward the best), not landscape structure | **C** (threatens the E4 premise) | A0 shadow accuracy on S1 (separable; HI expected) and S8 | none possible by design; this is exactly what C1–C3 test | if S1 is systematically labelled HR/HB in smoke → STOP and report |
| R-BIC-5 | the MPA elite-directed moves (all agents moving toward one elite) produce rank-1 displacement structure unrelated to the landscape | H | eigenvalue spectrum of `CUR_M` logged | centring the window; declared | report |
| R-BIC-6 | partition threshold α_link (Bonferroni) too conservative, making HB degenerate to HI on S2/S4 | H | adjusted Rand index; degeneracy frequency | fixed a priori | report; no retuning |
| R-BIC-7 | a dense-rotated function yields a single-block partition, so HB ≡ HR (degenerate); the candidate set changes over time | M | degeneracy flags | degenerate candidates are removed, never double-counted (§4.3) | report |
| R-BIC-8 | the JOINT selector wins only because it has 2× the samples (a sample-size advantage, not a heterogeneity benefit) | **H** (interpretation) | POOLED control; C4 | the POOLED control has the same sample count | report; E4 is interpreted only relative to POOLED and DECOUPLED |
| R-BIC-9 | the DECOUPLED placebo is not a clean placebo (column permutation also alters the joint Stouffer statistic in a way that biases toward HI) | M | reported with the C3 caveat | declared | report |
| R-BIC-10 | the shared-structure assumption fails because HBA and MPA genuinely see different structure (different regions, as in S7), so JOINT is penalised | M | agreement between HBA-only and MPA-only choices (logged) | reported | report |
| R-BIC-11 | hysteresis κ delays switches (S6 latency) or locks in an early wrong choice | M | switch latency; forced-switch count | κ fixed | report |
| R-BIC-12 | the warm-up period (fewer than W or 2W samples) covers a large fraction of the run at low success rates | M | fraction of no-evidence epochs reported (§6.4) | default HI during warm-up | if > 50% of epochs in smoke → report (not a patch trigger) |

## 5. Exchange mapping

| id | failure mode | Sev | Detect | Mitigation | If triggered |
|---|---|---|---|---|---|
| R-MAP-1 | mapping bug (the full mask does not give x_d) | C | T-MAP | unit test | STOP |
| R-MAP-2 | unequal expected transfer size across HI, HB and HR (block units move more coordinates) confounds the structure comparison | H | log the number of coordinates changed and ‖x' − x_r‖ | per-unit p_x fixed; declared | report; no rebalancing after results |
| R-MAP-3 | the HR eigenbasis from the receiver's Σ̂ is unstable when eigenvalues nearly coincide (arbitrary rotation within an eigenspace) | M | eigen-gap logged | declared; the full mask is still exact | report |
| R-MAP-4 | boundary clipping of x' distorts structured moves | L | clip frequency logged | same `clip` as the backbone | report |
| R-MAP-5 | rank-matched donors are the same pair repeatedly (low diversity of exchange) | L | unique-pair count logged | tournament for the receiver | report |

## 6. Baseline fairness and comparison validity

| id | failure mode | Sev | Detect | Mitigation | If triggered |
|---|---|---|---|---|---|
| R-BASE-1 | MPHBS altered (intentionally or by a refactor), so the reproduction is no longer valid | C | T-MPHBS; re-run of the reproduction with FE totals 199,930 / 149,857; comparison to `baseline_validation_results.md` | MPHBS code is frozen; N1 lives in `src/n1/` | STOP |
| R-BASE-ASYM | N1 uses a single backbone (FAD before) on all domains, while MPHBS uses domain configurations (A-C1 on FIR) | M | declared (§10, §11) | not engineered away | report as a limitation |
| R-BASE-2 | the A10 "ACoS-like" control is not ACoS; reviewers may read it as ACoS | M | naming | explicitly labelled a reimplementation of the idea (§11) | report |
| R-BASE-3 | synthetic S7 initialisation (basins per population) favours or harms particular arms | M | applied identically to all arms, including A4 | declared | report |
| R-BASE-4 | common random numbers diverge after the first exchange, so pairing is weaker than it appears | L | inherent | paired tests are still valid (the seed is the pairing unit) | report |
| R-BASE-5 | initial populations differ between A4 and the N1 arms | M | T-INIT | `rng_init = default_rng(seed)`, as in MPHBS (§15.1) | fix only by the spec rule; STOP if T-INIT fails |

## 7. Experimental-design and inference risks

| id | failure mode | Sev | Detect | Mitigation | If triggered |
|---|---|---|---|---|---|
| R-DES-1 | synthetic labels do not correspond to the structure the displacements can reveal (e.g. S5 multimodality; S8 isotropic in expectation) | H | S5 excluded from confirmatory tests; S8 prediction pre-declared | §12 | report |
| R-DES-2 | S7 "expected H0" is wrong (exchange between basins may help, e.g. via basin hopping) | H | oracle | the label is **validated by oracle, not assumed** | report the oracle outcome as is |
| R-DES-3 | S6 transition occurs too late or too early for any switch to be observable within budget | M | post-hoc transition time | r0 fixed | report; C7 counts "no transition" runs separately |
| R-DES-4 | multiple testing across many families inflates false positives | M | — | Holm per family; families fixed in §14.3 | — |
| R-DES-5 | the Wilcoxon normal approximation at n = 30 with ties | L | — | exact p-values where available; logged (§14.4) | — |
| R-DES-6 | performance differences on CEC2022/FIR are uninterpretable if the mechanism is inactive there (as happened with LA-KAIE) | H | exchange FE share, state trace, and selection logs on CEC/FIR | mechanism metrics are reported **alongside** performance | report; no claim about mechanism benefit without active-mechanism evidence |
| R-DES-7 | E4 passive data (A0) differ from the active regime (A7), so passive accuracy does not transfer | M | secondary active comparison A7 vs A5/A6 | §6.2 | report both |

## 8. Research-integrity risks

| id | failure mode | Sev | Detect | Mitigation | If triggered |
|---|---|---|---|---|---|
| R-INT-1 | hyperparameters changed after seeing full-test results | C | git freeze tags (§15.6); `DEVELOPMENT_CHANGES.md` | freeze protocol | STOP |
| R-INT-2 | functions, runs or seeds removed or selected | C | the run count per cell is asserted = 30 in the analysis script | §14.5 | STOP |
| R-INT-3 | the hypothesis definition changed after results | C | spec under git; freeze tag | §15.6 | STOP |
| R-INT-4 | a baseline modified to favour N1 | C | R-BASE-1 | frozen code | STOP |
| R-INT-5 | novelty claimed from an experimental improvement; components marketed as novel | C | review of generated reports | scope statement in the spec header | STOP; correct the report |
| R-INT-6 | manuscript prose written | C | — | not produced | STOP |
| R-INT-7 | proceeding past Stage 3 without user approval | C | — | STOP at `MECHANISM_SMOKE_REPORT.md` | STOP |
| R-INT-8 | smoke results (5 runs) used to tune anything | C | smoke seeds (`n1_smoke`) are disjoint from experiment seeds; smoke is for defects only | §15.1 | only a **defect** fix (a failing unit test) is allowed; it is logged and reported |
| R-INT-9 | a change-point detector added before the gate | H | §13.3 gate | — | STOP |

## 9. Engineering risks

| id | failure mode | Sev | Detect | Mitigation | If triggered |
|---|---|---|---|---|---|
| R-ENG-1 | runtime: BIC and eigendecompositions every U=5 iterations for 5 selectors × 2 populations; at D=31 the cost is small per epoch but multiplied over 30 runs × 13 arms × 20 instances | M | timing in the smoke run | U=5; one thread per run; parallel runs | report the estimated cost; ask before reducing the arm count |
| R-ENG-2 | non-determinism across machines (BLAS, NumPy version) | L | pinned environment; single thread | `capture_environment` | report |
| R-ENG-3 | log size (per-exchange records: ~24 × thousands of iterations × runs) | L | disk check | compressed `npz`/parquet per run | — |
| R-ENG-4 | CEC2022 library state shared across processes | L | existing validated wrapper | unchanged | — |

---

## Consistency checklist (spec ↔ register)

- [x] Every **C** risk has a detection mechanism defined in the spec (§5.5,
  §9, §13.1, §15.6) or in this register.
- [x] No parameter mentioned in the register is left undefined in spec §10.
- [x] Leakage risks R-LK-1…5 map to spec §4.3, §4.8, §3.2, §9 and §12.
- [x] The E4 interpretation risks R-BIC-8 and R-BIC-9 map to the POOLED and
  DECOUPLED selectors (spec §4.4) and tests C3 and C4 (§14.3).
- [x] "STOP and report" conditions at smoke level:
  - R-SPRT-4;
  - R-BIC-POWER;
  - R-BIC-4;
  - any failing T-* test.

  They are evaluated in `MECHANISM_SMOKE_REPORT.md`.
