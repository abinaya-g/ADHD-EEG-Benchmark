## Diagnosis of the P4 failure (written 2026-09-29; manual analysis, appended to the generated report)

### Discrepancy
- One of 12 CEC2022 instances differs: **F10** (Composition Function 2), MPHBS B-C1.
- Holm-corrected Mann–Whitney p = 0.020.
- Medians: ours 100.42, authors 10.24.
- All other CEC instances (Holm p ≥ 0.965) and all 8 FIR cases agree.
- P1 and P2 (status, exact FE totals 199,930 / 149,857) pass everywhere.

### What the data show
- F10 is **bimodal**. Runs either escape to a basin with error of roughly 0.3–20, or stall
  in a local basin at error ≈ 100.3–100.6.
- The difference is in the **escape rate** (final error < 50):

  | source | escape rate |
  |---|---|
  | ours (these 30 runs + 10 earlier runs) | 8 / 40 = 20% |
  | authors | 17 / 30 = 57% |

  Fisher exact two-sided p = 0.0023.
- The earlier `baseline_validation` comparison (10 runs, p = 1 after Holm) lacked the power
  to detect this. Those 10 runs show the same low escape rate (3/10).
- The backbone-only variants **do not** show it:

  | variant | our escape rate | authors' escape rate |
  |---|---|---|
  | MPHB (HBA + MPA + FAD, no mediator) | 0/10 | 0/30 |
  | MPA | 0/10 | 0/30 |
  | HBA | 0/10 | 0/30 |

- The authors' MPHBS is the **only** algorithm in their table that escapes on F10. The other
  nine algorithms escape in 0/30 runs.

### Diagnosis (not yet confirmed)
- The difference is most probably located in the **MPHBS Phase 2 mediator**. This is the
  SARSA-style dimension-wise exchange, including its acceptance rule and the injection of
  the global best into both populations. It is not in the shared `HybridState` backbone.
  Candidates, none yet verified:
  1. a semantic mismatch in the mediator (e.g. the ε-decay schedule, the reward, or the
     probabilistic acceptance of worse states, `min(p, 0.25)`);
  2. a difference in how improved global bests are written back to the populations;
  3. a difference between the published MATLAB version and the version that produced
     `task_results.csv`.
- Chance alone is unlikely (p = 0.0023 on the escape rate), but it cannot be fully excluded:
  the test was chosen after seeing the data. The pre-registered test is the
  Holm-corrected Mann–Whitney, p = 0.020.

### Harmless or blocking?
- **Blocking** for:
  - any claim that A4 is a faithful MPHBS reproduction on CEC2022;
  - any N1-vs-MPHBS comparison on CEC2022 (Stages 5–6).
- **Not evidence against the N1 backbone.** N1 uses `HybridState.phase1` and `fads_greedy`
  only, and MPHB (exactly that backbone plus the Pbest synchronisation) matches the authors
  on all 12 CEC instances and on FIR (earlier validation: 0/80 differences).
- Under the stage rules, Stage 0 is nevertheless recorded as **NOT PASSED**. The next
  stage requires a user decision.
- **Nothing was patched.**

### Proposed next step (requires approval)
- A line-by-line audit of `lakaie/algorithms/mphbs.py` Phase 2 against `MPHBS_main.m`
  (`third_party/mphbs_reference/`), with an F10 diagnostic: escape rate per configuration
  on the n1_unit seeds.
- Any fix would be a **correctness fix to the port**, justified by a code mismatch and
  covered by a test. It would not be justified by performance.
- After any fix, Stage 0 would be re-run in full.
