# Pilot report (phases 0–5) — decision point before the full experiment

Generated 2026-09-28. Pilot = 5 paired runs per instance at the **full**
budgets (CEC2022: 12 functions, D = 20, 200,000 FEs; FIR: 8 cases, D = 31,
150,000 FEs), seeds from `master_seeds.pilot` (disjoint from the main
experiment). LA-KAIE uses the pre-registered default configuration
(`configs/controller.yaml`); no parameter was tuned. **With 5 runs, nothing
below is a confirmatory result**; Holm-corrected instance tests cannot reach
significance at this sample size. Tables: `results/tables/pilot/`,
statistics: `results/statistics/pilot/`, figures: `results/figures/pilot/`.

## Validation status

| phase | result |
|---|---|
| 0 static checks | compileall + pyflakes clean |
| 1 unit tests | 49 passed (exact FE budgets, budget guard, shift invariance, determinism, all variants) |
| benchmark validation | CEC2022: official C == official Python (≤1.7e-16), F(o) = F*; FIR: FFT == direct DTFT, analytic checks — PASS |
| baseline validation | 0/80 distributional differences vs the authors' published MATLAB runs (Holm); FE totals identical to the authors' audit |
| 2–3 smoke | 125/125 runs ok |
| 4–5 pilot | 720 + 520 runs ok; `validate_experiment.py --campaign pilot` PASS |

## CEC2022 (anchor LA-KAIE-Full)

| competitor | instance W/T/L (Full's view) | avg rank Full vs competitor |
|---|---|---|
| HBA | 11/0/1 | 3.42 vs 4.67 |
| MPA | 2/0/10 | 3.42 vs 2.08 |
| MPHB | 3/0/9 | 3.42 vs 2.67 |
| MPHBS (B-C1) | 3/0/9 | 3.42 vs 2.17 |

Ablation stage (8 variants): Friedman p = 0.007, W = 0.23. Best variants:
**FixedPolicy** (always A4) avg rank 2.58, **NoAdaptiveController** 3.21,
RandomExchange 3.83, Full 4.58. Over all 12 methods FixedPolicy has the best
average rank (3.50), ahead of MPA (4.42) and MPHBS (4.83); Full is 7.00.

**Reading:** on CEC2022 the learned controller is harmful relative to simply
performing 24 cross-population exchanges per iteration.

## FIR (anchor LA-KAIE-Full)

| competitor | instance W/T/L (Full's view) | avg rank Full vs competitor |
|---|---|---|
| HBA | 8/0/0 | 1.63 vs 4.00 (post hoc Holm p = 0.008) |
| MPA | 8/0/0 | 1.63 vs 3.38 |
| MPHB | 8/0/0 | 1.63 vs 4.38 (post hoc Holm p = 0.002) |
| MPHBS (A-C1) | 3/0/5 | 1.63 vs 1.63 (run level 21/0/19) |

Ablation: learned controller beats FixedPolicy (Holm p < 0.001),
NoAdaptiveController (p = 0.009) and RandomExchange (8/0/0, n.s.); but
NoKnowledgeReward (avg rank over all methods 3.00), DimensionOnly (3.38) and
NoLandscape (3.50) rank ahead of Full (4.38). Supplementary gray-box
**FIRPrior** beats Full on 6/8 cases (1.25 vs 1.75, n.s.).

## Mechanism diagnostics (why)

| | CEC2022 Full | FIR Full |
|---|---|---|
| final S8 (interaction strength) | 0.008 | 0.057 |
| variables left as singleton groups | 19.9 / 20 | 30.4 / 31 |
| group-level transfers | 15 % | 2 % |
| dominant action | A8 local refinement 35 % | A8 local refinement 73 % |
| exchange evaluations per iteration | 8.0 of 24 | 10.5 of 24 |

* The black-box interaction estimator finds almost no significant couplings
  during the search. A diagnostic with ideal isotropic samples around the
  true optimum (never available to the algorithm) shows it *can* detect
  coupling on smooth rotated functions (F1: up to 27 % of pairs), but not on
  the multimodal rotated functions (F4, F5, F10), where local ruggedness
  dominates any quadratic model. The group-preserving mechanism — intended as
  the main novelty — is therefore nearly inactive on both benchmarks, and
  the ablations that remove it (DimensionOnly, NoInteraction on CEC) do not
  lose performance.
* On FIR the benefit of LA-KAIE over the backbone comes mainly from A8
  (elite-covariance local refinement) chosen by the controller, not from
  interaction-group exchange.
* On CEC2022 the controller under-uses exchange (A1 ends the phase early,
  8 of 24 evaluations used) and over-uses A8; the fixed policy that always
  exchanges does better.

## Open issues requiring a decision (STOP conditions §26 / §31)

1. **Novelty overlap** (`literature/NOVELTY_ASSESSMENT.md`): group-preserving
   exchange with learned interactions exists (RV-GOMEA, RVIntX, VIGPSO);
   landscape-aware contextual-bandit operator selection exists (LA-BHH,
   RL-HPSDE, DE-DDQN).
2. **The claimed main mechanism is empirically inactive** in the pilot, and
   the pre-registered design does not outperform MPHBS (CEC2022: worse;
   FIR: tie).

The full 30-run experiment has **not** been started. Any design change made
now would be made after seeing pilot results on the test benchmarks; to limit
test-set overfitting, a redesign should be developed and selected on the
disjoint tuning set (CEC2022 D = 10) only, recorded in
`logs/DEVELOPMENT_CHANGES.md`, and then evaluated once in the main
experiment, with this pilot reported as exploratory.
