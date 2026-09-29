# T-SPRT calibration report (spec §7.2)

This report gives the **empirical behaviour** of the SPRT-style prequential sequential likelihood-ratio decision rule. **No classical Wald type-I/type-II guarantee is claimed.** The nominal α = 0.05 and β = 0.05 only set the boundaries.

## Level 1: semi-synthetic replay (0 FE)

- Source: the logged, online-estimated p0 sequences of 80 A7 smoke runs.
  - Spec §7.2 names A0 as the source, but A0 has no exchange candidates. This is a documented deviation.
- Outcomes are simulated. Each p0 sequence is replayed 20 times per condition.
- The ± value is the binomial Monte Carlo SE over decisions. Decisions within a sequence are dependent, so this SE is optimistic.

| condition | rate ± SE | decisions | stopping n: median [IQR] | p95 | truncation |
|---|---|---|---|---|---|
| null_sigma0.0 (false activation) | 0.0660 ± 0.0007 | 138392 | 74 [45, 128] | 200 | 0.124 |
| null_sigma0.5 (false activation) | 0.1470 ± 0.0009 | 143766 | 69 [40, 126] | 200 | 0.128 |
| positive_sigma0.0 (false suppression) | 0.0785 ± 0.0007 | 147468 | 68 [39, 120] | 200 | 0.112 |
| positive_sigma0.5 (false suppression) | 0.1011 ± 0.0007 | 170600 | 55 [30, 101] | 200 | 0.083 |

Additional Level 1 quantities:

- null_sigma0.0: {'frac_sequences_ending_active': 0.1}
- null_sigma0.5: {'frac_sequences_ending_active': 0.15375}
- positive_sigma0.0: {'frac_outcomes_while_suppressed': 0.12855558537961215}
- positive_sigma0.5: {'frac_outcomes_while_suppressed': 0.12926808117372351}

## Level 2: pipeline harness (test fixtures, not N1 variants)

- Arms: H_null (native-equivalent exchange candidates) and H_pos (donor = known optimum).
- Problems: S1 and S3, D ∈ {10, 20}, 5 runs each.

| condition | rate ± SE | decisions | stopping n: median [IQR] | p95 | truncation |
|---|---|---|---|---|---|
| null (false activation) | 0.0000 ± 0.0000 | 1951 | 28 [21, 38] | 66 | 0.000 |
| positive (false suppression) | 0.2469 ± 0.0067 | 4139 | 24 [20, 45] | 152 | 0.032 |

Additional Level 2 quantities:

- null: {'frac_runs_ending_active': 0.0, 'n_runs': 20}
- positive: {'frac_iterations_suppressed': 0.8042813364876006, 'n_runs': 20}

## Gross-miscalibration check

- The criterion is Level 1 with σ_u = 0: false activation 0.0660 and false suppression 0.0785, each against the limit 3 × 0.05 = 0.15.
- **Result: not grossly off.**
- Caveat on the Level 2 harness: exchange receivers are chosen by tournament, so they are fitter than average.
  - Under the null, a native-equivalent step from a fitter parent succeeds *less* often than the average native attempt at the same step length.
  - The harness null is therefore conservative toward suppression.
  - This is reported as-is.
