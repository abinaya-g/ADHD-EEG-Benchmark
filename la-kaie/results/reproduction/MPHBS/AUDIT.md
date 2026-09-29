## Mediator audit (2026-09-29; approved by the user: "Audit mediator first")

### 1. Line-by-line comparison: `lakaie/algorithms/mphbs.py` vs `MPHBS_main.m` (source commit b3e96ca)

**No semantic mismatch was found** in any of:
- the Phase 1 HBA / MPA movement (1B), the HBA refresh (1C), the MPA evaluation and memory
  (1D), and FAD before Phase 2 (1E);
- the global best before Phase 2;
- the elite + random subset and the in-subset ranking (stable sorts);
- the closest-action projection;
- the block schedule of active dimensions;
- the K = 7 candidate policies: greedy (first max, `m1 >= m2`), ε-decay (decay
  `max(1, E/4)`), random, second-best (stable descending sort), and look-ahead ε-decay
  `min(E, ep + randi(1..max(1, floor(E/10))))`;
- the proxy score (1-based rank, novelty, distance to the *current* `Best_P`) and its first-max
  winner;
- the reward;
- the relaxed acceptance probability `min(p, 0.25)` (the random number is drawn only when the
  candidate does not improve);
- the SARSA update on active dimensions (target `r + γ Q'`, next action only if accepted);
- the subset-memory replacement;
- the global-best injection (in both codes it is overwritten by the subsequent subset
  reintegration);
- reintegration, best synchronisation, and the `fit_old` / `X_MPA_old` handling (not
  re-synchronised after Phase 2 in either code);
- the schedules (CF, α, prog_global).

**No code was changed.**

### 2. Additional evidence

| sample (F10, final error < 50 = "escape") | escape rate |
|---|---|
| authors B-C1: **one** set of 30 seeded runs, repeated identically in their external, sensitivity and ablation tables | 17/30 (57%) |
| authors, neighbouring SARSA configurations B-C2, B-C3, B-C4 | 10/30, 14/30, 7/30 (34% pooled) |
| authors, RANDOM mirrors (RB-C1…C4, BR-C1, R-best) | 1–7/30 (≈ 13%) |
| ours B-C1: 10 earlier runs + 30 Stage 0 runs + 30 diagnostic runs (n1_unit seeds) | 3/10, 5/30, 10/30 → 18/70 (26%) |
| ours B-C1 with `random_actions` (NOT identical to the authors' mirror, which also drops the Q-table and the closest-action projection) | 14/30 |

Tests:
- ours 18/70 vs authors 17/30: Fisher p = 0.0055.
- ours 18/70 vs the authors' neighbouring SARSA configurations 31/90: Fisher p = 0.30.
- These Fisher tests were chosen after seeing the data. The pre-registered test is the Stage 0
  Mann–Whitney (Holm p = 0.020).

### 3. Conclusion (honest status)
- The audit **did not identify a code defect**, so there is nothing to fix under the
  "correctness fix only" rule.
- The F10 difference remains **unexplained**. Our escape rate lies within the range of the
  authors' neighbouring SARSA configurations, but below their single B-C1 sample.
- Possible explanations that cannot be settled without MATLAB (bitwise reproduction is
  impossible across RNGs):
  1. the authors' B-C1 sample is at the high end of its sampling distribution;
  2. an effect of RNG-dependent or floating-point behaviour on this bimodal function;
  3. an undocumented difference in the authors' evaluation harness.
- Our port's SARSA component shows **no advantage over our random-action variant on F10**.
  This limits any claim that A4 reproduces MPHBS's SARSA benefit on this function.

### 4. Classification
- Stage 0 remains formally **NOT PASSED** (P4, CEC2022 F10). The other 11 CEC instances,
  all 8 FIR cases, and all FE totals pass.
- **Not blocking** for Stages 3a/3b/4, which are synthetic and do not use MPHBS. The N1
  backbone (MPHB) matches the authors.
- **Must be carried forward** into any CEC2022 comparison involving A4 (Stages 5–6):
  - report F10 with this caveat;
  - report the authors' published F10 values alongside ours.
- Proceeding requires the user's decision.
