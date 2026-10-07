# SHE Stage 1 — follow-up questions (existing code and data only; no new runs)

Date: 2026-10-07. Stage 1 gate = STOP; it is accepted, and this note does not change it.

## (a) Are the models nested? **No.**

**Code** (`lakaie/she/evidence.py::features`):

```python
if g == 0: [1, log1p(||z1||)]
if g == 1: [1, z1, z1*z1]
if g == 2: [1, z1, z1*z1] (+ z1[k]*z1[l] for (k,l) in E_t)
if g == 3: [1, z3, z3*z3]
```

**Spec** (`experiments/SHE_SPEC.md` §5):
- h0: [1, log(1+‖ζ⁰‖)]
- h1: [1, ζ¹_k, (ζ¹_k)²]
- h2: h1 vector plus [ζ¹_kζ¹_l for (k,l) ∈ E_t]
- h3: [1, ζ³_k, (ζ³_k)²]

**Consequence:**
- h1 and h3 do **not** contain h0's log(1+‖z‖) term. log(1+‖z‖) is not in the span of {1, z_k, z_k²}.
- h1 can represent ‖z1‖² (equal coefficients on z_k²), but not the log-magnitude shape.
- So the Stage 1 evidence compared different step-size predictors. It did not measure the
  incremental value of structure.

**Next step:** Amendment 2 (spec §25), which is pre-registered and not run.

## (b) Zero-variance coordinates in artefact windows: **not countable from stored data**

**What is stored:**
- per run, the time-averaged off-diagonal correlation matrices `A_H` and `A_M`;
- the window sizes `A_*_n` (43 windows at D=10, 87 at D=20 for every run);
- the windows themselves were not stored, and no per-window zero-variance count was logged.

**What can be established:**
- **No coordinate was zero-variance in every window.** None of the 360 runs has an exactly-zero
  off-diagonal entry in A_H or A_M.
- **Warnings did occur.** The campaign's stderr holds 38 warning lines = 19 occurrences. NumPy prints
  a given warning once per worker lifetime; there were 18–19 worker lifetimes (360 tasks,
  `maxtasksperchild=20`). So at least one occurred in most worker lifetimes, but the per-run and
  per-cell counts **cannot** be recovered.
- **Recomputing r without those windows is not possible** from stored data. It would need a
  deterministic replay of the 360 runs, which reproduces the same windows; that was not done (no new
  runs).
- **Whether H-E4(b) changes is not verified.**

**Plausibility (inference, not tested):**
- A zero-variance coordinate most plausibly arises late in convergence: S1 runs reach f = 0 exactly,
  so a coordinate sits at its optimum in every successful step.
- Setting those correlations to 0 shrinks |A| towards 0. That would shrink the MPA–HBA difference
  rather than create it. The observed r = 5.1 / 7.2 (MPA 0.65–0.70 vs HBA 0.07–0.11) is therefore
  unlikely to be an artefact of this.

Amendment 2 B-2 adds the missing per-window logging.

## (c) What existing results show about MPHBS's learned mediator

**Data:** `results/raw/{baseline_validation,pilot}`. Paired by seed; two-sided Wilcoxon per instance;
Holm across instances.

| campaign (runs/instance) | domain | comparison | median W/T/L (first vs second) | Holm-significant better / worse |
|---|---|---|---|---|
| baseline_validation (10) | CEC2022 | MPHBS vs MPHB | 7/0/5 | 3 / 0 |
| baseline_validation (10) | FIR | MPHBS vs MPHB | 8/0/0 | **8 / 0** |
| pilot (5) | CEC2022 | MPHBS vs MPHB | 6/0/6 | 0 / 0 |
| pilot (5) | CEC2022 | MPHBS vs LA-KAIE-FixedPolicy | 2/0/10 | 0 / 0 |
| pilot (5) | CEC2022 | MPHBS vs LA-KAIE-RandomExchange | 7/0/5 | 0 / 0 |
| pilot (5) | FIR | MPHBS vs MPHB | 8/0/0 | 0 / 0 |
| pilot (5) | FIR | MPHBS vs LA-KAIE-FixedPolicy | 7/0/1 | 0 / 0 |
| pilot (5) | FIR | MPHBS vs LA-KAIE-RandomExchange | 7/0/1 | 0 / 0 |

**Reading:**
- With 5 runs, the smallest two-sided Wilcoxon p is 0.0625, so **no pilot comparison can reach
  significance**. The pilot rows are descriptive only.
- **MPHBS vs MPHB** compares the mediator *and its configuration* (FAD placement, exchange
  evaluations) against **no exchange**. Learned and random mediation are not separated: our code has
  **no MPHBS-with-random-mediator variant**.
- **LA-KAIE-FixedPolicy and LA-KAIE-RandomExchange** are LA-KAIE exchange designs, not MPHBS mediator
  ablations. They also differ in FAD placement on FIR (LA-KAIE: before; MPHBS A-C1: after).
- **The paper's own learned-vs-random evidence** is from the full paper text, as summarised in
  `RESEARCH_DECISION_REPORT.md` §1.1: SARSA beats its matched RANDOM mirror on CEC2022 but not on FIR;
  B-C1 is statistically indistinguishable from MPHB and MPA on CEC2022; on FIR it wins all 8 cases but
  is not significant against MPHB after Holm.
- **Our 10-run FIR result** (MPHBS better than MPHB on 8/8 after Holm) uses a different test, run
  count and sample than the paper's. It does not contradict the paper's direction, but is not
  comparable to its significance statement.

**What existing data cannot answer:** whether MPHBS's *learning* adds value over random or fixed
mediation within MPHBS. That would need a new run with a random mediator (not done).

## Tag

`she-stage1-negative` was created on commit `ad47c82` **locally**. Pushing it was refused with
HTTP 403 by the session's git proxy, for both the ADHD-EEG-Benchmark remote and the la-kaie remote.
**The tag is not on GitHub.**
