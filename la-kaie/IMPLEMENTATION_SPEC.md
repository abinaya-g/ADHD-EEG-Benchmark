# IMPLEMENTATION_SPEC — Revised N1

Working name: **N1**. Full name: *statistical structural evidence for
information exchange between heterogeneous optimizers*.

Status: **specification only.** No N1 code has been written, and no N1
experiment has been run. This document fixes every definition and
hyperparameter *before* any N1 result exists. Any later change must be
logged in `logs/DEVELOPMENT_CHANGES.md` with:
- the date,
- the reason,
- the stage at which the change was made.

Changes made after full-test results have been seen are forbidden (see §15.6).

Revision: **rev. 2** (corrections from the user's specification review;
see `SPEC_REVIEW_REPORT.md`). Rev. 2 changes terminology, analysis
priorities, benchmark roles and gates. The core algorithm of rev. 1 is
preserved.

**Implementation status: BLOCKED.**
- Implementation may start only after both of the following:
  1. the literature-verification gate (§16, Gate 0) is separately
     resolved;
  2. this revised specification has passed independent review.
- No code, test run, smoke run or experiment is permitted before then.

Scope statement (not a claim):
- N1 is **not** claimed to be novel. The novelty classification remains
  **B — "promising, but important unresolved overlap/evidence gaps
  remain"** (`N1_FINAL_NOVELTY_DECISION.md`).
- It is not upgraded to "novel", "validated novelty", "publication-ready" or
  any stronger status.
- The absence of a direct duplicate in the currently accessible literature
  is **not** evidence of novelty.
- The following are standard building blocks and are **not** presented as
  contributions:
  - BIC
  - covariance estimation
  - eigen-coordinate transformation
  - linkage learning
  - adaptive coordinate selection
  - transfer-probability adaptation
  - negative-transfer suppression
  - model selection by itself
- The object under study is narrower. It asks two things:
  1. **E3:** is the outcome of exchange informative, relative to a
     native-search control?
  2. **E4:** does joint HBA+MPA structural evidence provide information
     beyond single-optimizer evidence **and beyond pooled evidence from the
     same observations**?
     - JOINT has about twice the observations of either single-optimizer
       selector. JOINT > HBA-only / MPA-only is therefore **not
       sufficient** for E4 (risk R-E4-SAMPLE).
     - The intended evidence pattern is:
       **JOINT > POOLED and JOINT > HBA-only and JOINT > MPA-only**.
       Only this pattern supports "optimizer-specific heterogeneous
       evidence contributes beyond sample size".
     - **JOINT vs POOLED is the primary E4 control** (§14.6).
     - E4 results are not converted into a novelty claim.

Notation:

| symbol | meaning |
|---|---|
| D | dimension |
| N | population size per optimizer |
| t | iteration index |
| FE | objective evaluations |
| H | HBA population |
| M | MPA population |
| lb, ub | box bounds |
| σ(·) | logistic function |
| logit | inverse of σ |

---

## 1. Final algorithm definition

N1 is a two-population hybrid. The two populations are:
- an HBA population, and
- an MPA population (with FAD).

The **native backbone is inherited unchanged** from the validated MPHB/MPHBS
port: `lakaie.algorithms.hybrid_common.HybridState`, which provides
`phase1` and `fads_greedy`.

Only the information-exchange stage (MPHBS "Phase 2") is replaced, by a
two-stage decision:

- **Stage 1 (E3, H0 test):** an **SPRT-style prequential sequential
  likelihood-ratio decision rule** (short name: "SPRT-style rule") decides
  whether exchange is currently informative.
  - It borrows Wald's LLR accumulation and boundaries.
  - Classical Wald error guarantees are **not** assumed to apply (§7.1).
  - "Informative" means that exchange candidates succeed more often than a
    native-search control predicts, at matched step length.
  - The state is **ACTIVE** or **SUPPRESSED**.
- **Stage 2 (structure selection):** used while ACTIVE, and for probes while
  SUPPRESSED.
  - The exchange representation is chosen among:
    - HI (independent coordinates),
    - HB (blocks),
    - HR (rotated eigen-directions).
  - The choice is made by BIC over **native successful displacements**.
  - Displacements from both optimizers are combined as a product of
    likelihoods: a shared structure, with parameters specific to each
    population.

The hypothesis set is therefore H0 ∪ {HI, HB, HR}. Selection is
hierarchical:
- H0 is decided by Stage 1 only.
- The structure is decided by Stage 2 only.

### 1.1 Iteration loop (one iteration t)

```
0. plan:     e_t = E_x if state == ACTIVE (or arm ignores H0) else r_probe   (A0: e_t = 0)
             stop if FE_remaining < 3N + e_t
1. native:   HybridState.phase1(obj, rng_native, prog_t, prog_T)            # 2N FE
             -> log native attempts + successes (HBA, MPA)
2. native:   HybridState.fads_greedy(obj, rng_native, CF)                   # N FE
             -> log native attempts + successes (FAD)
3.           HybridState.global_from_bests()
4. control:  refit native-control logistic model on native-attempt buffer   # 0 FE
5. evidence: append standardized successful displacements to windows        # 0 FE
             if t % U == 0: recompute BIC for all selectors (driving + shadow), apply hysteresis
6. exchange: for e = 1..e_t:
                build candidate x' by the mapping of the driving structure k*
                predict p0 = p_nat(l_e, pop_r)  (logged BEFORE evaluation)
                f' = obj(x')                                               # 1 FE each
                y = 1[f' < f(x_r)]; greedy replacement; SPRT-style LLR update (prequential)
7.           HybridState.sync_global_keep(); Recorder.iteration(...)
```

Progress variable: `prog_t / prog_T = FE_used / MaxFE`, where

```
prog_T = T_ref = floor(MaxFE / (3N + E_x))
prog_t = T_ref · FE_used / MaxFE
```

As a result, the HBA/MPA schedules (the MPA phase thresholds, CF, and the
HBA α) depend on consumed FE, not on the iteration count. An arm that
suppresses exchange runs more native iterations at the same schedule
position.

---

## 2. State variables

| variable | type | initial | updated |
|---|---|---|---|
| `X_H, F_H` | N×D, N | uniform in box (`rng_init`) | phase1, exchange |
| `X_M, F_M, X_M_old, fit_old` | as in HybridState | as in HybridState | phase1, FAD, exchange (memory sync) |
| `Pg, Pbest` | global best | `global_from_bests` | steps 3, 7 |
| `state` ∈ {ACTIVE, SUPPRESSED} | enum | ACTIVE | step 6 (SPRT-style decision) |
| `LLR, n_sprt` | float, int | 0, 0 | step 6; reset after each decision |
| `k*` (driving structure) | {HI, HB, HR} | HI (default until evidence available, §6.4) | step 5, every U iterations |
| `k*_s` (shadow structures) | per selector s ∈ {HBA, MPA, JOINT, POOLED, DECOUPLED} | HI | step 5 |
| `P_s` (partition, per selector) | list of index blocks | None | step 5 |
| `Σ̂_{s,pop}` | D×D per selector, per population | None | step 5 |
| `scale s ∈ R^D` | per-coordinate scale | computed at recording time | step 5 |
| `FE_used` | int | 2N (initialisation) | every evaluation (global counter) |
| `t` | int | 0 | per iteration |

The global FE counter is the single `CountedObjective` instance. No other
counter exists.

---

## 3. Data structures

1. **Native-attempt buffer `NA`.**
   - A FIFO ring of the last `W_nat` native attempts (§10).
   - Each record holds `(t, pop, op, l, y)`:
     - `pop` ∈ {H, M};
     - `op` ∈ {HBA, MPA, FAD};
     - `l = log(‖(x_cand − x_parent) / s‖₂ + ε_l)`;
     - `y = 1[f_cand < f_parent]`.
   - For MPA, the parent is the memory position that the move is compared
     with in `memory_saving`.
2. **Structural windows `Z_pop`, pop ∈ {H, M}.**
   - A FIFO ring of the last `2W` standardized successful native
     displacements `z = (x_new − x_old) / s`. The scale s is frozen at the
     time of recording.
   - The ring is split into:
     - **OLD** = the older W rows;
     - **CUR** = the newer W rows.
   - Only native operators contribute. Exchange-accepted moves **never**
     enter `Z_pop`; this prevents self-confirmation.
   - FAD-accepted moves are stored in `Z_M` with tag `op=FAD`. A flag
     `include_fad` (default True) controls whether they are used.
3. **Decoupled placebo window `Z̃_M`.**
   - A copy of `Z_M` in which each column is permuted independently.
   - The permutations come from `rng_placebo`, and are redrawn at each
     selection epoch.
   - This preserves the marginals and destroys cross-coordinate dependence.
4. **Exchange log.** Per exchange, the record holds:
   - `(t, e, pop_r, idx_r, idx_d, k*, mask/units, l_e, p0, p1, f_r, f', y, LLR_after, state)`.
5. **Selection log.** Per selection epoch and per selector, the record
   holds:
   - `BIC_k` and log-likelihood `ℓ_k` for each available k;
   - parameter counts `q_k`;
   - the partition;
   - the predictive log-likelihood `PLL_k`;
   - the chosen k before and after hysteresis;
   - the margin (the gap between the best and second-best BIC);
   - ΔBIC of every candidate relative to HR, and of HR relative to the
     best (R-BIC-POWER diagnostics);
   - window sizes (the number of samples used for structural fitting, per
     population);
   - the degeneracy flags;
   - at each selection epoch, the OLD/CUR window contents (standardized
     displacements, float32). These are needed for the offline JOINT-HALF
     analysis (§4.4) and for the BIC-power diagnosis (§16, Gate 7).
6. **Recorder.** The existing `lakaie.core.Recorder`, plus per-phase FE
   counts (`fe_phase1`, `fe_fad`, `fe_exchange`, `fe_probe`).

---

## 4. Exact update equations

### 4.1 Scale vector (standardization)

At recording time t:

```
s_j = max( std_j( [X_H; X_M] ),  s_floor · (ub_j − lb_j) )
```

- `std_j` is the population standard deviation of coordinate j over both
  populations (2N points), with ddof = 0.
- The same s_j is used for:
  - displacements recorded in iteration t;
  - the step length l of native attempts;
  - the exchange mapping (§8).

### 4.2 Gaussian models on a window

Let Z ∈ R^{n×D} be a window, with mean z̄ and scatter matrix
`S = (1/n) Σ (z_i − z̄)(z_i − z̄)ᵀ`, which is the MLE.

Common eigenvalue floor for every model:

```
ε_Σ = ε_rel · tr(S)/D
```

| model | covariance Σ̂_k | free parameters q_k |
|---|---|---|
| HI | `diag(max(S_jj, ε_Σ))` | D (mean) + D |
| HB (partition P = {B_1..B_m}) | block-diag of `S[B_b,B_b]`; eigenvalues of each block floored at ε_Σ | D + Σ_b |B_b|(|B_b|+1)/2 |
| HR | S with eigenvalues floored at ε_Σ | D + D(D+1)/2 |

Log-likelihood:

```
ℓ_k(Z) = −(n/2) [ D log 2π + log det Σ̂_k + tr(Σ̂_k^{-1} S) ]
```

BIC:

```
BIC_k = −2 ℓ_k + q_k log n
```

- log det is computed by Cholesky of the floored matrix.
- The floor applies only when the corresponding eigenvalue is below ε_Σ. The
  log records whether it was applied.

**Pre-registered formulation: unchanged in rev. 2.**
- The BIC above, with the parameter counts q_k in the table, is the
  pre-registered formulation for the smoke test.
- For D = 20, HR has q = 20 + 210 = 230 free parameters. With n = W = 100
  per population, the penalty is 0.5·230·ln 100 ≈ 530 nats per population
  (BIC units: 2 × 530).
- This complexity penalty may systematically disfavour HR (risk
  R-BIC-POWER). **S3 is therefore a hard diagnostic gate (§16, Gate 7).**
- The penalty, parameter count, window size and model definitions are not
  changed at this stage.
- If Gate 7 fails, any change requires diagnosis, a documented proposal and
  user approval. A change is never made after full benchmark results
  exist.

### 4.3 Block partition (prequential; no leakage)

- The partition used to **score** HB on CUR is estimated **only from OLD**.
  OLD and CUR are disjoint, and OLD is strictly earlier in time.
- CUR never influences the partition that it scores. This holds for all
  selectors.

Construction:
1. Compute the correlation matrix R of OLD.
2. Apply the Fisher transform `ζ_ij = atanh(clip(r_ij, −0.999999, 0.999999)) · sqrt(n_old − 3)`.
3. Draw an edge (i, j) iff `|ζ_ij| > z_{1 − α_link / (2·D(D−1)/2)}`
   (two-sided test with Bonferroni correction over all pairs).
4. P = the connected components of the edge graph.

This procedure is deterministic, uses no randomness and requires no FE.

Degeneracy:
- P consists only of singletons → HB ≡ HI; HB is marked `degenerate=HI`.
- P is a single block → HB ≡ HR; HB is marked `degenerate=HR`.

A degenerate HB is **not** a separate candidate. It is removed from the
candidate set, so that the same model is never counted twice.

**Joint partition.**
- The test statistic becomes the sample-size-weighted Fisher z. With
  population-specific correlations, it is

  ```
  ζ_ij = (sqrt(n_H − 3) atanh r^H_ij + sqrt(n_M − 3) atanh r^M_ij) / sqrt(2)
  ```

  (Stouffer combination).
- The threshold is the same as in step 3.

### 4.4 Selectors (all computed in every run; zero FE; zero RNG from the native stream)

| selector | observations | structure / parameters | partition source | score | role in E4 |
|---|---|---|---|---|---|
| HBA-only | `CUR_H` (n = W) | one model | `OLD_H` | `BIC_k(CUR_H)` | single-optimizer comparator |
| MPA-only | `CUR_M` (n = W) | one model | `OLD_M` | `BIC_k(CUR_M)` | single-optimizer comparator |
| JOINT | `CUR_H` and `CUR_M` (n = 2W) | **shared structural hypothesis k and shared partition; population-specific mean and covariance (population-specific likelihood)** | joint Stouffer on `OLD_H`, `OLD_M` | `BIC_k^J = BIC_k(CUR_H) + BIC_k(CUR_M)` | method under test |
| POOLED | `[CUR_H; CUR_M]` concatenated (n = 2W, **exactly the same observations as JOINT**) | **one common pooled structural model**: a single mean and covariance; no optimizer-specific parameters or evidence | pooled `[OLD_H; OLD_M]` | `BIC_k` on the pooled window | **PRIMARY E4 sample-size control** |
| DECOUPLED | `CUR_H` and `C̃UR_M` (column-permuted; n = 2W) | as JOINT | joint Stouffer on `OLD_H`, `ÕLD_M` | as JOINT | placebo: no shared cross-coordinate evidence from MPA |
| JOINT-HALF (offline analysis only; never drives) | the newest W/2 rows of `CUR_H` and of `CUR_M` (n = W), with the matching newest W/2 of each OLD | as JOINT | as JOINT, on the halved OLD | as JOINT | sample-matched to the single-optimizer selectors (E4 Q5) |

JOINT and POOLED:
- The JOINT score is the BIC of the product likelihood. Its parameter count
  is the sum of the population-specific counts, and log n uses each
  population's own n. This is equivalent to summing the two BICs for a
  shared structure label and partition.
- POOLED sees exactly the same observations as JOINT. It differs only by
  removing the optimizer-specific parameterisation.
- Any JOINT advantage over POOLED therefore cannot be explained by the
  number of observations. That is why JOINT vs POOLED is the primary E4
  control.

JOINT-HALF:
- It is computed offline from the logged windows (§3.5) after the run.
- It uses no FE and no RNG, and has no influence on the run.
- It is an **analysis device**, not an algorithmic mechanism.

### 4.5 Selection rule and hysteresis

Let:
- K_s be the available non-degenerate candidates of selector s;
- `k̂ = argmin_{k∈K_s} BIC_k`. Ties within 1e-9 go to the smaller q_k.

Switching rule:
- If `k̂ ≠ k_cur` and `BIC_{k_cur} − BIC_{k̂} > 2κ`, set `k_cur ← k̂`.
  (A ΔBIC of 2κ equals a log-evidence of κ.)
- If `k_cur` is no longer available (degenerate or unavailable), set
  `k_cur ← k̂` unconditionally. The log records this as a forced switch.

The rule is evaluated every U iterations, and only when the window is full
(§6.4).

### 4.6 Predictive log-likelihood (logged only; not used for selection in v1)

For each k:
1. Fit the mean, Σ̂_k and (for HB) the partition, all on OLD only.
2. Score on CUR:

   ```
   PLL_k = Σ_{z∈CUR} log N(z; z̄_OLD, Σ̂_k^OLD)
   ```

CUR is only scored; it never contributes to any fitted quantity. PLL is
available once 2W samples exist.

### 4.7 Native-control model (Stage 1)

This is a logistic regression on the NA buffer:

```
logit p_nat(y=1 | l, pop) = θ_0 + θ_1 · l + θ_2 · 1[pop = M]
```

- It is fitted by L2-regularized IRLS, with penalty λ_nat on θ_1 and θ_2 and
  none on θ_0.
- At most 25 IRLS iterations; tolerance 1e-8.
- It is refitted every iteration at step 4, using attempts observed up to
  and including this iteration's native phase. These are all observed
  before any exchange of iteration t is predicted.

### 4.8 SPRT-style prequential sequential likelihood-ratio decision rule (Stage 1)

Terminology:
- "SPRT-style" means that the rule uses Wald's log-likelihood-ratio
  accumulation and Wald's boundary formulas.
- It does **not** mean that Wald's error guarantees hold; §7.1 explains
  why.

For an exchange (or probe) candidate with receiver population pop_r and
standardized step length

```
l_e = log(‖(x' − x_r)/s‖₂ + ε_l)
```

the steps are:

1. Clip l_e to `[q_0.01, q_0.99]` of l in NA. The log records the fraction
   of values out of support.
2. `p0 = clip(p_nat(l_e, pop_r), p_clip, 1 − p_clip)`.
3. `p1 = σ(logit(p0) + δ)`.

Both p0 and p1 are logged **before** f(x') is evaluated.

Observe `y = 1[f(x') < f(x_r)]`. Then:

```
LLR ← LLR + y·log(p1/p0) + (1−y)·log((1−p1)/(1−p0)),   n_sprt ← n_sprt + 1
A = log((1−β)/α)     B = log(β/(1−α))
```

Decisions are taken only when `n_sprt ≥ n_min`:

| condition | decision | action |
|---|---|---|
| `LLR ≥ A` | INFORMATIVE | state ← ACTIVE |
| `LLR ≤ B` | UNINFORMATIVE | state ← SUPPRESSED |
| `n_sprt = n_max`, neither bound reached | truncated | `LLR ≥ (A+B)/2` → INFORMATIVE, else UNINFORMATIVE; flagged `truncated=True` |

After every decision: `LLR ← 0`, `n_sprt ← 0` (reset policy `reset_after_decision`).

A change of state takes effect at the next iteration's plan step (step 0).
Exchanges already planned in the current iteration are completed.

H1 in this rule means "the log-odds of exchange success exceed the
native-control log-odds by δ". H0 means "they equal it". Negative transfer
(an offset below 0) is absorbed by H0: the LLR drifts toward B.

Components of the rule (documented as required):

| component | definition |
|---|---|
| p0 estimation | online, prequential: the logistic native-control model (§4.7), refitted each iteration on the last W_nat native attempts. p0 is therefore **estimated and time-varying**, not known |
| dependence | outcomes within an iteration share the same populations, control fit and structure; receivers can repeat; outcomes are **not independent** |
| evidence accumulation | additive LLR increments per outcome (formula above), accumulated across iterations until a decision |
| decision boundaries | A = log((1−β)/α), B = log(β/(1−α)); these are Wald's formulas, used as **nominal thresholds** only |
| minimum samples | n_min = 20 outcomes before any decision |
| truncation | at n_max = 200 outcomes, decided by the midpoint rule; flagged |
| reset | LLR and n reset to 0 after every decision (`reset_after_decision`) |

### 4.9 Greedy replacement after exchange

- If `f' < f(x_r)`: the receiver is replaced, `(x_r, f_r) ← (x', f')`.
- If the receiver is in M: set `X_M_old[r], fit_old[r] ← x', f'`. This is
  the same memory synchronisation that `fads_greedy` performs.
- Population bests and `Pg` are updated in step 7.

---

## 5. FE accounting

| component | FE per iteration | source of points |
|---|---|---|
| initialisation | 2N (once) | new points |
| HBA offspring | N | new points |
| MPA move evaluation | N | new points |
| FAD candidates | N | new points |
| exchange candidates (ACTIVE) | E_x | new points |
| probe candidates (SUPPRESSED) | r_probe | new points |
| native-control fit, covariance, partition, BIC, PLL, SPRT-style rule, shadow selectors, placebo, diagnostics | **0** | only already evaluated points and their stored fitness |
| structural labels (S1–S4), S6 position-based reference label, S7 offline reference label, JOINT-HALF (§4.4, §12, §14) | **0**, computed offline after the run | stored positions, outcomes and windows only; never visible to the algorithm |

Rules:
1. Every call to the objective goes through the single `CountedObjective`
   of the run. `BudgetExceeded` is a hard error, never caught.
2. There are no hidden evaluations: the objective handle is not passed to
   the evidence, selection or diagnostic modules. A unit test asserts this
   by passing a guard that raises on call.
3. Probes are part of the declared budget MaxFE. Nothing is added beyond
   MaxFE.
4. Stop rule: the run ends when `FE_remaining < 3N + e_t` at step 0.
   Unused FE are logged as `fe_unused`.
   - They are always `< 3N + E_x`: 114 for N = 30 and E_x = 24.
   - For MPHBS (A4), the authors' calibrated T is kept, and totals are as
     in `experiments/baseline_validation_results.md`.
5. Audit: at the end of the run,

   ```
   fe_phase1 + fe_fad + fe_exchange + fe_probe + 2N == obj.fe_used
   ```

   The run is marked invalid otherwise.
6. MaxFE:
   - synthetic: 10000·D;
   - CEC2022: 200000;
   - FIR: 150000.
7. Per-iteration cost:
   - N1 ACTIVE: 3N + E_x;
   - N1 SUPPRESSED: 3N + r_probe;
   - A0: 3N;
   - MPHBS B-C1: 3N + 1 + 24.

   The small difference from MPHBS (N1 has no single Pbest evaluation) is
   declared. It is not compensated.

---

## 6. Hypothesis-selection procedure

The procedure is hierarchical and runs in a fixed order at each iteration.

1. **H0 vs exchange** (§7): the SPRT-style rule state from previous outcomes determines
   e_t.
2. **Structure** (§4.3–4.5): the driving selector sets k* every U
   iterations. Shadow selectors are computed and logged at the same epochs.
3. **Mapping** (§8): uses k* and the driving selector's partition or
   eigenbasis.

### 6.1 Driving selector per arm

- A5: HBA-only.
- A6: MPA-only.
- A7: JOINT.
- A8: JOINT.
- A11: DECOUPLED.
- P: POOLED.
- A10: the success-rate control rule (§11).
- A1, A2 and A3 have fixed k*. They use the JOINT selector's partition and
  eigenbasis when k* = HB or HR.

### 6.2 Passive (shadow) design for E4

In every arm, all five selectors are computed on the same windows. They
consume no FE and no native-stream randomness.

**Primary E4 analysis:** the shadow selections in **A0** runs, which have no
exchange. There, the choice of selector cannot feed back into the
trajectory, so selector comparisons are paired and free of confounding.

Within this analysis, **JOINT vs POOLED is the primary comparison**
(§14.6). The active arms A5, A6, A7 and P are secondary.

### 6.3 Change-point detector

**Not part of N1 and not to be added.** It is not specified, not planned
for this study, and not a pending feature.

A change-point mechanism may be considered in the future only as a
separately justified extension, with a new specification and review. That
can happen only after all of the following have been validated:
- E3;
- E4;
- structural selection;
- BIC behaviour;
- SPRT-style calibration;
- FE accounting;
- the leakage controls.

### 6.4 Warm-up and availability

| condition | available candidates |
|---|---|
| fewer than W samples in `CUR_pop` for a population the selector needs | none; k* stays at HI (the default mapping, analogous to MPHBS dimension-wise exchange) |
| W ≤ samples < 2W | {HI, HR}, scored on the newest W; HB unavailable (no disjoint OLD) |
| ≥ 2W samples | {HI, HB (if non-degenerate), HR} |

Epochs with no evidence are excluded from accuracy metrics. The number of
such epochs is reported.

---

## 7. H0 test

- Rule: the SPRT-style prequential sequential likelihood-ratio decision
  rule of §4.8, with the native-control model of §4.7.
- Data: the outcomes of exchange candidates (ACTIVE) and probe candidates
  (SUPPRESSED). There are no separate test evaluations.

Thresholds:
- α and β are **nominal design parameters** that set the Wald-formula
  boundaries. Defaults are α = β = 0.05, giving A = 2.944 and B = −2.944.
- They are **not** claimed to be the realised error probabilities (§7.1).
- These are conventional values. **They have no domain-specific
  justification.**
- They are configurable, and are fixed before experiments.
- Realised error behaviour is measured by **T-SPRT calibration** (§7.2)
  before any H0/S7 result is interpreted.

Effect margin δ = log 2:
- The alternative is that exchange doubles the odds of success relative to
  native search at the same step length.
- This is a pre-declared, arbitrary but explicit choice.
- A sensitivity analysis over δ ∈ {log 1.5, log 3} is run **only** on the
  synthetic suite. It is reported in full, and is not used to pick a value
  after seeing results.

Other policies:

| policy | setting |
|---|---|
| minimum sample count | n_min = 20 outcomes before any decision |
| maximum | n_max = 200 (truncation, §4.8) |
| reset | after every decision (default); alternatives `none` and `sliding(n_max)` are configurable but not used in v1 |
| initial state | ACTIVE |
| probes when SUPPRESSED | r_probe per iteration, using the current k* mapping and the same receiver/donor rule |
| A8 (no H0) | the SPRT-style rule is computed and logged as a shadow, but state is forced to ACTIVE |

### 7.1 Statistical assumptions (no formal guarantee claimed)

Wald's bounds give approximate type I/II error control (≤ α, ≤ β) for
i.i.d. observations with **known** simple hypotheses, no truncation and no
reset. The N1 rule departs from these assumptions:
1. p0 is **estimated online** and changes over time. The hypotheses are
   therefore random and moving.
2. Observations are **adaptive and dependent**:
   - receivers are chosen from populations shaped by earlier outcomes;
   - outcomes within an iteration share state.
3. There is **truncation** at n_max.
4. There is a **reset** after each decision, followed by repeated testing
   over a run. The number of decisions per run is random, so per-run error
   rates are not α or β.
5. p0 is clipped, and the step-length covariate is clipped.

Consequently:
- **no formal type-I/type-II guarantee is claimed**;
- α and β are nominal;
- the realised behaviour is reported only as measured by T-SPRT calibration
  (§7.2).

Any statement about error rates must cite the calibration, not α and β.

### 7.2 T-SPRT calibration (required before interpreting S7 or any H0 result)

T-SPRT calibration runs in two levels. Both use **only** the pipeline code
of §4.7–4.8, unchanged.

**Level 1: semi-synthetic replay.** This uses zero FE and requires no
optimisation run beyond the A0 smoke runs.
- Take logged native-attempt buffers and exchange step lengths from A0
  runs.
- Generate exchange outcomes as Bernoulli draws with log-odds equal to:
  - the estimated native-control log-odds + 0 (**controlled no-transfer**),
    or
  - the estimated native-control log-odds + δ (**controlled positive
    transfer**).
- Draw outcomes independently, and also with an induced within-iteration
  correlation through a shared per-iteration latent offset
  u_t ~ N(0, σ_u²), σ_u ∈ {0, 0.5}. These values are fixed a priori.
- Run the rule exactly as specified: estimated p0, n_min, n_max,
  truncation and reset.

**Level 2: pipeline harness.** This runs on synthetic functions, inside the
declared calibration budget of the harness runs. The harness is a test
fixture and is **not** an N1 variant.
- **Controlled no-transfer:** each exchange candidate is replaced by a
  native-equivalent candidate. The HBA/MPA operator is applied to the
  receiver, so by construction its success probability is that of a native
  attempt. This is on S1 and S3.
- **Controlled positive transfer:** the donor is replaced by the known
  optimum o, on S1 and S3.

Reported quantities, per condition:
- **False activation rate**: under controlled no-transfer, the fraction of
  decisions that are INFORMATIVE, and the fraction of runs whose state
  ends ACTIVE.
- **False suppression rate**: under controlled positive transfer, the
  fraction of decisions that are UNINFORMATIVE, and the fraction of
  iterations spent SUPPRESSED.
- **Stopping-time distribution**: the number of outcomes per decision
  (median, IQR, 95th percentile) and the truncation fraction.
- Monte Carlo standard errors for all of the above.

Handling of the results:
- Calibration results are reported **as measured**.
- α, β, δ, n_min and n_max are **not** retuned on them.
- If the realised rates are grossly off, the rule is to STOP and report
  (risk R-SPRT-CALIBRATION). "Grossly off" means the false activation or
  false suppression rate exceeds 3× the nominal value in Level 1 with
  σ_u = 0.

The separate unit test **T-SPRT-UNIT** (§13.1) only checks the
implementation under the idealised Wald assumptions.

---

## 8. Exchange mapping

### 8.1 Pairing for exchange e

1. **Receiver population.** pop_r = M if e is odd, H if e is even. The
   alternation restarts every iteration, so ACTIVE iterations are balanced
   (E_x = 24 is even).
2. **Receiver.** A binary tournament in pop_r, using `rng_exchange`: the
   lower f wins, and ties go to the lower index.
3. **Donor.** The member of the other population with the **same fitness
   rank** as the receiver within its own population. Ranks are computed at
   the start of step 6 and **not** updated during the step; ties go to the
   lower index.
4. `d = x_d − x_r` (raw coordinates) and `d̃ = d / s`.

### 8.2 Units and mask

- Units are drawn independently, each with probability p_x.
- If no unit is drawn, one is chosen uniformly.
- Masks use `rng_exchange`.

| k* | units | candidate |
|---|---|---|
| HI | D coordinates | `x' = x_r + m ⊙ d`, m ∈ {0,1}^D |
| HB | blocks of the driving partition P | `x'_j = x_r,j + d_j` for j in selected blocks; others unchanged |
| HR | D eigen-directions of the **receiver population's** Σ̂ (HR model) from the driving selector | `x' = x_r + S U (m ⊙ Uᵀ d̃)`, with S = diag(s) and U the eigenvectors |

All three mappings reduce to `x' = x_d` when every unit is selected, and to
`x_r` when none is. This makes the structures comparable.

After mapping, `x' ← clip(x', lb, ub)`. The same `clip` as the backbone is
used.

### 8.3 H0

When SUPPRESSED, no normal exchange takes place; there are only r_probe
probes. The FE saved go to additional native iterations via the stop rule
(§5.4).

### 8.4 Fixed-structure arms (A1–A3)

- These arms use the same mapping with k* fixed.
- Before the evidence is available (§6.4):
  - A2 falls back to HI until a non-degenerate joint partition exists;
  - A3 falls back to HI until W samples exist.
- Fallback epochs are logged.

---

## 9. Numerical safeguards

| item | safeguard |
|---|---|
| scale s_j | floor `s_floor·(ub_j − lb_j)`, s_floor = 1e-12 |
| covariance | eigenvalue floor ε_Σ = ε_rel·tr(S)/D with ε_rel = 1e-8; if tr(S) = 0 (all displacements identical), the window is flagged `degenerate_window` and the selector keeps its previous k* |
| Cholesky failure after flooring | add 10·ε_Σ·I once; if it fails again, mark the epoch `numerical_failure` and keep k* |
| correlation for partition | columns with zero variance get correlation 0 with all others (isolated singletons) |
| Fisher transform | r clipped to ±0.999999 |
| logistic fit | ridge λ_nat; p clipped to [p_clip, 1 − p_clip] with p_clip = 1e-4; if NA has no successes or no failures, θ is set to the intercept-only MLE of the smoothed proportion (y_sum + 0.5)/(n + 1) and the fit is flagged `separation` |
| step length | ε_l = 1e-12 inside the log |
| non-finite f | treated as `y = 0` (failure) and counted; the candidate is never accepted; counts are reported |
| non-finite displacement | excluded from windows and counted |
| RNG isolation | `rng_init = default_rng(seed)` (initialisation only; identical to MPHBS); `rng_native`, `rng_exchange`, `rng_placebo` spawned from `SeedSequence([seed, 1])` (§15.1); evidence and shadow code consume **no** RNG except `rng_placebo` (DECOUPLED only) |
| BLAS | pinned to one thread (as in `lakaie/__init__.py`) for determinism |

Consequence of RNG isolation: in A0, the trajectories are **bit-identical**
whether or not the shadow selectors run. A regression test (T-SHADOW)
asserts this.

---

## 10. Hyperparameters

All values are fixed **a priori**. None was tuned on any N1 result, because
no N1 result exists. There is no per-benchmark tuning.

| name | value | justification / origin |
|---|---|---|
| N | 30 per population | MPHBS setting |
| E_x | 24 | equals MPHBS B-C1 episodes E = n_sub·N_i = 8·3, giving an equal exchange budget per iteration |
| r_probe | 2 | minimum even number that keeps both receiver directions probed; about 8% of E_x |
| p_x | 0.2 | a priori; about 20% of units per exchange |
| W | max(60, 5D) | n ≥ 3D keeps the full covariance well-posed (n/q is small, see risk R-BIC-POWER) |
| U | 5 iterations | amortises the cost of selection |
| κ | 3 (ΔBIC = 6) | "strong" on the Kass–Raftery scale, used as a conventional anchor; not tuned |
| α_link | 0.01 (Bonferroni over pairs) | conventional |
| α, β | 0.05, 0.05 | conventional (§7) |
| δ | log 2 | §7 |
| n_min, n_max | 20, 200 | §7 |
| W_nat | 10·3N = 900 native attempts | about 10 iterations of native attempts |
| λ_nat | 1e-2 | weak ridge |
| include_fad | True | FAD is an MPA operator |
| ε_rel, s_floor, p_clip, ε_l | 1e-8, 1e-12, 1e-4, 1e-12 | numerical only |
| backbone | HybridState with FAD **before** exchange, all domains | a single fixed backbone; MPHBS's domain-specific placement is **not** copied into N1 (declared asymmetry, risk R-BASE-ASYM) |

Forbidden simultaneous tuning, as instructed:
- exchange probability p_x;
- operator;
- N;
- E_x;
- structure;
- optimizer parameters.

**None of these is tuned.** The only planned sensitivity analysis is δ (§7),
on the synthetic suite, reported in full.

---

## 11. Baseline definitions

| id | definition | purpose |
|---|---|---|
| A0 | no exchange (e_t = 0) + all selectors as passive shadows | **trajectory-neutral selector comparison**; primary data for E4 (JOINT vs POOLED vs HBA-only vs MPA-only) |
| A1 | fixed HI exchange, always active (E_x per iteration) | fixed independent-coordinate representation |
| A2 | fixed HB (joint partition), always active | fixed block representation |
| A3 | fixed HR (joint receiver eigenbasis), always active | fixed rotated representation |
| A4 | **MPHBS**: validated port, unchanged; B-C1 for synthetic and CEC2022, A-C1 for FIR (the authors' domain configurations); results in `results/reproduction/MPHBS/` | published reference method |
| A5 | HBA-only driving selector + H0 | single-optimizer structural evidence (HBA) |
| A6 | MPA-only driving selector + H0 | single-optimizer structural evidence (MPA) |
| A7 | JOINT driving selector + H0 | **full proposed method** |
| A8 | JOINT driving selector, H0 disabled (SPRT-style rule logged only) | isolates the contribution of suppression/activation (E3) |
| A9 | ≡ A7; the same configuration under a second label, not run twice | kept only for table correspondence |
| A10 | **neutral success-rate representation-selection control**: the driving structure is chosen by probability matching over {HI, HB, HR} on exchange success rates (floor 0.1 per structure, learning rate 0.1); always active; same mapping and joint partition/eigenbasis | compares statistical structural evidence against a success-based control. It is conceptually inspired by success-based coordinate-system selection (e.g. ACoS). **It is not ACoS and must not be called an ACoS baseline**: the published ACoS algorithm has not been reproduced from its paper or code |
| A11 | DECOUPLED driving selector + H0 | no shared structural evidence (placebo) |
| **P** | **POOLED driving selector + H0** | **PRIMARY E4 SAMPLE-SIZE CONTROL** (active-regime counterpart of the passive JOINT vs POOLED comparison in A0) |

MPHBS rules:
- MPHBS is **not modified**.
- Before any N1 comparison, the MPHBS reproduction is re-run on the N1
  seeds, and its FE totals are re-verified against the validated values
  (199,930 for CEC B-C1; 149,857 for FIR A-C1).

Standalone HBA and MPA (existing validated ports) are reported as
reference rows only.

---

## 12. Synthetic benchmark definitions

Common settings:
- D ∈ {10, 20}; box [−100, 100]^D.
- Shift o ~ U[−80, 80]^D, from the instance seed.
- N = 30; MaxFE = 10000·D; R = 30 runs.
- f* = 0, and error = f.
- Rotations are Haar-random orthogonal matrices from QR with a sign
  correction, drawn from the instance seed.

Conditioning weights:

```
c_i = 10^{4(i−1)/(D−1)}
```

**Roles of the conditions (rev. 2):**

| group | conditions | use |
|---|---|---|
| **PRIMARY structure-recovery tests** | S1, S2, S3 | deterministic functions with a fixed, known structural label; the **only** conditions in the confirmatory structure tests (C1–C4, E4 Q1–Q5) |
| **SECONDARY / CONTROL conditions** | S4, S5, S6, S7, S8 | reported with pre-declared secondary analyses; never pooled with S1–S3 in confirmatory tests |

| id | definition | label / role |
|---|---|---|
| S1 | separable ellipsoid `f = Σ c_i z_i²`, z = x − o | **primary**; structural label HI |
| S2 | block-rotated ellipsoid: fixed coordinate permutation π, blocks of 5 (D=10: 2 blocks; D=20: 4 blocks), each block with its own rotation Q_b; `f = Σ c_i (Q z_π)_i²` | **primary**; structural label HB, known partition = π-blocks |
| S3 | dense rotated ellipsoid: `f = Σ c_i (Q z)_i²` | **primary**; structural label HR; **hard diagnostic gate for HR detectability (§16, Gate 7)** |
| S4 | mixed: D=20 has 10 separable + two rotated 5-blocks; D=10 has 5 separable + one rotated 5-block (permuted) | secondary; structural label HB, known partition |
| S5 | rotated Rastrigin: `f = Σ (y_i² − 10 cos 2πy_i + 10)`, y = 0.0512·Q z | secondary robustness; weak label HR (multimodal) |
| S6 | switching: `f = (1−w) f_S1(z) + w f_S3(z)`, `w = σ((r0 − ‖z‖)/κ_r)`, r0 = 10, κ_r = 1 | secondary; time-varying position-based reference label (HI far from o, HR near o) |
| S7 | two basins: `f = min(f_S3^{Q_A}(x − o_A), f_S3^{Q_B}(x − o_B) + Δ)`, with ‖o_A − o_B‖ ≥ 100 and Δ = 1. HBA is initialised uniformly in `o_A ± 20` and MPA in `o_B ± 20` (all arms, including A4) | secondary; **controlled null-transfer scenario** (see below) |
| S8 | `f = Σ c_i (Q_e z)_i²`, where Q_e is a fresh rotation for **every evaluation**, drawn from a stream keyed by `(instance_seed, run_seed, FE index)` | secondary; **stochastic placebo / robustness condition with no stable coordinate structure**; **no HI/HB/HR label exists** |

Labels and reference labels:
- **S1–S4:** the structural labels follow from the construction of the
  function. For S2 and S4 the partition is known, and the adjusted Rand
  index against it is reported in addition to label accuracy.
- **S6:** this is a *position-based reference label*, not a structural
  truth about the displacement distribution.
  - The per-epoch label is HR if more than 50% of the CUR samples of the
    relevant populations were recorded at points with w(x_old) ≥ 0.5, and
    HI otherwise.
  - It is computed offline from stored positions.
- **S7 — controlled null-transfer scenario.**
  - S7 is *constructed* so that exchange between the two populations is
    expected to be weak or unhelpful. Each population searches a
    different, differently rotated basin.
  - Its purpose is limited to three things:
    1. controlled evaluation of suppression behaviour;
    2. comparison against an offline reference;
    3. nothing more. In particular, it is **not** proof of a universally
       correct H0 detector.
  - The comparison uses an **offline reference label based on batch
    exchange-vs-native evidence**: the hindsight full-run logistic
    estimate of the log-odds offset of exchange vs the native control in
    **A8** runs, with a 95% CI.
    - CI upper bound < δ → reference "uninformative";
    - CI lower bound > 0 → reference "informative";
    - otherwise "undetermined".
  - This reference label **is NOT an independent ground-truth oracle.**
    It is built from the same kind of evidence (exchange vs native
    success) and the same native-control model as the online rule.
    Agreement between the online rule and the reference therefore measures
    consistency between a sequential and a batch estimate of the same
    quantity, not correctness against an external truth.
  - The same reference label is applied to S1–S4 in A8 runs.
- **S8 — stochastic placebo.**
  - S8 has no stable coordinate structure, so it is **not** reported as
    having a known HI/HB/HR label, and it is not a structure-recovery
    benchmark.
  - It tests whether the structural-selection mechanism **avoids
    confidently exploiting a structure that is not stable across
    evaluations**. Pre-declared measures, in A0 shadows and A7:
    - the fraction of epochs selecting HB or HR;
    - the fraction of epochs in which HB or HR wins with a BIC margin > 2κ
      ("confident selection");
    - the switch frequency;
    - the predictive log-likelihood gap PLL(HR) − PLL(HI). A structure that
      is not stable should not generalise from OLD to CUR.
  - These are compared with S1 and S3 (test C8).

Seeds for instance generation are separate from run seeds (§15).

---

## 13. Ablations

### 13.1 Unit tests (before any experiment)

| test | description |
|---|---|
| T-FE | per-phase FE sum equals counter; the evidence modules receive a raising objective guard |
| T-SHADOW | A0 with and without shadow selectors gives bit-identical trajectories |
| T-PREQ | the HB partition is a function of OLD only: permuting or altering CUR leaves P unchanged; p0 is logged before evaluation (log-order assertion) |
| T-BIC | on Gaussian samples drawn from known diagonal, block and full covariances (n = W, D ∈ {10, 20}), BIC selects the true model at a rate reported (**not** asserted at a fixed value); the implementation is checked against `scipy.stats.multivariate_normal.logpdf` |
| T-SPRT-UNIT | implementation check only, under idealised Wald assumptions (i.i.d. Bernoulli, **known** p0, no truncation or reset): empirical type I/II error ≤ nominal + 3 Monte Carlo SE. It says nothing about the realised N1 behaviour, which is covered by T-SPRT calibration (§7.2) |
| T-E4-OBS | JOINT and POOLED receive the identical multiset of observations at every epoch (row-count and checksum equality); JOINT-HALF uses exactly W total rows |
| T-MAP | every mapping with the full mask gives x_d, and with an empty mask gives x_r; the HR mapping is exact at identity U |
| T-DEGEN | singleton/single-block partitions are removed from candidates; zero-variance windows are handled |
| T-MPHBS | the existing MPHBS and backbone tests still pass unchanged |
| T-INIT | the initial populations of every N1 arm and of A4 are identical for the same seed |

### 13.2 Ablation matrix

A0–A11 and P (§11). They answer:

| question | comparison |
|---|---|
| **E4 primary control (passive)** | **JOINT vs POOLED** shadow accuracy in A0, on S1–S3 |
| E4 single-optimizer comparisons (passive) | JOINT vs HBA-only, JOINT vs MPA-only in A0, on S1–S3 (necessary, **not sufficient**) |
| E4 sample-matched check | JOINT-HALF vs HBA-only / MPA-only (offline, A0) |
| E4 placebo | JOINT vs DECOUPLED |
| E4 (active, secondary) | A7 vs P; A7 vs A5, A6 |
| E3 | A7 vs A8 (H0 on/off); H0 decisions vs the offline reference label; T-SPRT calibration |
| value of structure | A1/A2/A3 vs A7; A10 (success-rate control) vs A7 |
| value of exchange | A0 vs A7/A8 |
| reference | A4 (MPHBS) vs A7 |

### 13.3 No additional mechanisms

- No mechanism beyond those in §1–§8 is added. This includes the
  change-point detector (§6.3).
- The gates in §16 govern progress. They do not unlock new features.

---

## 14. Statistical protocol

### 14.1 Units

- The unit of analysis is the run: 30 per (function, D, arm).
- Seeds are common across arms, so tests are paired by seed.

### 14.2 Mechanism metrics

Per run:
- selection accuracy (after warm-up epochs);
- confusion matrix;
- latency to the first correct selection;
- false-selection rate;
- BIC margins;
- adjusted Rand index (S2 and S4);
- H0 decisions: state trace, number of decisions, truncation fraction;
- precision and recall of UNINFORMATIVE vs the offline reference label
  (§12). This is **agreement with a batch estimate, not accuracy against a
  ground truth**;
- BIC-power diagnostics per epoch:
  - samples used;
  - BIC of HI, HB and HR;
  - ΔBIC;
  - selected-model frequencies;
  - whether HR is ever selected on S3;
- switch latency and number of switches (chattering) on S6;
- FE share spent on exchange and on probes.

### 14.3 Pre-registered confirmatory tests

Significance level 0.05 throughout, with Holm correction within each
family.

Confirmatory structure tests use **only S1, S2 and S3** (§12).

| id | hypothesis | data | test | family |
|---|---|---|---|---|
| C1 | JOINT selector accuracy > 1/3 (chance over 3 labels) | A0 shadows, S1–S3 × D | one-sided Wilcoxon signed-rank on (acc − 1/3) | 6 tests |
| **C2 (E4 primary)** | **JOINT accuracy > POOLED accuracy** | A0 shadows, S1–S3 × D | one-sided paired Wilcoxon | 6 tests |
| C3 (E4) | JOINT accuracy > HBA-only; JOINT accuracy > MPA-only | A0 shadows, S1–S3 × D | one-sided paired Wilcoxon | 12 tests |
| C4 (E4 placebo) | JOINT accuracy > DECOUPLED accuracy on S2, S3 | A0 shadows | one-sided paired Wilcoxon | 4 tests |
| C9 (E4 sample-matched) | JOINT-HALF accuracy > HBA-only; JOINT-HALF > MPA-only | A0 shadows (offline), S1–S3 × D | one-sided paired Wilcoxon | 12 tests |
| C5 (E3) | UNINFORMATIVE decisions agree with the offline reference label | A7 vs reference from A8, S1–S3 and S7 | precision/recall with percentile bootstrap 95% CI (10,000 resamples over runs); **interpreted only after T-SPRT calibration (§7.2) and only as agreement with a batch estimate** | descriptive + CI |
| C6 (E3) | A7 final error ≠ A8 final error | S1–S4, S7, S8 × D | two-sided paired Wilcoxon | 12 tests |
| C7 | S6: switch HI→HR in ≥ 80% of runs, with latency ≤ 20 selection epochs after the position-based reference transition | A7 | one-sided exact binomial (H0: p ≤ 0.8) | 2 tests |
| C8 (S8 placebo) | the fraction of *confident* HB/HR selections (margin > 2κ) on S8 < on S3 | JOINT shadows in A0 | one-sided Mann–Whitney over runs | 2 tests |

Secondary analyses (not confirmatory):
- S4: the C1–C4 analogues, as a separate family.
- S5: descriptive accuracy against its weak label.
- S8:
  - Jensen–Shannon divergence of selection distributions vs S1, with a
    permutation test (10,000 permutations);
  - the PLL(HR) − PLL(HI) gap;
  - switch frequency.
- Active-regime E4: A7 vs P, A5 and A6, on selection accuracy and final
  error (§14.6 Q3).

### 14.4 Performance comparisons (secondary)

Final error on the synthetic suite, CEC2022 (12 functions, D = 20) and FIR
(8 cases):
- Friedman test on per-instance means, with Kendall's W.
- Paired Wilcoxon of A7 vs each other arm per instance, Holm-corrected
  within the benchmark.
- W/T/L counts.
- Convergence curves.

The code reuses `lakaie/analysis/stats.py` (holm, wilcoxon_p, friedman_block_analysis, paired_instance_tests).

The existing Wilcoxon helper uses the normal approximation; exact p-values
are used for n = 30 when available in SciPy. This is to be confirmed at
implementation, and the choice logged.

### 14.5 Reporting rules

- All runs, all functions and all arms are reported.
- No run is removed. Failed or invalid runs (T-FE audit failure,
  exceptions) are reported with their cause and count.
- No superiority wording before §14.3–14.4 are complete.
- Effect sizes are reported alongside p-values: the median paired
  difference and the matched-pairs rank-biserial correlation.
- S7 results are described only as behaviour in a *controlled null-transfer
  scenario*. S8 results are described only as *robustness under a
  stochastic placebo*.
- Neither is described as ground-truth H0 detection or structure recovery.

### 14.6 E4 analysis (defined before any experiment)

All primary E4 analyses use A0 shadow selectors on S1–S3, D ∈ {10, 20}, 30
paired runs. Accuracy is computed per run over evidence-bearing epochs
(§6.4).

| question | analysis | status |
|---|---|---|
| **Q1** Does JOINT outperform HBA-only? | C3 (accuracy, passive); secondary: A7 vs A5 (active accuracy and final error) | necessary, not sufficient |
| **Q2** Does JOINT outperform MPA-only? | C3 (accuracy, passive); secondary: A7 vs A6 | necessary, not sufficient |
| **Q3** Does JOINT outperform POOLED? | active regime: A7 vs P (selection accuracy against S1–S3 labels; final error, paired Wilcoxon, two-sided, Holm over S1–S3 × D) | secondary (feedback-confounded) |
| **Q4** Does JOINT provide more accurate structural selection than POOLED on S1–S3? | **C2** (passive, trajectory-neutral) | **PRIMARY E4 control** |
| **Q5** Does the evidence persist after controlling for the number of observations? | (i) C2: JOINT vs POOLED have identical observations by construction (checked by T-E4-OBS); (ii) C9: JOINT-HALF (n = W) vs HBA-only / MPA-only (n = W) | required for the E4 pattern |

**E4 decision rule** (intersection-union, fixed in advance). For a cell
(function ∈ {S1, S2, S3}, D), the E4 evidence pattern holds only if **all**
of the following are significant after Holm within their families:
- C2: JOINT > POOLED;
- C3: JOINT > HBA-only;
- C3: JOINT > MPA-only;
- C9: JOINT-HALF > HBA-only and JOINT-HALF > MPA-only.

Reporting of the rule:
- Results are reported per cell as the number of cells, out of 6, where
  the pattern holds, and for which functions.
- Cells in which all selectors are at ceiling (accuracy ≥ 0.95 for all)
  are reported as "ceiling: pattern not assessable". They are **not**
  counted as support.

Interpretation:
- JOINT > HBA-only / MPA-only **alone does not demonstrate E4**. It is
  consistent with a pure sample-size effect.
- **JOINT > POOLED is the critical additional evidence** that
  optimizer-specific (heterogeneous) evidence contributes beyond sample
  size.
- JOINT ≤ POOLED on all cells means E4 is not supported, and it is reported
  as such.
- None of these outcomes is converted into a novelty claim.

---

## 15. Reproducibility protocol

### 15.1 Seeds (`configs/seeds.json`, created at implementation Stage 1)

- Run seed:
  `seed = master[campaign] + benchmark_offset + 1000·instance + run`, with
  run = 1..R. This is the same formula as `configs/reproducibility.yaml`.
- New masters:

  | campaign | master |
  |---|---|
  | `n1_unit` | 70000000 |
  | `n1_smoke` | 81000000 |
  | `n1_synthetic` | 71000000 |
  | `n1_pilot` (CEC and FIR pilots; separated by benchmark offset) | 72000000 |
  | `n1_main` (CEC and FIR full) | 20260405 (the `main` master; never used before, because the LA-KAIE full experiment was not run) |

- Benchmark offsets:

  | benchmark | offset |
  |---|---|
  | CEC2022 | 0 |
  | FIR | 100000 |
  | SYN | 200000 |

- Instance seeds for the synthetic suite: `91000000 + 100·fid + D`.
- Every realised seed (run seed, spawned stream keys, instance seed) is
  written into `configs/seeds.json` and into each raw-result record.
- All arms use the same run seed per (instance, run).
- **Initial populations are identical across all arms, including A4
  (MPHBS).**
  - `rng_init = numpy.random.default_rng(seed)` is the same generator and
    seed that the validated MPHBS port uses.
  - N1 uses `rng_init` only for `HybridState` initialisation, which draws
    X_M and then X_H, exactly as in MPHBS.
  - `rng_native`, `rng_exchange` and `rng_placebo` are spawned from
    `SeedSequence([seed, 1])`. This is a separate entropy source, so they
    never collide with `rng_init`.
  - After initialisation, A4 continues on its single generator, as in the
    unmodified port. **MPHBS is not modified.**
  - A unit test asserts that the initial populations are identical.

### 15.2 Environment

- Pinned `requirements.txt` and `environment.yml` (existing).
- `scripts/capture_environment.py` output is stored per campaign.
- BLAS uses one thread.

### 15.3 Layout

| path | contents |
|---|---|
| `src/n1/` | implementation, importing the validated `lakaie` backbone and benchmarks |
| `configs/n1_*.yaml` | configurations |
| `tests/n1/` | unit tests |
| `experiments/` | run specifications |
| `results/raw/n1_*`, `results/reproduction/MPHBS/` | results |
| `logs/` | logs |
| `figures/` | figures |
| `statistics/` | statistics |
| `docs/` | documentation |

### 15.4 One command per stage (planned names)

| stage | command |
|---|---|
| — | **blocked until Gate 0 (§16) is resolved and this spec passes independent review** |
| 0 | `python scripts/n1_reproduce_mphbs.py` |
| 1 | `pytest tests/n1` |
| 3a | `python scripts/n1_run.py --campaign n1_smoke` (5 runs, synthetic S1–S3 plus control conditions) |
| 3b | `python scripts/n1_calibrate_sprt.py` (T-SPRT calibration, §7.2; Gate 8) → **STOP**, `MECHANISM_SMOKE_REPORT.md` (Gates 1–9) |
| 4 | `python scripts/n1_run.py --campaign n1_synthetic` (requires Gates 1–9 and user approval) |
| 5 | `python scripts/n1_run.py --campaign n1_pilot` |
| 6 | `python scripts/n1_run.py --campaign n1_main` |
| 7 | `python scripts/n1_stats.py` |

### 15.5 Logs

- Per run, the following are saved to disk:
  - the exchange log;
  - the selection log;
  - the NA summary;
  - the Recorder arrays;
  - per-phase FE.
- Run-level JSON records the git commit hash and the config hash.

### 15.6 Freeze

- Before Stage 4, the config files and this spec are tagged in git
  (`n1-freeze-synthetic`).
- Before Stage 6, the tag is `n1-freeze-main`.
- After a freeze, changes are allowed only for bug fixes. Each fix must be:
  - logged;
  - justified by a failing unit test, not by performance;
  - followed by a complete re-run of the affected campaign.

---

## 16. Experimental gates

Gates are evaluated in order.
- A failed gate means **STOP and report**, with no silent patch.
- Gate thresholds are fixed now, before any result. They are diagnostic
  thresholds for validity, not performance targets.
- Gates 1–9 must all pass before **any** full CEC2022, FIR or 30-run
  synthetic experiment.

| gate | requirement | operational criterion (fixed a priori) | evidence source |
|---|---|---|---|
| **Gate 0** (precondition to any implementation) | the literature-verification gate is resolved separately | every item in the outstanding list below is either verified from full text or code, or recorded as an unresolved evidence gap with explicit user acceptance to proceed despite it | literature report; user decision |
| Gate 1 | all unit tests pass | every T-* test in §13.1 passes, including T-SPRT-UNIT, T-E4-OBS and T-INIT | `pytest tests/n1` |
| Gate 2 | FE accounting is exact | FE audit (§5.5) holds in 100% of smoke runs; `fe_unused < 3N + E_x` | smoke logs |
| Gate 3 | no hidden objective evaluations during shadow/model-selection operations | T-FE raising guard passes; the FE count of an A0 run is identical with shadow selectors on and off (T-SHADOW) | unit tests; smoke |
| Gate 4 | prequential partitioning passes leakage tests | T-PREQ passes; the log-order assertion (p0 before f) holds for 100% of exchange records; no `op=EXCH` rows in structural windows | unit tests; smoke logs |
| Gate 5 | S1 structural selection behaves consistently with HI | in A0 smoke runs (5 runs × D ∈ {10, 20}), HI is the modal JOINT selection over evidence-bearing epochs in ≥ 4 of 5 runs for each D | smoke |
| Gate 6 | S2 structural selection behaves consistently with HB | as Gate 5, with HB modal; adjusted Rand index of the joint partition vs truth reported | smoke |
| **Gate 7** | S3 provides meaningful opportunity for HR selection | (a) opportunity: ≥ W successful native displacements per population available in ≥ 50% of selection epochs; (b) detectability: JOINT selects HR in ≥ 5% of evidence-bearing epochs pooled over smoke runs, for each D | smoke; BIC-power diagnostics (§3.5) |
| Gate 8 | T-SPRT calibration completed | Level 1 and Level 2 of §7.2 reported; no "grossly off" condition (§7.2) | calibration report |
| Gate 9 | the JOINT vs POOLED E4 analysis is operationally valid | T-E4-OBS passes on smoke logs; JOINT and POOLED each produce a valid selection in ≥ 90% of evidence-bearing epochs; logged windows suffice to compute JOINT-HALF | smoke logs |
| Gate 10 | no implementation changes after full benchmark results begin | git tags `n1-freeze-synthetic` and `n1-freeze-main` (§15.6); only failing-test bug fixes, logged, with a full re-run | git history; `logs/DEVELOPMENT_CHANGES.md` |

**Gate 7 failure (hard diagnostic gate).**
- Condition: HR selection on S3 is approximately absent (criterion (b)
  fails) **despite** sufficient successful native-displacement samples
  (criterion (a) holds).
- Meaning: this is evidence that the BIC/model-complexity formulation lacks
  adequate power for the intended HR detection task. The issue **must be
  diagnosed before proceeding** to full experiments.
- Diagnostics to report:
  - the samples used for structural fitting;
  - BIC of HI, HB and HR;
  - ΔBIC;
  - the selected-model frequencies;
  - sample counts;
  - whether HR was ever selected on S3.
- Handling: any remedy requires a documented proposal and user approval.
  BIC is never tuned after full experimental results. If criterion (a)
  fails, report under-sampling as a separate finding.

**Stage-3 stop.**
- The smoke run ends with `MECHANISM_SMOKE_REPORT.md`, which reports
  Gates 1–9.
- Stage 4 requires explicit user approval, even when all gates pass.

**Outstanding literature verification (Gate 0).** Full text or code is
still needed for:
- the ensemble knowledge-transfer framework with AIE + MAS
  (S2210650223001670) and related multitasking transfer-selection methods;
- LCC (arXiv 2504.17578) and LH-CC (doi 10.1145/3795095.3805054);
- ACoS (doi 10.1109/tcyb.2018.2802912) and its ASOC follow-up;
- GOMEA / FOS-related structural selection. The library code has been
  checked; the papers, including *Predetermined versus learned linkage
  models* (GECCO 2012), have not;
- dd-CMA-ES. The code has been checked; the full paper has not;
- statistical transfer-suppression methods: OKTPO-MFEA, MGAD, and Bayesian
  competitive knowledge transfer;
- island / multi-population EDA model-migration methods;
- relevant KBS 2023–2026 literature (`N1_NOVELTY_EVIDENCE_LOG.md` S14).

Anything that remains inaccessible is recorded as an **unresolved evidence
gap**, not as absence of overlap.
