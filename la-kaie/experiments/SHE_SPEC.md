# SHE specification — evidence-based structural-hypothesis selection for cross-population exchange

**Status.** Stage 0 specification, pre-registered on 2026-10-05, before any SHE code. Nothing in this
document was derived from or adjusted to any SHE result (none exist).

**Sources.** The design follows `NEXT_METHOD_DECISION.md` §8–§12 (formulation, architecture,
ablations, cost, risks). It reuses the E3 idea of `N1_CONCEPT.md` §3.3/§4.2 and the S1–S3 diagnostic
suite of `N1_CONCEPT.md` §12 (`src/n1/synthetic.py`). Every departure from §8 is listed in §22 with
its reason.

---

## 0. Pre-registration context

Read this before the rest.

### 0.1 Prior-art status

All prior-art statements in this repository are **UNVERIFIED** at full-text level:
- `N1_FINAL_NOVELTY_DECISION.md`: classification B, implementation not recommended until ACoS,
  AIE+MAS, LCC and MFEA-II are read in full.
- Later gates (`LAKAIE_NEXT_PAPER_RESEARCH_GATE.md`, `CAUSAL_EXCHANGE_UTILITY_NOVELTY_GATE.md`)
  found learned transfer gating established in evolutionary multitasking. **This bears on E3.**

The user directed implementation on 2026-10-05. **No novelty claim is made by this spec.** Any
claim in the final report will be labelled UNVERIFIED unless a full text has been read.

### 0.2 Prior empirical evidence bearing on Stage 1 (stated so the gate is not post hoc)

- **The N1 mechanism smoke** (`MECHANISM_SMOKE_REPORT.md`) failed Gates 5 and 6.
- **The D3 offline analysis** (`N1_D3_OFFLINE_ANALYSIS_REPORT.md`, classification B) found:
  - successful native HBA/MPA displacements are dominated by population and operator geometry;
  - neither whitening nor residualisation recovered an independent landscape component.
- **This was geometry evidence, not the outcome-prediction evidence SHE uses**, so it does not
  decide Stage 1. It makes a Stage 1 gate failure plausible. The prior expectation, recorded
  honestly here, is that **H-E4 (below) may fail**.

### 0.3 Pivot target named by the Stage 1 gate

The gate's stated pivot is R2: heterogeneous processes as FE-free structure estimators
(`N1_CONCEPT.md` §13).

Its closest formulation, "successful-step statistics as landscape evidence", was rated
**C — abandon** in `FINAL_LITERATURE_OVERLAP_GATE.md`. If the gate fails, the diagnostic report
will recommend R2 as instructed **and** restate this conflict.

---

## 1. Hard constraints and how they are enforced

| constraint | enforcement |
|---|---|
| 0 extra FEs vs MPHBS | Same total budget MaxFE as MPHBS (CEC 200,000; FIR 150,000; tuning 100,000). Every evaluation goes through the FE-counting objective; the run stops exactly at MaxFE. **FEs are only spent on:** backbone offspring, FAD candidates, and exchange candidates (1 FE each). Evidence, models, E3, eigenbasis and linkage use **no** FEs. Unit test T-FE (§19) |
| Fixed exchange budget E per iteration equal to MPHBS's E | E = 24 on CEC2022 (MPHBS B-C1: N_sub·N_i = 8·3); E = 72 on FIR (A-C1: 8·9) (`reference/MPHBS_NOTES.md`); E = 24 on the synthetic suite and the CEC2022 D=10 tuning set (the B-C1 setting). **E is the per-iteration cap on exchange slots, identical for all SHE variants.** How suppressed (E3) slots are used: decision **D-2** (§21) |
| Identical donor/receiver rule across all variants | §3.2; one function, no variant-specific branch |
| FAD placement identical across compared variants | Decision **D-1** (§21). Default: per-domain MPHBS placement (CEC and synthetic: *before* the exchange phase; FIR: *after*), applied identically to **every** SHE variant |
| Hyperparameters only from CEC2022 D=10 | §14; protocol and selection rule identical to `configs/sensitivity.yaml`. No value is changed after any D=20, FIR or synthetic result is seen |
| Existing code paths unmodified | New package `lakaie/she/`; new `configs/she.yaml`; SHE variants under a new top-level key `she_variants` in `configs/ablation.yaml`. That file's existing `variants` key is untouched |
| | **Two additive changes are needed** and are logged in `logs/DEVELOPMENT_CHANGES.md` when made: (i) a `SHE-*` branch in `lakaie/registry.py:method_family/run_method`, with existing branches unchanged; (ii) new campaign masters (§15) |
| | `hybrid_common.py`, `hba.py`, `mpa.py`, `mphb.py`, `mphbs.py`, `lakaie.py`, `components.py` and `src/n1/*` are imported read-only, never edited. Their sha256 freeze test (`tests/n1/test_n1_backbone_frozen.py`) must keep passing |

---

## 2. Backbone and FE accounting

Two populations P^H (HBA) and P^M (MPA), N = 30 each. The backbone is `lakaie.algorithms.hybrid_common.HybridState`, unchanged: `phase1` (HBA move plus greedy, N FEs; MPA move, evaluation and marine memory, N FEs), then `fads_greedy` (N FEs). Progress is FE-based, as in LA-KAIE (t/T := FE_used/MaxFE).

One iteration:

```
phase1                                   (2N FEs)
FADs stage   if placement == before      (N FEs)
every U iterations: refresh B_t, μ_t (§4.4); refit linkage E_t (§11)
exchange phase: up to E slots            (≤ E FEs; §3, §12)
FADs stage   if placement == after       (N FEs)
end-of-iteration: refit outcome models (§6), native control (§12.2); log
```

- **Per-iteration FE use:** 3N + (number of evaluated exchange slots).
- **MPHBS** additionally spends 1 FE per iteration on f(P_best) (MPHBS_NOTES §FE). SHE does not
  need that evaluation. The total budget is identical, so SHE simply fits more iterations into the
  same budget. This is reported, not hidden.
- **Final partial iteration:** when fewer than 3N FEs remain, the remainder is spent on exchange
  slots (E3 state ignored), exactly as LA-KAIE does. The FE counter is checked before every
  evaluation.

---

## 3. Exchange event

### 3.1 Notation

- Population a ∈ {H, M}; the other population is ā.
- x_r ∈ P^a is the receiver, x_d ∈ P^ā the donor; f_r, f_d are their stored fitness values.
- s_t ∈ ℝ^D is the per-coordinate scale: the standard deviation of the pooled current populations
  P^H ∪ P^M (60 points), floored at 10⁻¹²·(ub − lb).
- S_t = diag(s_t).

### 3.2 Donor/receiver rule (fixed; identical for all variants)

- **Direction:** alternates every slot, H→M then M→H (slot 1 has receiver population a = M when
  the iteration index is odd, H when even).
- **Receiver:** binary tournament in P^a (`src/n1/mapping.pick_receiver`, imported read-only).
- **Donor:** an independent binary tournament in P^ā.
- **RNG:** both draws come from the dedicated stream `rng_exchange` (§15.3).

### 3.3 Candidate, outcome and acceptance

For the hypothesis h chosen for this slot (§9), with transform T_h and mask m ∈ 𝓜_h (§4):

    x' = clip( T_h⁻¹[ (1 − m) ⊙ T_h(x_r) + m ⊙ T_h(x_d) ] ),        δ = x' − x_r  (after clipping)

- One FE: f' = f(x').
- **Outcome:** y = 1[f' < f_r] (strict).
- **Acceptance:** greedy. If y = 1, x' replaces x_r in P^a. If a = M, the MPA marine memory is
  synchronised as in LA-KAIE's `replace()`. The population best and the global best are refreshed.
- **Record:** e = (t, a, h_gen = h, π_gen = π̃_t(h), δ, y, f_r, f', idx_r, idx_d, E3 state, probe flag).

**h_0 exception** (decision **D-3**, §21). Under h_0 (m = 1), x' = x_d, so y = 1[f_d < f_r] is
known from stored values.
- **Default:** the h_0 slot spends **0 FEs**. The outcome is computed from stored fitness, the
  record is written with fe = 0, and acceptance is applied.
- The slot still counts as one of the E slots. The FE not spent is returned to the backbone (the
  run continues to MaxFE).

---

## 4. Hypotheses, transforms and masks

H = {h_0, h_1, h_2, h_3}. The transforms are affine and invertible.

| h | claim | T_h(x) | T_h⁻¹(z) | unit set | mask family 𝓜_h |
|---|---|---|---|---|---|
| h_0 | whole-vector migration (no structural decomposition) | x | z | — | {**1**} |
| h_1 | coordinate-separable transfer (MPHBS's implicit assumption) | x | z | coordinates 1..D | per-coordinate Bernoulli(p_x) |
| h_2 | sparse linkage: groups G_t = components of E_t (§11) | x | z | groups g ∈ G_t | per-group Bernoulli(p_x) on all coordinates of the group |
| h_3 | rotated: separable in the pooled-elite eigenbasis | B_tᵀ(x − μ_t) | B_t z + μ_t | eigen-components 1..D | per-component Bernoulli(p_x) |

**Masks.** Each unit is drawn independently with probability p_x. If no unit is drawn, one unit is
chosen uniformly (`src/n1/mapping.draw_mask` semantics). If E_t = ∅, then G_t = singletons and h_2
generates exactly like h_1 (h_2's *model* still differs only by E_t terms; §6).

**Invertibility.** B_t comes from `numpy.linalg.eigh` of a symmetric matrix and is orthogonal, so
T_h⁻¹(T_h(x)) = x up to floating point (test T-INV, §19).

### 4.4 Pooled-elite eigenbasis (h_3)

Every U iterations:
- Elite sets 𝓔^p are the best ⌈q_e·N⌉ members of each population p (q_e = 0.5).
- Each elite set is centred by its own mean (within-population centring, so the inter-population
  offset does not dominate). The centred sets are pooled (n_e = 2⌈q_e N⌉ points) and standardised
  by s_t.
- The covariance uses Ledoit–Wolf-type shrinkage towards (tr C/D)·I with a fixed intensity
  λ_LW = 0.2, a pre-registered constant (n_e ≈ D, so the sample covariance is near singular).
- B_t holds the eigenvectors of S_t·C_shr·S_t, in descending eigenvalue order.
- μ_t is the mean of the pooled elite (uncentred).
- **Guard:** if the population spread has collapsed (max_j s_t,j < 10⁻¹⁰·(ub − lb)_j), keep B_{t−1}.

---

## 5. Features

For every recorded exchange e, every hypothesis g computes its own features from the shared
displacement δ_e:

    ζ^g = S_g⁻¹ T̃_g δ_e

where:
- T̃_g is the linear part of T_g (I for h_0–h_2; B_tᵀ for h_3);
- S_g is diagonal: s_t for h_0–h_2; for h_3, the standard deviation of the pooled current
  populations projected onto B_t (floor as in §3.1).

| model | feature vector φ_g(δ) |
|---|---|
| h_0 | [1, log(1 + ‖ζ⁰‖)] (structure-free baseline) |
| h_1 | [1, ζ¹_k, (ζ¹_k)² for k = 1..D] |
| h_2 | the h_1 vector plus [ζ¹_k ζ¹_l for (k,l) ∈ E_t] |
| h_3 | [1, ζ³_k, (ζ³_k)² for k = 1..D] |

**Features are fixed at record time.** Features use B_t, s_t, E_t *as they were when e was
recorded*, and are stored, so later refreshes do not rewrite history. The h_2 feature set changes
when E_t is refitted. From then on its model is refitted on the stored raw δ with the new E_t, and
earlier prequential losses are kept.

**Fitness-derived features are excluded from all models.** There is no rank gap and no f values.
The outcome is a function of f, so fitness-based features would make the loss reflect donor
quality, not transfer structure (see D-3 and §22).

---

## 6. Outcome models (online L2-regularised logistic)

For g ∈ H:

    p_g(y = 1 | δ) = σ( θ_gᵀ φ_g(δ) )

θ_g is the MAP estimate of the discounted, importance-weighted, L2-penalised log-likelihood over the
exchange window 𝒲_t (the last W_x = 1000 records):

    J_g(θ) = Σ_{e∈𝒲_t} λ_m^{t−t_e} · w_e · [ y_e log p_g(δ_e) + (1 − y_e) log(1 − p_g(δ_e)) ] − (κ₂/2) ‖θ_{−0}‖²

- λ_m = 0.995 per exchange record; t − t_e is counted in records.
- κ₂ = 1.0; the intercept is not penalised.
- **Update:** one warm-started IRLS (Newton) step on J_g at the **end of each iteration**.
  Predictions for exchanges inside iteration t use θ_g from the end of iteration t − 1, so the
  prequential loss is strictly predictive.
- **Initialisation:** θ_g = (logit ȳ₀, 0, …) with ȳ₀ = 0.1. **This is a prior constant, not
  tuned.**
- **Clipping:** p_g ∈ [ε_p, 1 − ε_p], ε_p = 10⁻⁴.

## 7. Importance weights

    w_e = 1 / π̃_{t_e}(h_gen(e))

- π̃ is the floored selection probability actually used when e was generated (§9).
- Bounded by w_e ≤ 1/π_min.
- **Target distribution:** uniform over H (each generator contributes equally in expectation).
  This corrects the covariate shift caused by π concentrating on one generator.

## 8. Prequential loss with forgetting ρ

    ℓ_{g,e} = − log p_g^{(t_e − 1)}( y_e | δ_e )      (parameters fit before e; §6)
    L_g(n) = ρ · L_g(n − 1) + w_n · ℓ_{g,n}           (recursion over exchange records n; L_g(0) = 0)

- ρ = 0.995 per record.
- **Probes and h_0 records** enter like any other record (h_0 records carry fe = 0 but a valid
  (δ, y)).

## 9. Posterior weights, floor and selection

    π_n(g) ∝ π_0(g) · exp( −η · L_g(n) ),   π_0 uniform
    π̃_n(g) = π_min + (1 − |H| π_min) · π_n(g)

Defaults: η = 1.0 (nats; η = 1 with ρ = 1 and w ≡ 1 would be the exact Bayesian posterior of the
predictive models); π_min = 0.05.

**Selection (Thompson-style probability matching).** For each slot, draw h ~ π̃_n with
`rng_exchange`.

---

## 10. Eigenbasis refresh interval U

U = 20 iterations (the same value as LA-KAIE's interaction refit; pre-registered default). B_t, μ_t
(§4.4) and E_t (§11) are refreshed at iterations t ≡ 0 (mod U), and at t = 1 using the initial
populations.

## 11. L1 linkage estimation for h_2 (no extra FEs; own solver, no scikit-learn)

Every U iterations, on the last W_L = 1000 exchange records (identity frame, ζ¹ features fixed at
record time):

    min_{β, c}  − Σ_e w_e [ y_e log σ(u_e) + (1 − y_e) log(1 − σ(u_e)) ] + λ_L Σ_{k<l} |c_kl| + (κ₂/2)‖β_{−0}‖²
    u_e = β_0 + Σ_k (β_k ζ_k + β_{k+D} ζ_k²) + Σ_{k<l} c_kl ζ_k ζ_l

- **Solver:** FISTA (accelerated proximal gradient) with soft-thresholding on c only, step 1/L_lip
  (L_lip = ¼·‖Φ‖₂²·max w), 300 iterations or relative change < 10⁻⁶. Pair features are standardised
  before fitting.
- **λ_L:** chosen by BIC over a geometric path of 10 values from λ_max (the smallest λ with c = 0)
  down to 0.05·λ_max. BIC = −2·loglik + (#nonzero c + 2D + 1)·log(n_eff), with n_eff = Σw/max w.
- **Edges:** E_t = {(k,l) : ĉ_kl ≠ 0} at the BIC-selected λ.
- **Groups G_t:** edges are taken in descending |ĉ_kl| and merged union-find style, provided the
  merged component size stays ≤ g_max = 5 (the LA-KAIE max_group_size default). Remaining
  coordinates become singletons.
- **Guard:** fewer than 2D + 50 records → E_t = ∅.

---

## 12. E3 — native-controlled sequential test (0 extra FEs)

### 12.1 Why an e-process rather than Wald's SPRT

The N1 SPRT-style rule (`src/n1/sprt.py`) was calibrated in `results/n1_calibration/T_SPRT_CALIBRATION_REPORT.md`:
- false activation was 0.066 (σ_u = 0) and 0.147 (σ_u = 0.5) against a nominal α = 0.05;
- harness false suppression was 0.247.

The cause is that the Wald boundaries assume known p0/p1, while p0 comes from a fitted native model.

A test supermartingale ("e-process") keeps Ville's bound P(sup_n K_n ≥ 1/α) ≤ α for **any
predictable** choice of betting fraction and any predictable reference p0_n. The reference is
therefore allowed to be a model fitted on past data. This removes one of the two calibration
failures: plug-in boundaries.

**It does not remove misspecification of the native model:** validity is relative to the reference
p0 (§12.4). This standard property of betting e-processes is stated here as known methodology
**[K]**; no specific paper is cited.

### 12.2 Native control (outcomes the backbone already evaluates)

**Native records** (per evaluated offspring, at no FE cost beyond what the backbone already
spends):
- HBA: the offspring vs its parent in `phase1` (u = x_new − x_old, s = 1[f_new < f_old]).
- MPA: the moved agent vs its memory after `memory_saving` (u = x_t − x_{t−1}, s = 1[moved agent
  retained]).
- FADs: the candidate vs its parent in `fads_greedy`.

These are captured by an FE-transparent objective proxy plus before/after state snapshots in
`lakaie/she/` (the pattern of `src/n1/records.RecordingObjective`; the backbone is not edited).
Snapshots are memory copies, not evaluations.

**Native model:**

    p_nat(s = 1 | u) = σ( c_0 + c_1 ℓ(u) + c_2 ℓ(u)² ),   ℓ(u) = log ‖S_t⁻¹ u‖

- Fitted at the end of each iteration by L2 logistic regression (κ₂ = 1) on the last W_n = 3000
  native records.
- It is used for the exchanges of the next iteration, so it is predictable.

**Matched step length:** the exchange reference is p0_n = p_nat(δ_n) evaluated at the exchange's
own standardised length.

### 12.3 Two one-sided e-processes and the state machine

The state is ∈ {SUPPRESSED, ACTIVE}; the initial state is SUPPRESSED (decision **D-4**, §21).

**Activation (tests H0: "exchange is no better than native"),** used while SUPPRESSED:

    H0:  P(y_n = 1 | 𝓕_{n−1}) ≤ p0_n  for all n in the segment
    K_n = Π_{i in segment} (1 + λ_i (y_i − p0_i)),   λ_i = min( λ_cap / p0_i , max(0, m̂_{i−1} / v̂_{i−1}) )

- m̂ and v̂ are running segment means of (y − p0) and (y − p0)² + 10⁻³ (predictable;
  aGRAPA-style plug-in).
- λ_cap = 0.5.
- **Reject H0** (K_n ≥ 1/α) → state ACTIVE.

**Deactivation (tests H0': "exchange is better than native by margin β_min"),** used while ACTIVE:

    H0': P(y_n = 1 | 𝓕_{n−1}) ≥ p1_n,   p1_n = σ( logit p0_n + β_min )
    K'_n = Π (1 + λ'_i (p1_i − y_i)),   λ'_i = min( λ_cap / (1 − p1_i), max(0, m̂'_{i−1} / v̂'_{i−1}) )

- **Reject H0'** (K'_n ≥ 1/α) → state SUPPRESSED.

**Settings:**
- α = 0.05; β_min = log 2 (odds ratio 2, the N1 default δ).
- Each e-process and its running means reset at every state change (a new segment).
- **Error control is per segment, not over the whole run.** This is stated in the report.

**Slots:**
- **ACTIVE:** all E slots run (§3); every outcome updates K'.
- **SUPPRESSED:** r_probe = 2 probe slots per iteration (hypothesis drawn from π̃, so evidence
  keeps accumulating); every probe outcome updates K. The other E − r_probe slots are **not
  executed**, and their FEs are handled per decision D-2.
- **Recording:** every exchange record (active or probe) also updates the outcome models,
  prequential losses and π.

### 12.4 Statement of what E3 does and does not guarantee

- **Guaranteed:** if the exchange success probability never exceeds the native-model reference
  p0_n, then the probability that a segment ever activates is ≤ α. This holds even though p0 is
  re-fitted between iterations (predictable).
- **Not guaranteed:**
  - validity if p_nat is biased for the native process at that step length (the reference is the
    model, not the truth);
  - power (it depends on how often exchange beats native at matched length);
  - run-level error over many segments.
- **Calibration test (Stage 3):** a harness in which exchange candidates are replaced by native-like
  proposals at matched length (H_null) or by proposals with a planted odds-ratio advantage
  (H_pos), reusing the S1/S3 functions. Required: false activation ≤ 2α. The result is reported
  even if it fails.

### 12.5 Compliance

E3 uses only outcomes already paid for (native offspring, FADs, executed exchanges, probes inside
the E cap). Its **extra FE count is 0** by construction (test T-FE). No step of the test requires an
evaluation that the backbone or the exchange budget would not make anyway.

---

## 13. Passive (shadow) evidence for diagnostics

These are computed in every run, consume no RNG and no FEs, and never influence decisions:
- π^{(H)} and π^{(M)}: §6–§9 recomputed on the subsets of records with donor population H, resp. M
  (own models, losses and weights). These are the donor-split structure estimates for H-E4.
- Native-displacement artefact matrices A^p_t (p ∈ {H, M}):
  - the off-diagonal correlation matrix of standardised **successful** native displacements S_t⁻¹u;
  - computed over the last 2W_A successful records, W_A = max(60, 5D) (the N1 window rule);
  - logged every U iterations.
- Trajectory-neutrality test T-SHADOW-SHE: a run with shadow evidence on equals the run with it off,
  bit for bit.

---

## 14. Hyperparameters and the selection rule (CEC2022 D=10 only)

| symbol | default | role | tuned? |
|---|---|---|---|
| p_x | 0.2 | per-unit transfer probability | **yes**: {0.1, 0.2, 0.4} |
| η | 1.0 | evidence temperature | **yes**: {0.5, 1.0, 2.0} |
| ρ | 0.995 | prequential forgetting per record | **yes**: {0.99, 0.995, 0.999} |
| π_min | 0.05 | selection floor | **yes**: {0.025, 0.05, 0.10} |
| U | 20 | eigenbasis/linkage refresh (iterations) | **yes**: {10, 20, 40} |
| λ_m | 0.995 | model-fit forgetting | no (fixed) |
| κ₂ | 1.0 | L2 penalty | no |
| W_x, W_L, W_n | 1000, 1000, 3000 | windows | no |
| q_e, λ_LW | 0.5, 0.2 | elite fraction, shrinkage | no |
| g_max | 5 | max linkage group | no |
| α, β_min, λ_cap, r_probe | 0.05, log 2, 0.5, 2 | E3 | no |
| ε_p, ȳ₀ | 10⁻⁴, 0.1 | numerical clip, initial intercept | no |

**Protocol.** Same as `configs/sensitivity.yaml`:
- **Data:** CEC2022 D = 10, all 12 functions, 5 runs, 100,000 FEs, seed master `she_sensitivity`
  (§15).
- **Variant tuned:** SHE-Full (with E3).
- **Design:** one-factor-at-a-time around the defaults.
- **Rule:** a non-default level replaces the default only if its average rank (ranks of mean error
  over 12 functions) is better by ≥ 0.5. All replacements are applied together; no further tuning.
- **Scope:** the resulting single configuration is used for every SHE variant, every domain
  (CEC2022 D=20, FIR, synthetic diagnostics after Stage 1) and every campaign.

**Ordering.** Stage 1 runs **with the defaults above**, because it precedes tuning. This is
pre-registered: Stage 1 tests the mechanism, not a tuned method. Tuning happens only after Stage 1
passes and before the smoke/pilot campaigns. The tuned values are logged before any D=20 or FIR SHE
run.

---

## 15. Seeds, streams and campaigns

### 15.1 Seed formula

`seed = master[campaign] + benchmark_offset[benchmark] + 1000·instance + run`, as in
`configs/reproducibility.yaml`. The synthetic instance index is 10·fid + (1 if D=10 else 2)
(N1 amendment A-6); the SYN offset is 200000.

### 15.2 New masters

These go in `configs/she.yaml`, the only place they are defined; the existing masters are not
reused. The values are disjoint from all existing masters (checked by test T-SEED).

| campaign | master | use |
|---|---|---|
| she_unit | 74000000 | unit tests |
| she_diag | 75000000 | Stage 1 synthetic diagnostic (30 runs) |
| she_e3calib | 76000000 | Stage 3 E3 calibration harness |
| she_sensitivity | 77000000 | CEC2022 D=10 tuning |
| she_smoke | 78000000 | smoke (5 runs) |
| she_pilot | 79000000 | pilot |
| main | 20260405 (existing) | main 30-run campaign |

The main campaign reuses the **existing** `main` master so that SHE runs are paired with
MPHBS/MPHB/HBA/MPA under the same seeds (MPHBS authors' master). External references are rerun
inside every SHE campaign under that campaign's master, so every comparison is paired.

### 15.3 RNG streams

- `rng_init = default_rng(seed)` initialises the populations (identical to MPHBS).
- `rng_native`, `rng_exchange`, `rng_select` = `SeedSequence([seed, 2]).spawn(3)`.
  - `rng_native`: backbone moves and FADs. The backbone receives `rng_native` after initialisation.
  - `rng_exchange`: donor, receiver and masks.
  - `rng_select`: hypothesis draws.
- The shadow/passive computations consume no RNG.

Consequence: variants that differ only in selection share native and exchange streams as far as
their trajectories coincide.

---

## 16. Pre-registered hypotheses

All tests are two-sided unless stated otherwise. α = 0.05. Holm correction within each named family.
Effect sizes are paired medians with 95% bootstrap CIs (10,000 resamples, seeded) and rank-biserial
correlation.

| id | prediction | data | measure | test | holds iff |
|---|---|---|---|---|---|
| **H-S1** identification | π concentrates on the true hypothesis: S1 → h_1, S2 → h_2, S3 → h_3 | Stage 1: SHE-Full without E3 (primary) and SHE-Uniform's passive π (secondary); S1–S3 × D ∈ {10, 20} × 30 seeds | π̄(h) = mean of π̃_n(h) over records after the first 10% of FEs. Contrast Δ = π̄(truth) − max_{h≠truth} π̄(h). **On S1** Δ₁ = π̄(h_1) − max(π̄(h_0), π̄(h_3)), with h_2 reported separately (h_2 nests h_1, so a tie is expected when E_t ≈ ∅) | one-sided Wilcoxon signed-rank Δ > 0 per cell; family = 6 cells | ≥ 5 of 6 cells significant **and** median Δ > 0 in all 6 |
| H-S1-CEC | on CEC2022 F1–F5 (dense rotations, per the benchmark definition) mean π̄(h_3) > π̄(h_1) | Stage 5 main, SHE-Full | π̄(h_3) − π̄(h_1) per run | one-sided Wilcoxon per function; family = 5 | ≥ 4 of 5 significant |
| **H-E4** premise | structure estimates from HBA-donor and MPA-donor exchanges agree, while native operator artefacts differ | Stage 1 runs (SHE-Full without E3) | (a) **agreement on S2, S3:** Δ computed separately from π^{(H)} and π^{(M)} (§13), both > 0; JSD(π̄^{(H)}, π̄^{(M)}) (base 2). (b) **artefact difference on S1**, where any off-diagonal correlation of successful native steps is an artefact: r = rmse_off(Ā^H, Ā^M) / sqrt(1/n_H + 1/n_M) (time-averaged A, noise-floor normalised) | (a) one-sided Wilcoxon Δ^{(H)} > 0 and Δ^{(M)} > 0 per cell (family = 8: 2 functions × 2 D × 2 sources); JSD median with CI. (b) one-sided Wilcoxon r > 2 per D (family = 2) | (a) ≥ 7 of 8 significant **and** the upper 95% CI of median JSD < 0.10 in all 4 cells; **and** (b) both significant |
| H-S2 decision relevance | evidence-based selection differs behaviourally from success-rate selection and is not worse in final quality | Stage 5 main, SHE-Full vs SHE-SuccessRate, paired seeds | **behaviour:** per run, the hypothesis-choice distribution in consecutive windows of 50 iterations; per instance, the mean over windows of JSD between the paired runs' window distributions. **Quality:** final error | (i) the per-instance median JSD exceeds its seed-permutation null (10,000 permutations of variant labels within pairs), one-sided, family = 20 instances; (ii) paired Wilcoxon on final error, family = 20 | (i) ≥ 50% of instances significant **and** (ii) no instance where SHE-Full is significantly worse |
| H-P performance (secondary) | at equal FEs SHE-Full is not worse than the best fixed-hypothesis variant chosen post hoc per domain | Stage 5 main | final error; average rank over instances per domain selects the best of SHE-h0..h3 | paired Wilcoxon per instance, family = instances per domain | no instance significantly worse. SHE vs MPHBS/MPHB/HBA/MPA is reported with the same test and **no hypothesis attached** |
| H-E3 calibration | E3 false activation ≤ 2α on the null harness | Stage 3 harness | fraction of null segments activating | binomial upper 95% CI | upper CI ≤ 0.10 |

---

## 17. Decision rules (pre-registered)

**Stage 1 gate.** STOP if **either**:
1. **π near uniform:** in ≥ 4 of the 6 H-S1 cells, the median over seeds of max_h π̄(h) < 0.35
   (uniform = 0.25); **or**
2. **H-E4 fails** (as defined in §16).

On STOP:
- write `results/SHE_DIAGNOSTIC_REPORT.md` with all effect sizes and CIs, recommending the pivot
  to R2 **and** recording the R2 conflict (§0.3);
- do not implement Stages 2–5.

If H-S1 fails but neither STOP condition holds, the gate result is reported as **"identification
not supported"** and the user decides. Nothing is auto-proceeded.

**Stage 3.** If H-E3 fails, E3 is reported as miscalibrated:
- SHE-Full keeps E3 (pre-registered);
- the "SHE without E3" variant is the reference for every E3 claim;
- E3 is **not** re-tuned.

**Stage 5.** Every hypothesis in §16 is reported as held or failed with its numbers. No redesign
after pilot or main results. The pilot is exploratory and is not used for tuning.

---

## 18. Variants and ablations (Stage 4; under `she_variants` in configs/ablation.yaml)

Every variant uses the identical backbone, FAD placement, E, donor/receiver rule and seeds.

| name | definition |
|---|---|
| SHE-Full | §3–§12 |
| SHE-h0 / -h1 / -h2 / -h3 | π̃ ≡ δ(h); evidence still computed passively |
| SHE-Uniform | π̃ ≡ uniform over H; evidence passive |
| SHE-SuccessRate (ACoS-like; equivalence to ACoS UNVERIFIED) | π_n(g) ∝ (a_g + 1)/(a_g + b_g + 2), the discounted Beta posterior mean of the success rate of exchanges **generated by g** (a, b discounted by ρ), with the same floor |
| SHE-NoShare | L_g accumulates only records generated by g, without weights; π uses the per-record mean loss scaled to a common effective count: π ∝ exp(−η · n̄ · L_g/n_g), with n_g = discounted count of g's records and n̄ = mean_g n_g |
| SHE-NoIW | w_e ≡ 1 |
| SHE-no-h2 / SHE-no-h3 | H without h_2, resp. h_3 (floor recomputed for \|H\| = 3) |
| SHE-h2-Random | E_t replaced at each refresh by a random graph with \|E_t\| equal to the learned graph's edge count in a shadow fit (placebo), drawn from `rng_select` |
| SHE-h2-FIRPrior (FIR only, **supplementary, labelled gray-box**) | E_t = pairs whose coupling in the FIR stopband quadratic form exceeds the same threshold rule as LA-KAIE's `fir_prior` |
| SHE-LikIntensity (MFEA-II-like; equivalence UNVERIFIED) | E3 replaced by an intensity r_t ∈ [0, 1]: each iteration, r_t maximises the likelihood of the last W_x exchange outcomes under y ~ Bernoulli(r·p̂_ex + (1 − r)·p0), with p̂_ex the pooled success frequency (1-D bounded scalar optimisation). The number of executed slots is round(r_t·E), with at least r_probe. **Our construction, labelled as such** |
| SHE-NoE3 | E3 removed: all E slots always run |
| externals | MPHBS (B-C1 CEC / A-C1 FIR as published), MPHB, HBA, MPA: existing code, configs unchanged |

**H-S2 uses behaviour, not only final error** (§16).

---

## 19. Unit tests (Stage 2/3)

| id | asserts |
|---|---|
| T-INV | T_h⁻¹(T_h(x)) = x (atol 10⁻⁹) for h_0–h_3, random B, μ |
| T-MASK | masks are non-empty; h_2 masks are unions of whole groups; h_0 = all ones; G_t partitions 1..D with max size ≤ g_max |
| T-LOSS | the recursive L_g equals the brute-force discounted weighted sum on a fixed sequence; predictions use pre-update θ |
| T-IRLS | one IRLS step on a small separable dataset matches a reference Newton step to 10⁻⁸ |
| T-FE | FE counter = MaxFE exactly; exchange FEs = number of evaluated (non-h_0) slots; E3, models, linkage and eigenbasis add 0 FEs |
| T-DET | same seed → bit-identical records; different seeds → different |
| T-SHADOW-SHE | shadow on/off → identical trajectories |
| T-EPROC | under simulated H0 data (y ~ Bernoulli(p0)), the empirical activation rate is ≤ α + MC error; K is non-negative |
| T-SEED | SHE masters are disjoint from all existing masters |
| T-FROZEN | the existing backbone sha256 freeze still passes |

Plus `pytest` and `pyflakes` clean over `lakaie scripts tests`.

---

## 20. Analysis plan (Stage 5)

**Statistics:** `scripts/analyze.py` and `make_figures.py` are extended additively. Paired Wilcoxon
with Holm per family; effect sizes as in §16. **Outputs:**
- convergence curves;
- π_t trajectories against the ground truth:
  - **CEC:** rotation density of M from the benchmark data files;
  - **FIR:** the coupling of the stopband quadratic form;
- exchange success rates; E3 suppression fraction and number of state changes;
- runtime overhead (wall time per run relative to MPHBS; SHE-internal timing split).

**Report:** `results/SHE_FINAL_REPORT.md` lists every §16 hypothesis as held or failed.

---

## 21. Decisions requiring user confirmation before Stage 1 code

| id | question | default pre-registered here | alternative |
|---|---|---|---|
| **D-1** | FAD placement | per-domain MPHBS placement (CEC/synthetic before, FIR after), identical for every SHE variant | one placement for all domains ("before", as LA-KAIE) |
| **D-2** | "fixed E" vs E3 suppression: what happens to the E − r_probe suppressed slots' FEs? | returned to the backbone (more native iterations; total = MaxFE). **E is the cap**, not a guaranteed spend. **This relaxes "fixed E" when E3 suppresses** | keep exactly E exchange FEs per iteration even when suppressed (suppression then only changes *which* candidates are evaluated, e.g. probes only, and the rest idle). **Not FE-neutral**, so rejected unless you prefer it |
| **D-3** | h_0 candidates are copies of the donor (outcome known without an FE) | 0 FE for h_0 slots; the unused FE goes to the backbone | evaluate anyway (wastes an FE re-evaluating a known point; MPHBS's note on avoiding re-evaluation) |
| **D-4** | E3 initial state | SUPPRESSED (literal reading of "suppress when the null is not rejected") | ACTIVE for a burn-in of 20 iterations, then the test |

---

## 22. Departures from NEXT_METHOD_DECISION.md §8–§11, with reasons

| item | §8 | here | reason |
|---|---|---|---|
| h_0 model | θ₀ + a‖z‖ + b·rankgap | θ₀ + a·log(1 + ‖z‖) | rankgap is a function of f_d − f_r, and for h_0 exchanges y = 1[f_d < f_r] exactly, so h_0 would win the evidence trivially regardless of structure |
| h_0 FE | 1 FE | 0 FE (D-3) | x' = x_d is an already-evaluated point |
| decision of exchange amount | not in §8 (fixed E) | E3 suppression (§12) | required by the user's Stage 3 |
| model update | "online" | one IRLS step per iteration on a discounted window | cost: per-exchange refits of four models would be O(E·W_x·p²) per iteration |
| test | Wald SPRT in N1 | betting e-process | the N1 SPRT calibration exceeded nominal α (§12.1) |

---

## 23. What this spec does NOT establish

- **No result exists.** H-S1, H-E4, H-S2, H-P and H-E3 are untested.
- **Novelty is not established;** all prior-art equivalences named here (ACoS-like, MFEA-II-like)
  are UNVERIFIED approximations.
- **The e-process validity statement is standard methodology [K]** and is not verified for this
  exact plug-in. Stage 3 checks it empirically.
- **Compute cost is estimated, not measured:**
  - IRLS: about 4 models × 1000 × (2D + 1)² per iteration;
  - FISTA: about 300 × 1000 × (D(D−1)/2 + 2D) every U iterations.
  - It may exceed the MPHBS mediator's cost; this will be measured, not assumed.

---

## 24. Amendment 1 (pre-registered 2026-10-05, after user decisions; before any SHE code or run)

### A-1 (D-1) FAD placement

Default confirmed:
- CEC2022 and the synthetic suite: *before* the exchange phase;
- FIR: *after* the exchange phase;
- identical for every SHE variant.

### A-2 (D-2) Exchange-FE regime per variant

**E is a cap only for the E3 variants** (SHE-Full, SHE-LikIntensity):
- slots not executed there return their FEs to native search;
- total FEs = MaxFE.

**Every other variant spends exactly E evaluated exchanges per iteration**, so the E3 effect is
isolated. These are:
- SHE-NoE3;
- SHE-h0 … SHE-h3;
- SHE-Uniform, SHE-SuccessRate, SHE-NoShare, SHE-NoIW;
- the leave-one-out variants;
- the h2 linkage-source variants;
- SHE-h0-eval.

The final partial iteration (§2) follows the same rule for all variants.

### A-3 (D-3) h0 accounting

This replaces the h0 paragraph of §3.3.

**h0 attempts are free records.** An h0 draw is executed at 0 FE (x' = x_d; y = 1[f_d < f_r]). Its
record (δ, y) enters the shared data and acceptance is applied. It does **not** consume one of the E
evaluated-exchange slots.

**Cap.** At most E h0 attempts per iteration. After that, h0 is excluded and draws are taken from
π̃ renormalised over the remaining hypotheses. This guarantees that the fixed-E variants spend
exactly E exchange FEs.

**Probes** (§12.3) must be evaluated exchanges: an h0 draw for a probe is redrawn among the non-h0
hypotheses.

**New variants:**
- **SHE-no-h0:** H = {h1, h2, h3}; floor recomputed for |H| = 3.
- **SHE-h0-eval:** every h0 attempt is evaluated (1 FE; a re-evaluation of the stored donor) and
  consumes a slot. Everything else as SHE-Full.
- **SHE-h0 (fixed):** under the fixed-E rule, all E slots per iteration are h0 attempts charged
  1 FE each, i.e. h0-eval semantics. Otherwise the variant would spend 0 exchange FEs. This is
  wasteful by design and stated as such.

**Reported per run:** the number of h0 attempts = **FEs saved by h0**, i.e. the FEs SHE-h0-eval
semantics would have spent.

### A-4 (D-4) E3 initial state and deadlock check

**Choice: option (a).** The initial state is SUPPRESSED, with the budget-counted probe rate that is
already in §12.3:
- r_probe = 2 evaluated, non-h0 exchanges per iteration while SUPPRESSED;
- each probe is charged 1 FE inside the E cap, and E − 2 slots return to native search.

**Deadlock analysis:**
- **Evidence never stops.** While SUPPRESSED, the activation e-process receives ≥ 2 outcomes per
  iteration, so the chain "suppressed → no exchange evidence → never rejects" cannot occur.
- **Iteration 1.** The e-process starts updating once a predictable reference exists: p_nat is first
  fitted at the end of iteration 1. Iteration-1 probes are recorded for the models but not added
  to K.
- **Expected activation time (plug-in approximation, not a guarantee).** With p0 = 0.10 and a true
  odds ratio of 2 (p = 0.182), the expected log-growth per outcome is ≈ KL = 0.029 nats. Reaching
  log(1/α) = 3.0 takes about 100 outcomes, i.e. about 50 iterations (≈ 3% of a D=20 run of ≈ 1,700
  iterations).
- **Weaker advantages give proportionally longer delays.** The Stage 3 harness reports the measured
  activation-time distribution.

**Alternative (b), a 20-iteration burn-in, is not adopted.** I'll show you this choice again before
Stage 3.

### A-5 η/ρ sensitivity sweep on the synthetic known-structure functions only

**Scope:**
- Functions S1, S2, S3 × D ∈ {10, 20}, with the 30 Stage 1 seeds (she_diag).
- CEC2022 and FIR are not touched.
- Levels: η ∈ {0.5, 1.0, 2.0} × ρ ∈ {0.99, 0.995, 0.999}, the full 3 × 3 grid.

**Data and method:**
- **Source:** the **passive π of the SHE-Uniform runs**.
- **Exact offline recomputation.** Under uniform selection the trajectory, the records, the
  per-record losses ℓ_{g,n} and the weights (w = 1/0.25) do not depend on (η, ρ). So π for every
  grid point is recomputed exactly from the logged ℓ_{g,n}, with **no extra runs and no FEs**.
- **SHE-Full feedback runs are not swept.** There η and ρ change the trajectory.

**Per grid point, report:**
- the H-S1 criterion of §16 (contrast Δ, one-sided Wilcoxon, Holm over 6 cells);
- the near-uniform statistic of §17 (median max_h π̄(h)).

**Decision rules (fixed in advance):**
1. **The Stage 1 gate (§17) is evaluated only at the defaults** (η = 1.0, ρ = 0.995):
   - primary: SHE-NoE3;
   - secondary, reported alongside: SHE-Uniform.
   The sweep can neither rescue nor fail the gate.
2. **Classification of the sweep:**
   - *robust*: H-S1 holds at ≥ 7 of 9 grid points;
   - *setting-sensitive*: it holds at 1–6;
   - *absent*: it holds at 0.
3. **If the gate fails and the sweep is setting-sensitive or robust,** the report says
   "identification may depend on evidence settings". **No default is changed:** defaults change
   only through the CEC2022 D=10 protocol (§14), and only after a gate pass.
4. **If the gate passes and the sweep is absent at non-default points,** this is reported as
   fragility.

### A-6 Stage 1 fallback assessment

If the gate fails, `results/SHE_DIAGNOSTIC_REPORT.md` assesses **both** fallbacks:
- **R2:** heterogeneous processes as FE-free structure estimators. Its closest formulation was rated
  C in `FINAL_LITERATURE_OVERLAP_GATE.md`.
- **F4:** an empirical study with structure diagnostics.

For each, the report states:
- which is defensible;
- what evidence supports it;
- what is unverified.

### A-7 Stage 1 implementation clarifications (no new design freedom)

**Runs.** Stage 1 runs SHE-NoE3 (primary; π-driven selection, exactly E evaluated exchanges per
iteration, h0 free per A-3) and SHE-Uniform:
- S1–S3 × D ∈ {10, 20} × 30 seeds;
- MaxFE = 10000·D; E = 24; FADs *before*; N = 30;
- seeds: she_diag master 75000000, SYN offset 200000, instance index 10·fid + (1 if D=10 else 2).

**Successful native displacements** (H-E4b artefact matrices) come from state snapshots taken
immediately before and after each backbone call:
- **HBA:** rows of X_H changed by `phase1`;
- **MPA population:** rows of X_M changed by `phase1` (moves retained by marine memory) and by
  `fads_greedy` (FAD successes). FAD is the MPA population's operator, as in D3.

Exchange replacements occur outside these calls and are therefore never counted as native.

**Burn-in.** π̄ excludes records generated in the first 10% of FEs (§16).
