# N1 concept — structural-hypothesis selection for information exchange between heterogeneous optimizers

Date: 2026-09-28. **Concept only.** No code, experiments, tuning or
manuscript text. Novelty status: **B (requires another literature check)**,
see `N1_NOVELTY_ASSESSMENT.md`. Nothing here was derived from, or adjusted
to, pilot performance numbers. The structural facts used (CEC2022 rotation
density, FIR coupling) come from the benchmark *definitions*.

---

## 1. Research problem

**Research question.** When two heterogeneous, model-free optimizers (here
HBA and MPA) run side by side, what structural model of the landscape
currently justifies transferring information between them? Can that be
decided *statistically*, from data the hybrid already generates (no extra
objective evaluations), including the possibility that **no** transfer
structure is supported?

**What is decided.** The decision object is the **structural hypothesis
governing exchange**. It is a statement about the problem ("which
coordinates can be moved from a donor to a receiver without destroying what
the receiver has achieved"), not a choice of operator, parameter or budget.
Once a hypothesis is selected, the exchange representation follows
deterministically (§5), and so do the operator and the amount of exchange.

**Why it is not simply:**

| framing | why N1 differs |
|---|---|
| operator selection (AOS) | operators are scored by their own success; N1 scores *hypotheses* by how well they explain data from both populations and from all exchanges |
| parameter adaptation | nothing continuous is tuned; the output is a discrete structural claim that can be checked against ground truth |
| variable-interaction / linkage learning | linkage learning *assumes* a sparse-linkage world and estimates it; N1 treats sparse linkage as one hypothesis among several, including "independent", "rotated" and "none" |
| coordinate transformation | a rotated frame is one hypothesis (H_R); N1 can reject it |
| evaluation allocation | the suppression decision follows from rejecting "exchange is informative" in a sequential test; the amount is not optimized |

## 2. Candidate structural hypotheses — analysis and minimal set

| proposed | assessment |
|---|---|
| H1 independent-coordinate exchange | **Genuine structural hypothesis:** the transfer-relevant structure is separable (the local Hessian / successful-step covariance is diagonal) |
| H2 grouped / linkage exchange | **Genuine:** block-diagonal structure under a permutation (disjoint groups, strong coupling within, none across) |
| H3 rotated / subspace exchange | **Genuine:** dense coupling that is separable only in an unknown orthogonal basis |
| H4 source-complementary exchange | **Not a structural hypothesis about the landscape.** It concerns *which population holds useful information* (the direction of transfer), which is orthogonal to *how* information can be transferred. Keeping it as a peer hypothesis would conflate two decision axes. It is removed from H and handled by a fixed, symmetric donor rule (§5). Source asymmetry is reported as a diagnostic, not selected |
| H5 exchange uninformative → suppress | **Genuine, and the sharpest:** the null hypothesis that cross-population displacements carry no more information than native search steps of the same size |

H1–H3 are **nested statistical models** of the successful-step covariance:

    diag (H_I)  ⊂  block-diag under partition 𝔅 (H_B)  ⊂  full (H_R),

with parameter counts D, Σ_b |b|(|b|+1)/2 and D(D+1)/2. Nesting makes an
information criterion the principled comparison: the richer model must earn
its extra parameters. They are therefore different structural *claims*, not
merely different operators. The operators of §5 are *consequences* of the
claims.

**Minimal hypothesis set:**

    𝓗 = { H_0 (no transferable structure),  H_I (independent),  H_B (block/linkage),  H_R (rotated/dense) }

with a two-stage decision: first H_0 vs "informative" (§4.2), then
H_I / H_B / H_R (§4.1).

## 3. Structure evidence (no future information, no extra FEs)

Each evidence source has a stated interpretation *for or against* specific
hypotheses. No generic feature vector is used.

### 3.1 Successful-step geometry, corroborated across heterogeneous optimizers (evidence for H_I / H_B / H_R)

For population p ∈ {HBA, MPA}, keep the W most recent **successful native
displacements**:
* HBA: u = x_new − x_old for offspring that replaced their parent;
* MPA: u = x_t − x_{t−1} for moves retained by marine memory.

Each is standardized per coordinate by the pooled scale s_t:
ũ = S_t⁻¹ u.

*Interpretation.* On a locally quadratic landscape, steps that are *accepted*
concentrate along low-curvature directions. Their covariance reflects the
inverse-Hessian structure (the rationale behind rank-μ covariance updates in
evolution strategies). Coupled variables produce correlated accepted steps;
separable landscapes produce a diagonal structure.

*Why heterogeneity matters (element E4).* The proposal distributions of HBA
(intensity-scaled moves towards its best with per-coordinate random factors)
and MPA (per-coordinate Brownian/Lévy steps) differ. Correlations created by
an operator's own proposal mechanism are unlikely to be *the same* in both
populations. Correlations created by the landscape through selection should
appear in both. The structural hypothesis is therefore scored on the **joint
likelihood of both populations' step sets under a shared structure with
population-specific parameters**. A structure supported by only one optimizer
is penalized.

### 3.2 Transfer-outcome predictiveness (evidence for H_I / H_B / H_R)

Every exchange e records the displacement δ_e = x'_e − x_{r,e} and the
outcome y_e = 1[f(x'_e) < f(x_{r,e})]. Under hypothesis k, a transfer should
succeed more often when δ_e is *typical* of successful steps under k's
covariance. The predictor is its Mahalanobis length
q_k(δ) = δ̃ᵀ Ω̂_k δ̃ (Ω̂_k the structured precision estimate, δ̃ = S⁻¹δ).

*Interpretation.* If H_R is true, a donor substitution that breaks the
rotated coupling produces a large q_R and fails. H_I would not predict that
failure. The hypothesis whose metric best predicts which transfers fail is
the better description of the *transfer-relevant* structure.

### 3.3 Exchange informativeness vs native control (evidence for or against H_0)

Native offspring (already evaluated) give a control relation between step
length and success: s_j = 1[native offspring improved] against ‖ũ_j‖.

*Interpretation.* If exchange outcomes are **not better** than native steps of
the same normalized length, cross-population information carries nothing
beyond what native search produces (evidence **for H_0**). This needs no
extra evaluations: the control data are the backbone's own offspring.

### 3.4 Evidence deliberately **not** used

Raw fitness values (scale- and shift-dependent), landscape features without a
structural interpretation, and future information. Stagnation and convergence
state enter only through the change-point rule (§4.3), never as evidence *for*
a structure.

## 4. Structural hypothesis test

### 4.1 Structure selection among H_I, H_B, H_R (stage 2)

**Geometry evidence** (information criterion; nested models):

    ℓ_t^{geo}(k) = Σ_{p∈{H,M}} Σ_{j∈W_t^p} log 𝒩( ũ_j ; 0, Σ̂_{k,t}^p ),
    BIC_t(k) = −2 ℓ_t^{geo}(k) + d_k log n_t,     n_t = |W_t^H| + |W_t^M|,

* Σ̂^p_{I} = diag of the sample covariance;
* Σ̂^p_{B} = block-diagonal restriction to 𝔅_t;
* Σ̂^p_{R} = Ledoit–Wolf-shrunk full covariance;
* d_k counts the parameters of **both** populations' covariances (2 × the
  per-population count). For H_B it also includes a term log|𝒫(𝔅_t)| for
  selecting the partition. The partition 𝔅_t is obtained deterministically by
  average-linkage clustering of the pooled absolute *partial* correlations
  |ρ_ij·rest| with threshold τ and maximum block size g, estimated on data
  **before** the scored window, so the penalty is honest.

**Transfer evidence** (prequential, i.e. predict-then-update, so no extra
complexity penalty is needed):

    p_{k,e} = σ( a_k^{(e−1)} + b_k^{(e−1)} · log(1 + q_k(δ_e)) ),
    T_t(k) = Σ_{e ≤ t} λ^{t−e} [ y_e log p_{k,e} + (1 − y_e) log(1 − p_{k,e}) ],

with (a_k, b_k) updated online by regularized logistic regression after each
outcome.

**Model score (log-evidence units):**

    M_t(k) = log π_0(k) − ½ BIC_t(k) + T_t(k),      k ∈ {I, B, R}.

This is a product of two likelihoods of disjoint data, geometry and outcomes,
treated as conditionally independent given k. A uniform prior π_0 is used.

### 4.2 Informativeness test H_0 vs H_A (stage 1): sequential probability ratio test

Native control model, fitted continuously on native offspring:
p_nat(s = 1 | ũ) = σ(c_0 + c_1 log‖ũ‖). For exchange e,

    p0_e = p_nat(δ̃_e)                           (H_0: exchange no better than native)
    p1_e = σ( logit p0_e + β_min ),  β_min > 0    (H_A: exchange better by a log-odds margin)
    Λ_t = Σ_{e since last decision} [ y_e log(p1_e/p0_e) + (1 − y_e) log((1 − p1_e)/(1 − p0_e)) ]

Wald boundaries: A = log((1 − β)/α), B = log(β/(1 − α)).
Λ_t ≥ A → **informative** (exchange active); Λ_t ≤ B → **H_0**
(suppress); otherwise continue sampling. Λ is reset at each decision.

### 4.3 Selected structure and reselection (change-point) rule

    H*_t = H_0                                        if the SPRT state is "suppressed"
    H*_t = argmax_{k∈{I,B,R}} M_t(k)                  otherwise, subject to hysteresis:
           switch from incumbent c to challenger k only if  M_t(k) − M_t(c) > κ   (log Bayes-factor threshold, e.g. κ = log 10)

Degradation detection: a Page–Hinkley test on the incumbent's per-exchange
log-loss ℓ_e = −log p_{c,e}(y_e):

    m_t = Σ_{e ≤ t} (ℓ_e − ℓ̄_t − ζ),   PH_t = m_t − min_{s ≤ t} m_s;   PH_t > h  ⇒  change-point

On a change point: truncate the windows W^p and the transfer history to the
data after the change point (forget stale evidence), reset Λ, and re-run the
selection. Forgetting λ < 1 in T_t also handles slow drift.

## 5. Information-exchange mapping X(H*)

Common to all hypotheses (fixed, so that **only the structure varies**):
* E exchanges per iteration when active;
* receiver/donor rule: alternate direction HBA→MPA and MPA→HBA each episode;
  receiver drawn rank-biased from its population, donor = the member of the
  other population with the nearest rank;
* transfer probability p_x per unit; greedy acceptance into the receiver;
  one FE per exchange.

With d = x_d − x_r:

    X(H_I):  m_i ~ Bernoulli(p_x), i = 1..D  (≥ 1 selected);           x' = x_r + m ⊙ d
    X(H_B):  m_b ~ Bernoulli(p_x), b ∈ 𝔅_t  (≥ 1);                       x' = x_r + Σ_b m_b P_b d
    X(H_R):  Û_t = eigenvectors of ½(Σ̂_R^H + Σ̂_R^M) (corroborated);  z = Û_tᵀ S_t⁻¹ d;
             m_k ~ Bernoulli(p_x);                                        x' = x_r + S_t Û_t (m ⊙ z)
    X(H_0):  no exchange; the E evaluations of the iteration are returned to native HBA/MPA search,
             except r_probe probe exchanges per iteration (representation drawn uniformly from {I, B, R})
             so that Λ and T can detect when exchange becomes informative again.

All candidates are clipped to the bounds. A small fraction ε_x of active
exchanges uses a non-selected representation, drawn uniformly. This keeps
the transfer evidence of competing hypotheses from becoming degenerate
(§4.1 scores all hypotheses on shared data).

## 6. Difference from MPHBS

| aspect | MPHBS | N1 |
|---|---|---|
| structural assumption | **fixed:** dimension-wise decomposition (per-dimension (rank, source) credit) | **inferred:** separable, block, rotated or none, from evidence |
| decision object | which component per dimension | which structural model governs transfer |
| learning mechanism | SARSA-style TD on Q(d, n, ch) with proxy screening | information criterion + prequential likelihood + SPRT + change-point test |
| "no exchange" | not available; the mediator always runs E = N_sub·N_i episodes | explicit null hypothesis with a statistical decision |
| configuration | domain-specific (B-C1 vs A-C1: FAD placement, K, N_i) | one configuration; structure adapts per instance and over time |
| use of heterogeneity | the two populations are component sources | also **independent witnesses** of landscape structure (corroborated geometry) |
| output | a solution | a solution **plus** an interpretable structural trace H*_t, checkable against ground truth |

MPHBS corresponds to fixing H*_t ≡ H_I with a different donor rule and no
null. The literature supports the distinction for this *specific* reference
method. It does **not** support claiming structure selection per se as new
(`N1_NOVELTY_ASSESSMENT.md` §4).

## 7. Literature search and 8. formal novelty test

See `N1_NOVELTY_ASSESSMENT.md`.

## 9. Mathematical specification (summary of symbols)

* **State** S_t = ( X_t^H, F_t^H, X_t^M, F_t^M, W_t^H, W_t^M, 𝓛_t, 𝓒_t, H*_{t−1}, Λ_t, PH_t ), where
  𝓛_t = {(δ_e, y_e, h_e)} is the exchange log and 𝓒_t = {(ũ_j, s_j)} the native control log.
* **Hypotheses** 𝓗 = {H_0, H_I, H_B, H_R}.
* **Evidence** E_t(H_k) = ( ℓ_t^{geo}(k), T_t(k) ) for k ∈ {I, B, R}; E_t(H_0) = Λ_t.
* **Score** M_t(k) = log π_0(k) − ½ BIC_t(k) + T_t(k).
* **Selection** H*_t: SPRT state → H_0; otherwise argmax M_t with hysteresis κ.
* **Mapping** X(H*_t): §5.
* **Updates:**
  * W^p: FIFO windows of successful native steps (size W);
  * Σ̂: recomputed every U iterations (or rank-1 updates);
  * (a_k, b_k): one regularized logistic Newton/SGD step per exchange;
  * T_t(k) ← λ T_{t−1}(k) + log-lik of y_t;
  * Λ_t ← Λ_{t−1} + LLR_t;
  * c_0, c_1 (native control): online logistic update per native offspring batch.
* **Reselection:** hysteresis κ; Page–Hinkley (ζ, h) on the incumbent's log-loss; window truncation at change points.
* **Hyperparameters** (fixed a priori, not tuned per instance): W, U, τ, g,
  λ, α, β, β_min, κ, ζ, h, p_x, E, r_probe, ε_x. A pre-registered
  sensitivity protocol on a disjoint tuning set would precede any main
  experiment.

## 10. Computational cost

* **Additional objective evaluations: 0.** Evidence comes from native
  offspring (already evaluated) and from exchange evaluations that are part
  of the fixed exchange budget. Probe exchanges during H_0 are drawn from the
  same budget. The FE counter is enforced as in the current code base.
* Per iteration (D dimensions, N per population, E exchanges, window W,
  refresh every U iterations):

  | step | time |
  |---|---|
  | collecting successful steps | O(N·D) |
  | covariance / partial-correlation refresh | O((W·D² + D³)/U) amortized (Cholesky of the full model; blocks cheaper) |
  | partition clustering | O(D³/U) (D ≤ 31 here) |
  | BIC for 3 models × 2 populations | O(W·D² + D³) per refresh |
  | per exchange: 3 Mahalanobis lengths + (for H_R) basis transform | O(D²) → **O(E·D²)** per iteration |
  | logistic / SPRT / Page–Hinkley updates | O(1) per exchange |

  Total overhead per iteration: O(N·D + E·D² + (W·D² + D³)/U). For D = 31,
  E = 24 that is on the order of 10⁴–10⁵ floating-point operations per
  iteration, far below the MPHBS mediator's K-candidate Python loops.
* **Memory:** O(W·D + D²) (two windows, three structured covariances,
  eigenbasis), plus O(1) per hypothesis for the outcome models.

## 11. Falsifiable hypotheses (pre-registered; none is "outperforms MPHBS")

| id | prediction | how it can fail |
|---|---|---|
| **P1 identification** | On synthetic functions with known structure (§12), the time-averaged selection frequency is highest for the true hypothesis: separable → H_I, block-separable → H_B, fully rotated → H_R. Identification accuracy > 1/3 (chance) and > a uniformly random selector | confusion matrix near uniform, or systematic misidentification |
| **P2 CEC2022 structure ordering** | Using rotation-matrix density from the benchmark *definition* (F1–F5 ≈ 0.80–0.84; F6–F12 ≈ 0.13–0.29 in the first block), the H_R selection share is higher on F1–F5 than on F6–F8 | H_R share not ordered by rotation density |
| **P3 rotated vs non-rotated** | On a rotated function and its unrotated counterpart (same base function), the selected representation differs (H_R vs H_I) | same selection for both |
| **P4 suppression** | On a synthetic *decoy* problem where the two populations are driven to different basins, H_0 is selected more often than on single-basin problems. In all runs, periods of H_0 are preceded by exchange success at or below the native control at matched step length | no relation between H_0 and relative exchange informativeness |
| **P5 corroboration** | Structure identification on the synthetic suite is more accurate with joint two-population evidence than with either population's evidence alone | single-population evidence equally accurate, which would falsify the claimed role of heterogeneity |
| **P6 evidence ≠ success credit** | The selection sequence under evidence-based scoring differs measurably from an ACoS-like rule that selects representations by their own success rate (divergence of choice distributions) | identical behaviour, i.e. the evidence machinery adds nothing |
| **P7 reselection** | On a *structure-switching* synthetic function (the structure changes between regions of the search space), the selected hypothesis changes after the population enters the new region, within a bounded latency | no switch, or chattering (switches without a structural change) |

Performance relative to MPHBS would be reported as a secondary outcome,
whatever it turns out to be.

## 12. Benchmark design

**Are CEC2022 and FIR sufficient?** No, not for testing the *mechanism*.
Their structural ground truth is only partially known (CEC2022: rotation
matrices are known, but multimodality and hybrid/composition construction
blur the local structure; FIR: the gray-box coupling of the stopband term is
known, the passband term's is not). With 12 + 8 instances they cannot
separate "identifies structure" from "happens to perform well". They remain
the application benchmarks for comparability with MPHBS.

**Recommended synthetic diagnostic suite** (not run; D ∈ {10, 20},
ground-truth structure known by construction):

| id | class | construction | ground truth |
|---|---|---|---|
| S1 | separable | ellipsoid Σ 10^{6(i−1)/(D−1)} x_i² | H_I |
| S2 | block-separable | block-diagonal rotations R = diag(R_1..R_{D/4}) of the ellipsoid, blocks of 4 | H_B (known partition) |
| S3 | fully rotated | ellipsoid composed with a dense random orthogonal R | H_R |
| S4 | partially separable | half the variables separable, half in a rotated block | H_B (mixed) |
| S5 | interaction-dense multimodal | rotated Rastrigin | H_R or H_0 (a robustness case; prediction registered as H_R share > H_I) |
| S6 | structure switching | separable ellipsoid far from the optimum, rotated near it (smooth blend by distance) | H_I → H_R over time (P7) |
| S7 | decoy / null | product of two separated basins with the populations initialized in different basins | H_0 (P4) |
| S8 | placebo | S1 with the coordinates randomly re-rotated **per evaluation** (no stable structure) | no consistent hypothesis; low identification confidence |

Metrics are **mechanism** metrics, not a performance table: selection
confusion matrix; time to correct identification; switching latency and
false-switch rate; suppression precision/recall relative to measured
exchange informativeness; agreement between the populations' geometry
evidence; overhead; FE use (must equal the budget).

## 13. Decision and alternatives

**Decision: B — promising but requires another literature check**
(`N1_NOVELTY_ASSESSMENT.md` §5 lists the seven items to verify with full
texts: LCC/LH-CC, ACoS, GOMEA FOS-type selection, dd-CMA-ES, statistical
transfer-suppression in multitasking, island/multi-population EDAs, KBS
2023–2026).

The defensible core, if verification succeeds, is the conjunction of:
**(E1)** a structural hypothesis governing *inter-population* transfer;
**(E2)** evidence-based selection;
**(E3)** a sequential null test of exchange informativeness against a
native-search control;
**(E4)** heterogeneous optimizers used as corroborating witnesses of
structure.
Without E3 and E4 the idea is incremental relative to BOA/ACoS/LCC.

**If verification downgrades N1 to C or D**, the recommendation is **not** to
modify N1 to look novel, but to pursue a different research question.
Candidates:

* **R1 — Predictability of exchange utility.** Can the utility of
  information exchange between heterogeneous optimizers be *predicted* from
  observable, FE-free population statistics, and which statistics carry that
  information? This is an explanatory-modelling study (knowledge
  extraction) with a falsifiable predictive claim, evaluated by
  out-of-instance prediction accuracy rather than optimization performance.
* **R2 — Heterogeneous search processes as structure estimators.** Does the
  *agreement or disagreement* between differently biased optimizers provide
  an unbiased, FE-free estimate of landscape structure (separability,
  coupling, rotation) that a single optimizer cannot provide? The
  contribution would be an estimator with statistical properties, validated
  on known-structure functions, and usable by any hybrid.

**STOP.** Awaiting your instruction.
